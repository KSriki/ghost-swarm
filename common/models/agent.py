"""Refactored base agent class with integrated A2A server and hybrid inference."""

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog

from common.communication.a2a import A2AServer, A2AClient
from common.models.messages import (
    AgentInfo,
    AgentRole,
    AgentStatus,
    Message,
    MessageType,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
# Import from communication.concurrency (correct path for your structure)
from common.communication.concurrency import get_executor_pool, ExecutorPool

logger = structlog.get_logger(__name__)


class BaseAgent(ABC):
    """
    Refactored Base class for all Ghost agents.
    
    Each agent runs its own A2A server AND can connect to other agents.
    No central A2A hub needed - pure peer-to-peer communication.
    
    Includes:
    - Hybrid Inference (SLM + LLM routing)
    - ExecutorPool integration for non-blocking operations
    - Statistics tracking
    - Concurrency with uvloop
    """

    def __init__(
        self,
        agent_id: str | None = None,
        role: AgentRole = AgentRole.WORKER,
        capabilities: list[str] | None = None,
        max_load: int = 10,
        a2a_port: int | None = None,
        llm_client: Any | None = None,  # Claude/Anthropic client for hybrid inference
    ) -> None:
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier (generated if not provided)
            role: Agent role
            capabilities: List of capabilities this agent supports
            max_load: Maximum concurrent tasks
            a2a_port: Port for this agent's A2A server
            llm_client: LLM client (e.g., Anthropic) for hybrid inference
        """
        self.agent_id = agent_id or f"{role.value}-{uuid4().hex[:8]}"
        self.role = role
        self.status = AgentStatus.INITIALIZING
        self.capabilities = capabilities or []
        self.max_load = max_load
        self.current_load = 0
        
        # Each agent runs its own A2A server
        self.a2a_port = a2a_port or self._default_port_for_role(role)
        self.a2a_server = A2AServer()
        
        # Registry of other agents (discovered via Redis/service discovery)
        self.peer_agents: dict[str, str] = {}  # agent_id -> a2a_url
        
        # Client connections to other agents (lazy initialized)
        self.peer_connections: dict[str, A2AClient] = {}
        
        self.active_tasks: dict[UUID, asyncio.Task[Any]] = {}
        
        # Server control
        self.server_task: asyncio.Task[Any] | None = None
        self.shutdown_event = asyncio.Event()
        
        # ExecutorPool for non-blocking CPU/IO operations
        # Used by TaskMaster classification, inference, and other concurrent operations
        self.executor_pool: ExecutorPool = get_executor_pool()
        
        # Hybrid Inference Engine (SLM + LLM)
        # Each agent can now choose between SLM and LLM based on task complexity
        self.llm_client = llm_client
        self.inference_engine: Optional[Any] = None
        
        # Will be initialized in start() if llm_client provided
        if llm_client:
            self._init_inference_engine()

        logger.info(
            "agent_initialized",
            agent_id=self.agent_id,
            role=role.value,
            a2a_port=self.a2a_port,
            capabilities=self.capabilities,
            hybrid_inference=self.inference_engine is not None,
            executor_pool_workers=f"{self.executor_pool.max_process_workers}p/{self.executor_pool.max_thread_workers}t",
        )

    def _default_port_for_role(self, role: AgentRole) -> int:
        """Get default A2A port for agent role."""
        port_map = {
            AgentRole.ORCHESTRATOR: 8765,
            AgentRole.WORKER: 8766,
            AgentRole.ROUTER: 8767,
            AgentRole.EVALUATOR: 8768,
            AgentRole.OPTIMIZER: 8769,
        }
        return port_map.get(role, 8770)
    
    def _init_inference_engine(self) -> None:
        """Initialize hybrid inference engine."""
        try:
            from common.inference import HybridInferenceEngine
            
            self.inference_engine = HybridInferenceEngine(
                llm_client=self.llm_client,
                slm_provider=os.getenv("SLM_PROVIDER", "llama_cpp"),
                slm_endpoint=os.getenv("SLM_ENDPOINT"),
                enable_fallback=os.getenv("ENABLE_SLM_FALLBACK", "true").lower() == "true",
            )
            
            logger.info(
                "hybrid_inference_initialized",
                agent_id=self.agent_id,
                slm_provider=self.inference_engine.slm_provider.provider_type.value,
            )
        except ImportError as e:
            logger.warning(
                "hybrid_inference_unavailable",
                agent_id=self.agent_id,
                error=str(e),
                message="Inference module not found. Agent will not support hybrid inference.",
            )
        except Exception as e:
            logger.error(
                "hybrid_inference_init_failed",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True,
            )

    @property
    def info(self) -> AgentInfo:
        """Get current agent information."""
        return AgentInfo(
            agent_id=self.agent_id,
            role=self.role,
            status=self.status,
            capabilities=self.capabilities,
            current_load=self.current_load,
            max_load=self.max_load,
            metadata={
                "a2a_port": self.a2a_port,
                "a2a_url": f"ws://{self.agent_id}:{self.a2a_port}",
                "hybrid_inference": self.inference_engine is not None,
            },
        )

    async def start(self) -> None:
        """Start the agent and its A2A server."""
        logger.info("starting_agent", agent_id=self.agent_id)

        try:
            # Register message handlers with our A2A server
            self.a2a_server.register_handler(
                MessageType.REQUEST,
                self._handle_incoming_request,
            )
            self.a2a_server.register_handler(
                MessageType.HEARTBEAT,
                self._handle_heartbeat,
            )
            
            # Start our A2A server (non-blocking) and track the task
            self.server_task = asyncio.create_task(self._run_a2a_server())
            
            # Wait a bit for server to start
            await asyncio.sleep(1)
            
            self.status = AgentStatus.IDLE

            # Discover other agents with retry logic
            await self._discover_peers()

            # Send initial heartbeat (with retry)
            await asyncio.sleep(2)  # Extra wait for orchestrator to be ready
            await self._broadcast_heartbeat()

            # Run agent-specific initialization
            await self.initialize()

            logger.info("agent_started", agent_id=self.agent_id, status=self.status.value)
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error("agent_start_failed", agent_id=self.agent_id, error=str(e))
            raise

    async def _run_a2a_server(self) -> None:
        """Run this agent's A2A server in a non-blocking way."""
        import websockets
        
        try:
            logger.info(
                "starting_a2a_server",
                agent_id=self.agent_id,
                port=self.a2a_port,
            )
            
            # Start server on 0.0.0.0 so it's accessible from other containers
            async with websockets.serve(
                self.a2a_server.handle_client,
                "0.0.0.0",
                self.a2a_port,
            ):
                logger.info(
                    "a2a_server_listening",
                    agent_id=self.agent_id,
                    port=self.a2a_port,
                )
                
                # Wait for shutdown signal
                await self.shutdown_event.wait()
                
                logger.info("a2a_server_shutting_down", agent_id=self.agent_id)
            
            logger.info("a2a_server_closed", agent_id=self.agent_id)
            
        except asyncio.CancelledError:
            logger.info("a2a_server_cancelled", agent_id=self.agent_id)
            raise
        except Exception as e:
            logger.error(
                "a2a_server_error",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True,
            )
            raise

    async def _discover_peers(self) -> None:
        """
        Discover other agents in the swarm.
        
        In Docker, we use service names:
        - ghost-orchestrator:8765
        - ghost-worker-1:8766, ghost-worker-2:8766, ghost-worker-3:8766
        
        In production, this would query Redis or a service registry.
        """
        try:
            from common.config.settings import get_settings
            settings = get_settings()
        except Exception as e:
            logger.warning(
                "settings_unavailable",
                agent_id=self.agent_id,
                error=str(e),
                message="Using default settings",
            )
            # Fallback to default
            class DefaultSettings:
                max_workers = 3
            settings = DefaultSettings()
        
        # Discover based on role
        if self.role == AgentRole.WORKER:
            # Workers need to know about orchestrator
            self.peer_agents["orchestrator"] = "ws://ghost-orchestrator:8765"
        
        elif self.role == AgentRole.ORCHESTRATOR:
            # Orchestrator discovers workers
            # In Docker: ghost-worker-1, ghost-worker-2, ghost-worker-3
            # TODO: Dynamic discovery via Redis
            for i in range(1, settings.max_workers + 1):
                worker_id = f"worker-{i}"
                self.peer_agents[worker_id] = f"ws://ghost-worker-{i}:8766"
        
        logger.info(
            "peers_discovered",
            agent_id=self.agent_id,
            peer_count=len(self.peer_agents),
            peers=list(self.peer_agents.keys()),
        )

    async def _get_peer_connection(self, peer_id: str) -> A2AClient:
        """Get or create connection to a peer agent with retry logic."""
        if peer_id not in self.peer_connections:
            # Create new connection with retry
            peer_url = self.peer_agents.get(peer_id)
            if not peer_url:
                raise ValueError(f"Unknown peer: {peer_id}")
            
            client = A2AClient(self.agent_id)
            
            # Retry connection up to 3 times
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await client.connect(url=peer_url)
                    self.peer_connections[peer_id] = client
                    logger.info(
                        "peer_connection_established",
                        agent_id=self.agent_id,
                        peer_id=peer_id,
                        attempt=attempt + 1,
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            "peer_connection_retry",
                            agent_id=self.agent_id,
                            peer_id=peer_id,
                            attempt=attempt + 1,
                            error=str(e),
                        )
                        await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                    else:
                        logger.error(
                            "peer_connection_failed",
                            agent_id=self.agent_id,
                            peer_id=peer_id,
                            error=str(e),
                        )
                        raise
        
        return self.peer_connections[peer_id]

    async def stop(self) -> None:
        """Stop the agent and clean up resources."""
        logger.info("stopping_agent", agent_id=self.agent_id)

        # Signal server to shutdown
        self.shutdown_event.set()
        
        # Wait for server to close
        if self.server_task:
            try:
                await asyncio.wait_for(self.server_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("server_shutdown_timeout", agent_id=self.agent_id)
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass

        # Cancel active tasks
        for task in self.active_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)

        # Close peer connections
        for client in self.peer_connections.values():
            try:
                await client.disconnect()
            except Exception as e:
                logger.warning("peer_disconnect_error", error=str(e))

        self.status = AgentStatus.OFFLINE

        # Run agent-specific cleanup
        await self.cleanup()
        
        # Cleanup executor pool (only if last agent)
        # Note: ExecutorPool is shared globally, so we don't shut it down per-agent
        # It will be cleaned up on process exit
        # If needed, call: await self.executor_pool.shutdown()

        logger.info("agent_stopped", agent_id=self.agent_id)
    
    async def infer(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        force_llm: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Perform hybrid inference (SLM or LLM based on task complexity).
        
        This is the main method agents use for LLM calls. It automatically
        routes to SLM for simple tasks and LLM for complex tasks.
        
        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            force_llm: Force use of LLM regardless of complexity
            **kwargs: Additional model-specific parameters
            
        Returns:
            HybridInferenceResult with content and metadata
            
        Example:
            result = await self.infer("Parse this JSON: {...}")
            # Will use SLM for simple parsing
            
            result = await self.infer("Analyze the strategic implications...")
            # Will use LLM for complex reasoning
        """
        if not self.inference_engine:
            raise RuntimeError(
                f"Hybrid inference not enabled for agent {self.agent_id}. "
                "Provide llm_client during initialization."
            )
        
        return await self.inference_engine.infer(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            force_llm=force_llm,
            **kwargs,
        )
    
    def get_inference_stats(self) -> dict[str, Any]:
        """
        Get inference statistics (SLM vs LLM usage, costs, latency, etc.).
        
        Returns:
            Dict with statistics or empty dict if inference not enabled
        """
        if not self.inference_engine:
            return {}
        
        return self.inference_engine.get_stats()

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize agent-specific resources. Override in subclasses."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up agent-specific resources. Override in subclasses."""
        pass

    @abstractmethod
    async def process_task(self, task: TaskRequest) -> TaskResult:
        """
        Process a task. Must be implemented by subclasses.

        Args:
            task: Task to process

        Returns:
            Task result
        """
        pass

    async def _handle_incoming_request(self, message: Message[Any]) -> dict[str, Any]:
        """Handle incoming task request."""
        try:
            if self.current_load >= self.max_load:
                return {
                    "status": "error",
                    "error": "Agent at maximum load",
                }

            task = TaskRequest(**message.payload)

            # Update status
            self.current_load += 1
            if self.current_load > 0:
                self.status = AgentStatus.BUSY

            # Execute task asynchronously
            task_coroutine = self._execute_task(task, message.sender_id)
            task_future = asyncio.create_task(task_coroutine)
            self.active_tasks[task.task_id] = task_future

            return {"status": "accepted", "task_id": str(task.task_id)}

        except Exception as e:
            logger.error("request_handling_error", error=str(e), exc_info=True)
            return {"status": "error", "error": str(e)}

    async def _execute_task(self, task: TaskRequest, requester_id: str) -> None:
        """Execute a task and send the result back to requester."""
        import time

        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                self.process_task(task),
                timeout=task.timeout,
            )
            execution_time = time.time() - start_time

            # Send result back to requester
            response_message = Message[TaskResult](
                type=MessageType.RESPONSE,
                sender_id=self.agent_id,
                recipient_id=requester_id,
                correlation_id=task.task_id,
                payload=result,
            )

            # Get connection to requester and send
            client = await self._get_peer_connection(requester_id)
            await client.send_message(response_message)

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error="Task timeout",
                execution_time=execution_time,
            )

            response_message = Message[TaskResult](
                type=MessageType.ERROR,
                sender_id=self.agent_id,
                recipient_id=requester_id,
                correlation_id=task.task_id,
                payload=error_result,
            )

            client = await self._get_peer_connection(requester_id)
            await client.send_message(response_message)

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("task_execution_error", task_id=str(task.task_id), error=str(e))

            error_result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
            )

            response_message = Message[TaskResult](
                type=MessageType.ERROR,
                sender_id=self.agent_id,
                recipient_id=requester_id,
                correlation_id=task.task_id,
                payload=error_result,
            )

            try:
                client = await self._get_peer_connection(requester_id)
                await client.send_message(response_message)
            except Exception as send_error:
                logger.error("error_sending_failed", error=str(send_error))

        finally:
            # Clean up
            self.active_tasks.pop(task.task_id, None)
            self.current_load = max(0, self.current_load - 1)

            if self.current_load == 0:
                self.status = AgentStatus.IDLE

    async def _handle_heartbeat(self, message: Message[Any]) -> dict[str, Any]:
        """Handle heartbeat message from peer."""
        logger.debug("heartbeat_received", sender=message.sender_id)
        
        # Update peer registry if new peer
        if message.sender_id not in self.peer_agents:
            # Extract peer URL from metadata
            peer_info = message.payload
            if isinstance(peer_info, dict) and "a2a_url" in peer_info:
                self.peer_agents[message.sender_id] = peer_info["a2a_url"]
                logger.info("new_peer_discovered", peer_id=message.sender_id)
        
        return {"status": "ok"}

    async def _broadcast_heartbeat(self) -> None:
        """Broadcast heartbeat to all known peers."""
        heartbeat_message = Message[dict[str, Any]](
            type=MessageType.HEARTBEAT,
            sender_id=self.agent_id,
            recipient_id=None,  # Broadcast
            payload={
                "agent_info": self.info.model_dump(),
                "a2a_url": f"ws://{self.agent_id}:{self.a2a_port}",
            },
        )

        for peer_id in list(self.peer_agents.keys()):
            try:
                client = await self._get_peer_connection(peer_id)
                await client.send_message(heartbeat_message)
            except Exception as e:
                logger.warning(
                    "heartbeat_failed",
                    peer_id=peer_id,
                    error=str(e),
                )

    async def send_task(self, recipient_id: str, task: TaskRequest) -> None:
        """
        Send a task to another agent.

        Args:
            recipient_id: ID of the recipient agent
            task: Task to send
        """
        message = Message[dict[str, Any]](
            type=MessageType.REQUEST,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            payload=task.model_dump(),
        )

        client = await self._get_peer_connection(recipient_id)
        await client.send_message(message)
        
        logger.info(
            "task_sent",
            task_id=str(task.task_id),
            recipient=recipient_id,
        )