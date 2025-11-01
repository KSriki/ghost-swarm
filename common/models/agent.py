"""Base agent class for all Ghost agents."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID, uuid4

import structlog

from common.communication.a2a import A2AClient
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

logger = structlog.get_logger(__name__)


class BaseAgent(ABC):
    """
    Base class for all Ghost agents.

    Provides common functionality for agent communication, task handling,
    and lifecycle management.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        role: AgentRole = AgentRole.WORKER,
        capabilities: list[str] | None = None,
        max_load: int = 10,
    ) -> None:
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier (generated if not provided)
            role: Agent role
            capabilities: List of capabilities this agent supports
            max_load: Maximum concurrent tasks
        """
        self.agent_id = agent_id or f"{role.value}-{uuid4().hex[:8]}"
        self.role = role
        self.status = AgentStatus.INITIALIZING
        self.capabilities = capabilities or []
        self.max_load = max_load
        self.current_load = 0

        self.a2a_client = A2AClient(self.agent_id)
        self.active_tasks: dict[UUID, asyncio.Task[Any]] = {}

        logger.info(
            "agent_initialized",
            agent_id=self.agent_id,
            role=role.value,
            capabilities=self.capabilities,
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
        )

    async def start(self) -> None:
        """Start the agent and connect to A2A network."""
        logger.info("starting_agent", agent_id=self.agent_id)

        try:
            await self.a2a_client.connect()
            self.status = AgentStatus.IDLE

            # Start message receiver
            asyncio.create_task(
                self.a2a_client.receive_messages(self._handle_incoming_message)
            )

            # Send heartbeat
            await self._send_heartbeat()

            # Run agent-specific initialization
            await self.initialize()

            logger.info("agent_started", agent_id=self.agent_id, status=self.status.value)
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error("agent_start_failed", agent_id=self.agent_id, error=str(e))
            raise

    async def stop(self) -> None:
        """Stop the agent and clean up resources."""
        logger.info("stopping_agent", agent_id=self.agent_id)

        # Cancel active tasks
        for task in self.active_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)

        # Disconnect from A2A
        await self.a2a_client.disconnect()
        self.status = AgentStatus.OFFLINE

        # Run agent-specific cleanup
        await self.cleanup()

        logger.info("agent_stopped", agent_id=self.agent_id)

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

    async def _handle_incoming_message(self, message: Message[Any]) -> None:
        """
        Handle incoming messages from other agents.

        Args:
            message: Incoming message
        """
        logger.debug(
            "received_message",
            agent_id=self.agent_id,
            message_type=message.type.value,
            sender=message.sender_id,
        )

        try:
            if message.type == MessageType.REQUEST:
                await self._handle_task_request(message)
            elif message.type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(message)
            else:
                logger.warning(
                    "unknown_message_type",
                    agent_id=self.agent_id,
                    message_type=message.type.value,
                )
        except Exception as e:
            logger.error(
                "message_handling_error",
                agent_id=self.agent_id,
                error=str(e),
                exc_info=True,
            )

    async def _handle_task_request(self, message: Message[dict[str, Any]]) -> None:
        """Handle a task request message."""
        if self.current_load >= self.max_load:
            await self._send_error(
                message,
                "Agent at maximum load",
            )
            return

        try:
            task = TaskRequest(**message.payload)

            # Update status
            self.current_load += 1
            if self.current_load > 0:
                self.status = AgentStatus.BUSY

            # Execute task asynchronously
            task_coroutine = self._execute_task(task, message.sender_id)
            task_future = asyncio.create_task(task_coroutine)
            self.active_tasks[task.task_id] = task_future

        except Exception as e:
            await self._send_error(message, str(e))
            self.current_load = max(0, self.current_load - 1)

    async def _execute_task(self, task: TaskRequest, requester_id: str) -> None:
        """Execute a task and send the result."""
        import time

        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                self.process_task(task),
                timeout=task.timeout,
            )
            execution_time = time.time() - start_time

            # Send result back
            response_message = Message[TaskResult](
                type=MessageType.RESPONSE,
                sender_id=self.agent_id,
                recipient_id=requester_id,
                correlation_id=task.task_id,
                payload=result,
            )

            await self.a2a_client.send_message(response_message)

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

            await self.a2a_client.send_message(response_message)

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

            await self.a2a_client.send_message(response_message)

        finally:
            # Clean up
            self.active_tasks.pop(task.task_id, None)
            self.current_load = max(0, self.current_load - 1)

            if self.current_load == 0:
                self.status = AgentStatus.IDLE

    async def _handle_heartbeat(self, message: Message[Any]) -> None:
        """Handle heartbeat message."""
        logger.debug("heartbeat_received", sender=message.sender_id)

    async def _send_heartbeat(self) -> None:
        """Send heartbeat message."""
        heartbeat_message = Message[AgentInfo](
            type=MessageType.HEARTBEAT,
            sender_id=self.agent_id,
            recipient_id=None,  # Broadcast
            payload=self.info,
        )

        await self.a2a_client.send_message(heartbeat_message)

    async def _send_error(self, original_message: Message[Any], error: str) -> None:
        """Send error response."""
        error_message = Message[dict[str, str]](
            type=MessageType.ERROR,
            sender_id=self.agent_id,
            recipient_id=original_message.sender_id,
            correlation_id=original_message.id,
            payload={"error": error},
        )

        await self.a2a_client.send_message(error_message)

    async def send_task(self, recipient_id: str, task: TaskRequest) -> None:
        """
        Send a task to another agent.

        Args:
            recipient_id: ID of the recipient agent
            task: Task to send
        """
        message = Message[TaskRequest](
            type=MessageType.REQUEST,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            payload=task,
        )

        await self.a2a_client.send_message(message)