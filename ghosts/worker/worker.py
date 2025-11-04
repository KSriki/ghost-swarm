"""Worker Ghost - Executes tasks assigned by the orchestrator."""

import asyncio
from typing import Any

import structlog
from anthropic import Anthropic

from common.models.messages import (
    AgentRole,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from common.models.agent import BaseAgent
from common.logging.logger import get_logger
from common.config.settings import get_settings

logger = get_logger(__name__)


class WorkerAgent(BaseAgent):
    """
    Worker Ghost that executes tasks.

    Responsibilities:
    - Execute assigned tasks
    - Report results back to orchestrator
    - Maintain task execution state
    """

    def __init__(
        self,
        agent_id: str | None = None,
        capabilities: list[str] | None = None,
    ) -> None:
        """
        Initialize the worker agent.

        Args:
            agent_id: Unique identifier for this worker
            capabilities: List of task types this worker can handle
        """
        self.settings = get_settings()
        
        # Initialize Anthropic client for hybrid inference
        anthropic_client = Anthropic(api_key=self.settings.anthropic_api_key)
        
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.WORKER,
            capabilities=capabilities
            or [
                "llm_inference",
                "data_processing",
                "analysis",
                "generation",
            ],
            max_load=5,  # Can handle 5 concurrent tasks
            llm_client=anthropic_client,  # Enable hybrid inference
        )

        self.anthropic_client = anthropic_client

        logger.info(
            "worker_initialized",
            agent_id=self.agent_id,
            hybrid_inference=self.inference_engine is not None,
        )

    async def initialize(self) -> None:
        """Initialize worker-specific resources."""
        logger.info("initializing_worker", agent_id=self.agent_id)

        # Anthropic client already initialized in __init__ for hybrid inference
        # Register with orchestrator (in a real system)
        # For now, this is a placeholder
        await self._register_with_orchestrator()

    async def cleanup(self) -> None:
        """Clean up worker resources."""
        logger.info("cleaning_up_worker", agent_id=self.agent_id)
        self.anthropic_client = None

    async def process_task(self, task: TaskRequest) -> TaskResult:
        """
        Process a task assigned to this worker.

        Args:
            task: Task request

        Returns:
            Task result
        """
        import time

        start_time = time.time()

        try:
            logger.info(
                "processing_task",
                task_id=str(task.task_id),
                task_type=task.task_type,
                agent_id=self.agent_id,
            )

            # Route to appropriate handler based on task type
            if task.task_type == "llm_inference":
                result = await self._handle_llm_inference(task)
            elif task.task_type == "data_processing":
                result = await self._handle_data_processing(task)
            elif task.task_type == "analysis":
                result = await self._handle_analysis(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            execution_time = time.time() - start_time

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                metadata={
                    "agent_id": self.agent_id,
                    "task_type": task.task_type,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "task_processing_error",
                task_id=str(task.task_id),
                error=str(e),
                exc_info=True,
            )

            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
            )

    async def _handle_llm_inference(self, task: TaskRequest) -> dict[str, Any]:
        """
        Handle LLM inference task using HYBRID inference (SLM or LLM).
        
        This now automatically routes between SLM and LLM based on
        task complexity. Simple tasks use SLM, complex tasks use LLM.

        Args:
            task: Task request

        Returns:
            LLM/SLM response
        """
        if not self.inference_engine:
            raise RuntimeError("Hybrid inference not enabled")

        prompt = task.parameters.get("prompt", "")
        system = task.parameters.get("system")
        max_tokens = task.parameters.get("max_tokens", 1024)
        temperature = task.parameters.get("temperature", 0.7)
        force_llm = task.parameters.get("force_llm", False)

        logger.debug(
            "llm_inference_request",
            task_id=str(task.task_id),
            prompt_length=len(prompt),
            force_llm=force_llm,
        )

        # Use hybrid inference engine - automatically routes to SLM or LLM
        result = await self.infer(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            force_llm=force_llm,
        )

        logger.info(
            "inference_completed",
            task_id=str(task.task_id),
            provider=result.provider.value,
            model=result.model,
            tokens=result.tokens_used,
            latency_ms=result.latency_ms,
            complexity=result.complexity.value,
            fallback=result.fallback_occurred,
        )

        return {
            "response": result.content,
            "provider": result.provider.value,
            "model": result.model,
            "complexity": result.complexity.value,
            "latency_ms": result.latency_ms,
            "usage": {
                "tokens": result.tokens_used,
            },
            "fallback_occurred": result.fallback_occurred,
        }

    async def _handle_data_processing(self, task: TaskRequest) -> dict[str, Any]:
        """
        Handle data processing task.

        Args:
            task: Task request

        Returns:
            Processed data
        """
        # Placeholder for data processing logic
        data = task.parameters.get("data", [])

        logger.debug(
            "data_processing_request",
            task_id=str(task.task_id),
            data_size=len(data),
        )

        # Simulate processing
        await asyncio.sleep(0.1)

        return {
            "processed": True,
            "count": len(data),
            "message": "Data processed successfully",
        }

    async def _handle_analysis(self, task: TaskRequest) -> dict[str, Any]:
        """
        Handle analysis task.

        Args:
            task: Task request

        Returns:
            Analysis results
        """
        # Placeholder for analysis logic
        data = task.parameters.get("data", {})

        logger.debug(
            "analysis_request",
            task_id=str(task.task_id),
        )

        # Simulate analysis
        await asyncio.sleep(0.1)

        return {
            "analyzed": True,
            "insights": ["Insight 1", "Insight 2"],
            "confidence": 0.85,
        }

    async def _register_with_orchestrator(self) -> None:
        """Register this worker with the orchestrator."""
        # In a real implementation, this would send a registration message
        # to the orchestrator via A2A
        logger.info(
            "registered_with_orchestrator",
            agent_id=self.agent_id,
            capabilities=self.capabilities,
        )


async def main() -> None:
    """Main entry point for the worker."""
    from common import configure_logging

    configure_logging()

    import os
    worker = WorkerAgent(agent_id=os.getenv("AGENT_ID"))

    try:
        await worker.start()
        # Keep running
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("shutdown_signal_received")
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())