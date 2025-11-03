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
        )

        self.settings = get_settings()
        self.anthropic_client: Anthropic | None = None

        logger.info("worker_initialized", agent_id=self.agent_id)

    async def initialize(self) -> None:
        """Initialize worker-specific resources."""
        logger.info("initializing_worker", agent_id=self.agent_id)

        # Initialize Anthropic client
        self.anthropic_client = Anthropic(api_key=self.settings.anthropic_api_key)

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
        Handle LLM inference task using Claude.

        Args:
            task: Task request

        Returns:
            LLM response
        """
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not initialized")

        prompt = task.parameters.get("prompt", "")
        model = task.parameters.get("model", "claude-sonnet-4-5-20250929")
        max_tokens = task.parameters.get("max_tokens", 1024)

        logger.debug(
            "llm_inference_request",
            task_id=str(task.task_id),
            model=model,
            prompt_length=len(prompt),
        )

        # Use asyncio to run the synchronous Anthropic call
        loop = asyncio.get_event_loop()
        message = await loop.run_in_executor(
            None,
            lambda: self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ),
        )

        response_text = ""
        for block in message.content:
            if hasattr(block, "text"):
                response_text += block.text

        return {
            "response": response_text,
            "model": model,
            "usage": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
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
    import signal

    configure_logging()

    worker = WorkerAgent()
    shutdown_event = asyncio.Event()

    def signal_handler() -> None:
        """Handle shutdown signals."""
        logger.info("shutdown_signal_received")
        shutdown_event.set()

    # Register signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await worker.start()
        logger.info("worker_running", agent_id=worker.agent_id)
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
    except Exception as e:
        logger.error("worker_error", error=str(e), exc_info=True)
    finally:
        logger.info("shutting_down_worker")
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())