"""Orchestrator Ghost - Manages and coordinates worker agents."""

import asyncio
from typing import Any

import structlog

from common.models.messages import (
    AgentInfo,
    AgentRole,
    AgentStatus,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from common import BaseAgent
from common.logging.logger import get_logger
from common.inference import (
    TaskComplexity,
    create_classifier,
)
from common import get_executor_pool

logger = get_logger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Ghost that manages worker agents.

    Responsibilities:
    - Task distribution and routing
    - Worker health monitoring
    - Load balancing
    - Result aggregation
    """

    def __init__(self, agent_id: str | None = None) -> None:
        """
        Initialize the orchestrator agent with TaskMaster capabilities.

        Args:
            agent_id: Unique identifier for this orchestrator
        """
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.ORCHESTRATOR,
            capabilities=[
                "task_routing",
                "task_classification",  # TaskMaster capability
                "load_balancing",
                "worker_management",
                "result_aggregation",
            ],
            max_load=100,  # Can handle many concurrent orchestration tasks
            enable_gateway=True,  # Enable AgentGateway for inference routing
        )

        self.workers: dict[str, AgentInfo] = {}
        self.task_assignments: dict[str, str] = {}  # task_id -> worker_id
        self.pending_tasks: asyncio.Queue[TaskRequest] = asyncio.Queue()
        
        # TaskMaster: Classify task complexity for intelligent routing
        self.task_classifier = create_classifier()
        self.executor_pool = get_executor_pool()
        
        # Track routing decisions for analytics
        self.routing_stats = {
            "simple_tasks": 0,
            "moderate_tasks": 0,
            "complex_tasks": 0,
        }

        logger.info(
            "orchestrator_initialized",
            agent_id=self.agent_id,
            gateway_enabled=self.gateway_client is not None,
            task_master_enabled=True,
        )

    async def initialize(self) -> None:
        """Initialize orchestrator-specific resources."""
        logger.info("initializing_orchestrator", agent_id=self.agent_id)

        # Start worker discovery
        asyncio.create_task(self._discover_workers())

        # Start task distributor
        asyncio.create_task(self._distribute_tasks())

    async def cleanup(self) -> None:
        """Clean up orchestrator resources."""
        logger.info("cleaning_up_orchestrator", agent_id=self.agent_id)
        self.workers.clear()
        self.task_assignments.clear()

    async def process_task(self, task: TaskRequest) -> TaskResult:
        """
        Process orchestration tasks.

        Args:
            task: Task request

        Returns:
            Task result
        """
        import time

        start_time = time.time()

        try:
            if task.task_type == "distribute":
                # Add task to distribution queue
                await self.pending_tasks.put(task)

                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    result={"message": "Task queued for distribution"},
                    execution_time=time.time() - start_time,
                )

            elif task.task_type == "get_workers":
                # Return current worker status
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    result={
                        "workers": [w.model_dump() for w in self.workers.values()],
                        "total": len(self.workers),
                    },
                    execution_time=time.time() - start_time,
                )

            else:
                raise ValueError(f"Unknown orchestrator task type: {task.task_type}")

        except Exception as e:
            logger.error("orchestrator_task_failed", task_id=str(task.task_id), error=str(e))
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    async def _discover_workers(self) -> None:
        """Periodically discover and update worker information."""
        while True:
            try:
                # Broadcast heartbeat to discover workers
                await self._broadcast_heartbeat()

                # In a real implementation, workers would respond with their info
                # For now, this is a placeholder for the discovery mechanism

                await asyncio.sleep(10)  # Discovery interval

            except Exception as e:
                logger.error("worker_discovery_error", error=str(e))
                await asyncio.sleep(5)

    async def _distribute_tasks(self) -> None:
        """Distribute tasks to available workers with TaskMaster classification."""
        while True:
            try:
                # Get next task from queue
                task = await self.pending_tasks.get()
                
                # TaskMaster: Classify task complexity before routing
                await self.classify_and_route_task(task)

                # Find available worker
                worker = self._select_worker(task)

                if worker:
                    # Assign task to worker
                    await self.send_task(worker.agent_id, task)
                    self.task_assignments[str(task.task_id)] = worker.agent_id

                    logger.info(
                        "task_assigned",
                        task_id=str(task.task_id),
                        worker_id=worker.agent_id,
                        complexity=task.parameters.get("_complexity"),
                    )
                else:
                    # No available workers, re-queue
                    await self.pending_tasks.put(task)
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error("task_distribution_error", error=str(e))
                await asyncio.sleep(1)

    def _select_worker(self, task: TaskRequest) -> AgentInfo | None:
        """
        Select the best worker for a task using load balancing.

        Args:
            task: Task to assign

        Returns:
            Selected worker info or None if no workers available
        """
        available_workers = [
            worker
            for worker in self.workers.values()
            if worker.is_available and worker.status == AgentStatus.IDLE
        ]

        if not available_workers:
            return None

        # Simple load balancing: select worker with lowest load
        return min(available_workers, key=lambda w: w.current_load)
    
    async def classify_and_route_task(self, task: TaskRequest) -> dict[str, Any]:
        """
        TaskMaster: Classify task complexity and add routing metadata.
        
        This runs in the ExecutorPool for non-blocking classification,
        then adds complexity metadata to the task for workers to use.
        
        Args:
            task: Task to classify
            
        Returns:
            Dict with classification results and routing decision
        """
        # Extract prompt for classification
        prompt = task.description or task.parameters.get("prompt", "")
        
        # Classify complexity using executor pool (non-blocking)
        classification = await self.executor_pool.run_in_thread(
            self.task_classifier.classify,
            prompt,
        )
        
        # Update routing stats
        if classification.complexity == TaskComplexity.SIMPLE:
            self.routing_stats["simple_tasks"] += 1
        elif classification.complexity == TaskComplexity.MODERATE:
            self.routing_stats["moderate_tasks"] += 1
        else:
            self.routing_stats["complex_tasks"] += 1
        
        # Add complexity metadata to task parameters
        if not task.parameters:
            task.parameters = {}
        
        task.parameters["_complexity"] = classification.complexity.value
        task.parameters["_confidence"] = classification.confidence
        task.parameters["_taskmaster_classified"] = True
        
        # Decide routing strategy
        route_to_slm = classification.complexity in [
            TaskComplexity.SIMPLE,
            TaskComplexity.MODERATE,
        ]
        
        logger.info(
            "task_classified",
            task_id=str(task.task_id),
            complexity=classification.complexity.value,
            confidence=classification.confidence,
            route_to_slm=route_to_slm,
        )
        
        return {
            "complexity": classification.complexity.value,
            "confidence": classification.confidence,
            "route_to_slm": route_to_slm,
            "reasoning": classification.reasoning,
        }
    
    def get_routing_stats(self) -> dict[str, Any]:
        """Get TaskMaster routing statistics."""
        total = sum(self.routing_stats.values())
        
        return {
            **self.routing_stats,
            "total": total,
            "simple_percentage": (
                self.routing_stats["simple_tasks"] / total * 100
                if total > 0 else 0
            ),
            "complex_percentage": (
                self.routing_stats["complex_tasks"] / total * 100
                if total > 0 else 0
            ),
        }

    def register_worker(self, worker_info: AgentInfo) -> None:
        """
        Register a new worker.

        Args:
            worker_info: Worker information
        """
        self.workers[worker_info.agent_id] = worker_info
        logger.info(
            "worker_registered",
            worker_id=worker_info.agent_id,
            total_workers=len(self.workers),
        )

    def unregister_worker(self, worker_id: str) -> None:
        """
        Unregister a worker.

        Args:
            worker_id: Worker ID to unregister
        """
        if worker_id in self.workers:
            del self.workers[worker_id]
            logger.info(
                "worker_unregistered",
                worker_id=worker_id,
                total_workers=len(self.workers),
            )


async def main() -> None:
    """Main entry point for the orchestrator."""
    from common import configure_logging

    configure_logging()

    orchestrator = OrchestratorAgent()

    try:
        await orchestrator.start()
        # Keep running
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("shutdown_signal_received")
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())