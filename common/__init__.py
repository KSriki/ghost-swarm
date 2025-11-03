"""Common utilities and shared code for Ghost Swarm."""

# Communication
from common.communication.a2a import A2AClient, A2AProtocol, A2AServer

# Configuration
from common.config.settings import Settings, get_settings, reload_settings

# Logging
from common.logging.logger import configure_logging, get_logger

# Models - Agent
from common.models.agent import BaseAgent

# Models - Messages
from common.models.messages import (
    AgentInfo,
    AgentRole,
    AgentStatus,
    EvaluationResult,
    Message,
    MessageType,
    RAGContext,
    TaskRequest,
    TaskResult,
    TaskStatus,
)

from common.communication.concurrency import (
    ExecutorPool,
    get_executor_pool,
    run_cpu_bound,
    run_io_bound,
    cleanup_executors,
)

__version__ = "0.1.0"

__all__ = [
    # Agent
    "BaseAgent",
    # Communication
    "A2AClient",
    "A2AProtocol",
    "A2AServer",
    # Configuration
    "Settings",
    "get_settings",
    "reload_settings",
    # Logging
    "configure_logging",
    "get_logger",
    # Models
    "AgentInfo",
    "AgentRole",
    "AgentStatus",
    "EvaluationResult",
    "Message",
    "MessageType",
    "RAGContext",
    "TaskRequest",
    "TaskResult",
    "TaskStatus",
    # Concurrency
    "ExecutorPool",
    "get_executor_pool",
    # Task and Future Management
    "run_cpu_bound",
    "run_io_bound",
    "cleanup_executors",
]