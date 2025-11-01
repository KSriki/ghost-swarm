"""Common data models for agent communication and operations."""

from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of messages that can be exchanged between agents."""

    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class AgentRole(str, Enum):
    """Roles that agents can assume."""

    ORCHESTRATOR = "orchestrator"
    WORKER = "worker"
    ROUTER = "router"
    EVALUATOR = "evaluator"
    OPTIMIZER = "optimizer"


class TaskStatus(str, Enum):
    """Status of a task."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(str, Enum):
    """Status of an agent."""

    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


T = TypeVar("T")


class Message(BaseModel, Generic[T]):
    """Base message structure for agent communication."""

    id: UUID = Field(default_factory=uuid4)
    type: MessageType
    sender_id: str
    recipient_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: UUID | None = None
    payload: T
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class TaskRequest(BaseModel):
    """Request to execute a task."""

    task_id: UUID = Field(default_factory=uuid4)
    task_type: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)
    timeout: int = Field(default=300, ge=1)
    context: dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    """Result of a task execution."""

    task_id: UUID
    status: TaskStatus
    result: Any | None = None
    error: str | None = None
    execution_time: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentInfo(BaseModel):
    """Information about an agent."""

    agent_id: str
    role: AgentRole
    status: AgentStatus
    capabilities: list[str] = Field(default_factory=list)
    current_load: int = Field(default=0, ge=0)
    max_load: int = Field(default=10, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        """Check if agent is available for work."""
        return self.status == AgentStatus.IDLE and self.current_load < self.max_load


class EvaluationResult(BaseModel):
    """Result of an evaluation/validation check."""

    is_valid: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGContext(BaseModel):
    """RAG-enhanced context for agent operations."""

    query: str
    retrieved_documents: list[dict[str, Any]] = Field(default_factory=list)
    relevance_scores: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)