# Project Structure

```
ghost-swarm/
│
├── .env                          # Environment configuration (API keys, settings)
├── .gitignore                    # Git ignore patterns
├── pyproject.toml               # Project dependencies and configuration
├── README.md                    # Main documentation
├── main.py                      # Entry point
├── install.sh                   # Installation script
│
├── common/                      # Shared library
│   ├── __init__.py             # Package exports
│   ├── config/                 # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py         # Pydantic settings
│   ├── logging/                # Logging setup
│   │   ├── __init__.py
│   │   └── logger.py           # Structured logging
│   ├── communication/          # Agent2Agent protocol
│   │   ├── __init__.py
│   │   └── a2a.py             # A2A server/client implementation
│   └── models/                 # Shared data models
│       ├── __init__.py
│       ├── messages.py         # Message models
│       └── agent.py            # Base agent class
│
├── ghosts/                     # Agent implementations
│   ├── __init__.py
│   ├── orchestrator/          # Orchestrator agent
│   │   ├── __init__.py
│   │   └── orchestrator.py    # Main orchestrator implementation
│   └── worker/                # Worker agents
│       ├── __init__.py
│       └── worker.py          # Worker implementation
│
├── mcp_server/                # MCP server (future)
│   └── __init__.py
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_agent.py         # Agent tests
│   └── test_a2a.py           # A2A communication tests
│
└── docs/                      # Documentation
    └── architecture/
        └── ARCHITECTURE.md    # System architecture

```

## Key Files

### Configuration & Setup

- **`.env`**: Contains all environment variables including API keys
- **`pyproject.toml`**: Python project metadata and dependencies
- **`install.sh`**: Automated installation script

### Common Library

- **`common/config/settings.py`**: Type-safe configuration management using Pydantic
- **`common/logging/logger.py`**: Structured logging with JSON output
- **`common/communication/a2a.py`**: Agent2Agent WebSocket protocol
- **`common/models/agent.py`**: Base agent class with lifecycle management
- **`common/models/messages.py`**: Data models for inter-agent communication

### Agent Implementations

- **`ghosts/orchestrator/orchestrator.py`**: 
  - Task distribution
  - Worker management
  - Load balancing
  - Result aggregation

- **`ghosts/worker/worker.py`**:
  - Task execution
  - LLM inference
  - Data processing
  - Result reporting

## Module Dependencies

```
main.py
  └── common/
      ├── config/settings
      └── logging/logger

ghosts/orchestrator/orchestrator.py
  └── common/
      ├── models/agent (BaseAgent)
      ├── models/messages (TaskRequest, TaskResult)
      └── logging/logger

ghosts/worker/worker.py
  └── common/
      ├── models/agent (BaseAgent)
      ├── models/messages (TaskRequest, TaskResult)
      ├── config/settings
      └── logging/logger

common/models/agent.py
  └── common/
      ├── communication/a2a (A2AClient)
      └── models/messages (Message, AgentInfo, TaskRequest)

common/communication/a2a.py
  └── common/
      ├── config/settings
      ├── models/messages (Message)
      └── logging/logger
```

## Data Flow

```
1. Configuration Loading:
   .env → settings.py → Application

2. Agent Startup:
   Agent → A2AClient → A2AServer → WebSocket Connection

3. Task Processing:
   Orchestrator → TaskRequest → Worker → LLM → TaskResult → Orchestrator

4. Logging:
   Agent → structlog → JSON/Console Output
```

## File Purposes

### Core Files

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `common/config/settings.py` | Configuration | `Settings`, `get_settings()` |
| `common/logging/logger.py` | Logging setup | `configure_logging()`, `get_logger()` |
| `common/communication/a2a.py` | A2A protocol | `A2AServer`, `A2AClient`, `A2AProtocol` |
| `common/models/messages.py` | Data models | `Message`, `TaskRequest`, `TaskResult`, `AgentInfo` |
| `common/models/agent.py` | Base agent | `BaseAgent` |

### Agent Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `ghosts/orchestrator/orchestrator.py` | Orchestration | `OrchestratorAgent` |
| `ghosts/worker/worker.py` | Task execution | `WorkerAgent` |

### Support Files

| File | Purpose |
|------|---------|
| `.env` | Environment variables and secrets |
| `pyproject.toml` | Project metadata and dependencies |
| `install.sh` | Automated setup |
| `main.py` | Application entry point |

## Adding New Components

### Adding a New Agent Type

1. Create directory: `ghosts/new_agent/`
2. Create `__init__.py`
3. Create `new_agent.py`:
   ```python
   from common import BaseAgent, AgentRole
   
   class NewAgent(BaseAgent):
       def __init__(self):
           super().__init__(role=AgentRole.WORKER)
       
       async def initialize(self):
           pass
       
       async def cleanup(self):
           pass
       
       async def process_task(self, task):
           # Implementation
           pass
   ```

### Adding New Message Types

1. Add to `MessageType` enum in `common/models/messages.py`
2. Create corresponding handler in agents
3. Update tests

### Adding Configuration Options

1. Add field to `Settings` class in `common/config/settings.py`
2. Add to `.env` file
3. Document in README.md