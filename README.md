# Ghost Swarm 👻🐝

An AI agent system with true parallel processing, MCP server integration, and Agent2Agent communication framework.

## Overview

Ghost Swarm is a sophisticated multi-agent system built with Python 3.14 for true parallelization. It implements an orchestrator-worker pattern with advanced features:

- **Agent2Agent (A2A) Communication**: WebSocket-based idempotent RPC protocol
- **MCP Server Integration**: Full Model Context Protocol implementation with open-source LLM support
- **True Parallelization**: Python 3.14 features for concurrent agent execution
- **Router Pattern**: Intelligent task routing and distribution
- **Evaluator-Optimizer**: Validates outputs for logical fallacies and hallucinations
- **Pub/Sub Architecture**: Non-blocking continuous data ingestion
- **RAG Enhancement**: Context-aware agent operations
- **Multi-Provider LLM Support**: Use Claude, Ollama, OpenRouter, Groq, and more!

## Architecture

```
┌─────────────────┐
│  Orchestrator   │
│     Ghost       │
└────────┬────────┘
         │
    ┌────┴────┐
    │  Router │
    └────┬────┘
         │
    ┌────┴────────────────┐
    │                     │
┌───▼────┐         ┌─────▼───┐
│ Worker │◄────────┤Evaluator│
│ Ghost  │         │ Ghost   │
└────────┘         └─────────┘
```

## Project Structure

```
ghost-swarm/
├── ghosts/
│   ├── orchestrator/       # Orchestrator agent
│   ├── worker/            # Worker agents
│   └── ...               # Additional agent types
├── common/
│   ├── config/           # Configuration management
│   ├── logging/          # Structured logging
│   ├── communication/    # A2A protocol implementation
│   └── models/          # Shared data models
├── mcp_server/          # MCP server implementation
├── tests/               # Test suite
├── docs/                # Documentation
└── .env                 # Environment configuration
```

## Installation

### Prerequisites

- Python 3.13+ (configured for 3.14 compatibility)
- UV package manager
- Redis (for pub/sub)
- (Optional) Docker for containerized deployment

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ghost-swarm
```

2. Create and activate virtual environment with UV:
```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
uv pip install -e ".[dev]"
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Start Redis (for pub/sub):
```bash
docker run -d -p 6379:6379 redis:latest
# or use your local Redis installation
```

## Quick Start

### Start the A2A Server

```bash
python -m common.communication.a2a
```

### Start an Orchestrator

```bash
python -m ghosts.orchestrator.orchestrator
```

### Start Workers

```bash
# Terminal 1
python -m ghosts.worker.worker

# Terminal 2
python -m ghosts.worker.worker

# Terminal 3
python -m ghosts.worker.worker
```

## Configuration

All configuration is managed through environment variables in `.env`:

### AI Providers

```env
# Claude AI
ANTHROPIC_API_KEY=your-key-here

# OpenAI or compatible API
OPENAI_API_KEY=your-key-here
OPENAI_API_BASE=https://api.openai.com/v1
```

### Agent2Agent Communication

```env
A2A_HOST=0.0.0.0
A2A_PORT=8765
A2A_PROTOCOL=ws
```

### Worker Configuration

```env
MAX_WORKERS=4
WORKER_TIMEOUT=300
```

See `.env` for all available options.

## MCP Server & Open-Source LLMs

Ghost Swarm includes a full MCP (Model Context Protocol) server, allowing any LLM to interact with the agent swarm!

### Supported LLM Providers

- **Anthropic Claude** (your API key is configured)
- **Ollama** (free, local, private - recommended!)
- **OpenRouter** (100+ models, some free)
- **Together AI** (fast inference)
- **Groq** (ultra-fast, 500+ tokens/sec)
- **Any OpenAI-compatible API**

### Quick Start with Ollama (Free!)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2

# Start workers with Ollama
python -c "
from ghosts.worker.worker import WorkerAgent
import asyncio
asyncio.run(WorkerAgent(llm_provider='ollama').start())
"

# Use MCP client
python mcp_server/client.py mcp_server/server.py ollama llama3.2
```

### MCP Server Usage

```bash
# Start MCP server
python mcp_server/server.py

# Connect with any LLM provider
python mcp_server/client.py mcp_server/server.py [provider] [model]

# Examples:
python mcp_server/client.py mcp_server/server.py ollama llama3.2
python mcp_server/client.py mcp_server/server.py anthropic
python mcp_server/client.py mcp_server/server.py openrouter mixtral-8x7b
```

**Full MCP Guide**: See `docs/MCP_GUIDE.md` for complete documentation, cost comparisons, and best practices.

## Configuration

### Creating a Custom Ghost

```python
from common import BaseAgent, AgentRole, TaskRequest, TaskResult, TaskStatus

class MyCustomGhost(BaseAgent):
    def __init__(self):
        super().__init__(
            role=AgentRole.WORKER,
            capabilities=["custom_task"],
        )
    
    async def initialize(self):
        # Initialize resources
        pass
    
    async def cleanup(self):
        # Clean up resources
        pass
    
    async def process_task(self, task: TaskRequest) -> TaskResult:
        # Process the task
        result = {"message": "Task completed"}
        
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            result=result,
            execution_time=0.1,
        )
```

### Sending Tasks

```python
from common import TaskRequest
from uuid import uuid4

task = TaskRequest(
    task_id=uuid4(),
    task_type="llm_inference",
    description="Generate a summary",
    parameters={
        "prompt": "Summarize this text...",
        "max_tokens": 1024,
    },
)

await agent.send_task("worker-id", task)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_agent.py
```

## Development

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy .
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Docker Deployment

### Build Images

```bash
# Build orchestrator
docker build -t ghost-swarm-orchestrator -f Dockerfile.orchestrator .

# Build worker
docker build -t ghost-swarm-worker -f Dockerfile.worker .
```

### Run with Docker Compose

```bash
docker-compose up
```

## Roadmap

- [x] **MCP Server Integration** - ✅ Complete with open-source LLM support!
- [x] **Multi-Provider LLM Support** - ✅ Ollama, Claude, OpenRouter, Groq, Together AI
- [ ] Implement Router Ghost
- [ ] Implement Evaluator Ghost
- [ ] Implement Optimizer Ghost
- [ ] Add RAG capabilities with vector database
- [ ] Pub/Sub system integration with Redis
- [ ] Advanced monitoring and metrics
- [ ] Web UI for agent management
- [ ] Multi-stage Docker builds
- [ ] Kubernetes deployment manifests

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

[Your License Here]

## Acknowledgments

- [Google Agent2Agent Framework](https://github.com/a2aproject/A2A)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Anthropic Claude](https://www.anthropic.com/)

## MCP Integration 🔌

Ghost Swarm features **full MCP (Model Context Protocol) integration** using **Claude AI**:

### Built-in MCP Servers

- **Filesystem Server**: Secure file operations with access controls
- **Agent Management Server**: Coordinate Ghost agents via MCP
- Extensible architecture for custom servers

### Claude-Powered

Unlike most MCP implementations, Ghost Swarm uses:
- ✅ **Claude API** (superior reasoning with tools)
- ✅ **Open Source** (no vendor lock-in)
- ✅ **A2A Integration** (works with our Agent2Agent framework)

### Quick Example

```python
from mcp_server.filesystem import FilesystemMCPServer
from mcp_server.claude_client import ClaudeMCPClient

# Create MCP server
fs_server = FilesystemMCPServer(allowed_directories=["./data"])
await fs_server.setup()

# Use with Claude
claude = ClaudeMCPClient()
claude.register_mcp_server(fs_server)

# Claude automatically uses MCP tools
response = await claude.chat("List all Python files", use_tools=True)
```

### Documentation

- **[MCP Guide](docs/MCP_GUIDE.md)** - Complete usage guide
- **[MCP Implementation](docs/MCP_IMPLEMENTATION.md)** - Architecture details
- **[Examples](mcp_server/examples.py)** - Working code samples


## Scaling workers

docker-compose up --scale worker-1=5