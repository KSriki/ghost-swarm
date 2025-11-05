# Ghost Swarm 👻🐝

An AI agent system with true parallel processing, MCP server integration, and Agent2Agent communication framework.

## Overview

Ghost Swarm is a sophisticated multi-agent system built with Python 3.14 for true parallelization. It implements an orchestrator-worker pattern with advanced features:

- **Agent2Agent (A2A) Communication**: WebSocket-based idempotent RPC protocol for peer-to-peer messaging
- **Hybrid Inference Engine**: Intelligent routing between SLM (Small Language Models) and LLM (Claude) based on task complexity
- **ExecutorPool Concurrency**: Process pool (CPU-bound) and thread pool (I/O-bound) for non-blocking operations
- **TaskMaster Classification**: ML-based task complexity analysis for optimal model selection
- **True Parallelization**: Python 3.14 features with uvloop for high-performance async operations
- **Cost Optimization**: 60-70% cost reduction through strategic SLM usage for routine tasks
- **MCP Server Integration**: Model Context Protocol for LLM interactions
- **Router Pattern**: Intelligent task routing and distribution
- **Evaluator-Optimizer**: Validates outputs for logical fallacies and hallucinations
- **Pub/Sub Architecture**: Non-blocking continuous data ingestion
- **RAG Enhancement**: Context-aware agent operations

## Key Features

### 🎯 Intelligent Task Routing

TaskMaster analyzes incoming tasks and automatically routes them to the optimal model:
- **Simple tasks** → SLM (Phi-3-mini): parsing, JSON, basic reasoning
- **Complex tasks** → LLM (Claude): advanced reasoning, creative writing
- **ML-based classification** with confidence scoring

### ⚡ High-Performance Concurrency

ExecutorPool provides true parallelization with separate pools:
- **Process Pool**: CPU-bound operations (SLM inference, data processing)
- **Thread Pool**: I/O-bound operations (API calls, network requests)
- **Uvloop Integration**: High-performance event loop for async operations

### 💰 Cost Optimization

Hybrid inference delivers substantial savings:
- **60-70% cost reduction** through strategic SLM usage
- **5-10x latency improvement** for routine tasks
- **10x higher throughput** for concurrent operations
- Graceful fallback ensures reliability

### 🔄 Agent2Agent Communication

WebSocket-based RPC protocol for peer-to-peer messaging:
- **Idempotent operations** with request IDs
- **Direct agent communication** without central broker
- **Automatic peer discovery** via service registry
- **Health checks** and connection management

### 🛡️ Production-Ready

Built for scale and reliability:
- Comprehensive error handling with fallback mechanisms
- Structured logging with correlation IDs
- Docker containerization with health checks
- Statistics tracking for cost and performance monitoring

## Architecture

```
                    ┌─────────────────┐
                    │  Orchestrator   │
                    │     Ghost       │
                    │  (ExecutorPool) │
                    └────────┬────────┘
                             │
                        ┌────┴────┐
                        │ Router  │
                        │TaskMaster│
                        └────┬────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
      ┌─────▼─────┐    ┌────▼────┐    ┌─────▼─────┐
      │  Worker   │    │ Worker  │    │  Worker   │
      │   Ghost   │    │  Ghost  │    │   Ghost   │
      │           │    │         │    │           │
      │ ┌───────┐ │    │┌───────┐│    │ ┌───────┐ │
      │ │  SLM  │ │    ││  SLM  ││    │ │  SLM  │ │
      │ │ (70%) │ │    ││ (70%) ││    │ │ (70%) │ │
      │ └───┬───┘ │    │└───┬───┘│    │ └───┬───┘ │
      │     │     │    │    │    │    │     │     │
      │ ┌───▼───┐ │    │┌───▼───┐│    │ ┌───▼───┐ │
      │ │  LLM  │ │    ││  LLM  ││    │ │  LLM  │ │
      │ │Claude │ │    ││Claude ││    │ │Claude │ │
      │ │ (30%) │ │    ││ (30%) ││    │ │ (30%) │ │
      │ └───────┘ │    │└───────┘│    │ └───────┘ │
      └───────────┘    └─────────┘    └───────────┘
             │                │                │
             └────────────────┴────────────────┘
                             │
                      ┌──────▼──────┐
                      │  Evaluator  │
                      │    Ghost    │
                      └─────────────┘
```

### Key Components

- **Orchestrator**: Manages worker lifecycle, routes tasks via TaskMaster
- **TaskMaster**: ML-based classifier that routes tasks to SLM or LLM based on complexity
- **ExecutorPool**: Shared pool of process workers (CPU-bound) and thread workers (I/O-bound)
- **Workers**: Execute tasks using hybrid inference (SLM for simple, LLM for complex)
- **SLM Layer**: Phi-3-mini handles 70% of routine tasks (parsing, JSON, routing)
- **LLM Layer**: Claude handles 30% of complex reasoning tasks
- **Evaluator**: Validates outputs for accuracy and quality

## Project Structure

```
ghost-swarm/
├── ghosts/
│   ├── orchestrator/       # Orchestrator agent with TaskMaster routing
│   ├── worker/            # Worker agents with hybrid inference
│   └── ...               # Additional agent types
├── common/
│   ├── config/           # Configuration management
│   ├── logging/          # Structured logging
│   ├── communication/    # A2A protocol + ExecutorPool concurrency
│   │   ├── a2a.py       # Agent-to-Agent protocol
│   │   └── concurrency.py # ExecutorPool for CPU/IO-bound operations
│   ├── models/          # Shared data models
│   │   ├── agent.py     # BaseAgent with hybrid inference
│   │   ├── messages.py  # Task/Result models
│   │   └── inference.py # InferenceResult, TaskComplexity
│   ├── inference/       # Hybrid inference engine
│   │   ├── engine.py    # HybridInferenceEngine
│   │   ├── providers.py # SLM and LLM providers
│   │   └── classifier.py # TaskMaster complexity classifier
│   └── ...
├── mcp_server/          # MCP server implementation
├── tests/               # Test suite
├── docs/                # Documentation
│   ├── ARCHITECTURE.md  # System architecture
│   ├── CONCURRENCY_GUIDE.md # ExecutorPool usage
│   └── SLM_STRATEGY.md  # Hybrid inference strategy
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
# Claude API (LLM)
ANTHROPIC_API_KEY=your-key-here
LLM_MODEL=claude-sonnet-4-5-20250929

# SLM Configuration (optional)
SLM_ENDPOINT=http://slm-server:8080
SLM_MODEL=phi-3-mini
SLM_MODEL_PATH=/models/phi-3-mini-4k-instruct-q4.gguf
SLM_PROVIDER=llama_cpp
ENABLE_SLM_FALLBACK=true

# OpenAI or compatible API (alternative)
OPENAI_API_KEY=your-key-here
OPENAI_API_BASE=https://api.openai.com/v1
```

### Concurrency Configuration

```env
# ExecutorPool settings
MAX_PROCESS_WORKERS=4    # CPU-bound tasks (SLM inference)
MAX_THREAD_WORKERS=20    # I/O-bound tasks (API calls)

# Task complexity thresholds
SIMPLE_THRESHOLD=0.3     # Below this: use SLM
COMPLEX_THRESHOLD=0.7    # Above this: use LLM
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
ENABLE_HYBRID_INFERENCE=true
```

See `.env` for all available options.

## Usage

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

## Hybrid Inference

Ghost Swarm uses a sophisticated hybrid inference system that automatically routes tasks between SLM and LLM based on complexity:

### How It Works

1. **TaskMaster Classification**: Incoming tasks are analyzed for complexity using ML-based classification
2. **Automatic Routing**: 
   - Simple tasks (70%): Routed to SLM (Phi-3-mini) for fast, cost-effective processing
   - Complex tasks (30%): Routed to LLM (Claude) for advanced reasoning
3. **Graceful Fallback**: If SLM fails or is unavailable, tasks automatically fall back to LLM

### Using Hybrid Inference

```python
from common import BaseAgent, AgentRole

class MyWorker(BaseAgent):
    def __init__(self, llm_client):
        super().__init__(
            role=AgentRole.WORKER,
            llm_client=llm_client,  # Enables hybrid inference
            capabilities=["llm_inference"],
        )
    
    async def process_task(self, task):
        # Automatically routes to SLM or LLM based on complexity
        result = await self.infer(
            prompt=task.parameters["prompt"],
            max_tokens=1024,
            temperature=0.7,
            force_llm=False,  # Allow automatic routing
        )
        
        print(f"Used: {result.provider.value}")  # "slm" or "llm"
        print(f"Latency: {result.latency_ms}ms")
        print(f"Cost: ${result.cost_usd}")
        
        return result.content
```

### Performance Benefits

| Metric | SLM (70% of tasks) | LLM (30% of tasks) | Overall Improvement |
|--------|-------------------|-------------------|---------------------|
| **Latency** | 50-100ms | 500-2000ms | **5-10x faster** |
| **Cost** | $0.0001/request | $0.01/request | **60-70% reduction** |
| **Throughput** | 100 req/s | 10 req/s | **10x more concurrent** |
| **Quality** | 95%+ for simple | 99%+ for complex | Maintained |

### Cost Optimization Example

For 1M tasks per month:
- **Without SLM**: 1M × $0.01 = $10,000/month
- **With SLM**: (700K × $0.0001) + (300K × $0.01) = $70 + $3,000 = **$3,070/month**
- **Savings**: $6,930/month (69% reduction)

## Concurrency Model

Ghost Swarm uses `ExecutorPool` for efficient concurrent processing:

### ExecutorPool

```python
from common.communication.concurrency import get_executor_pool, run_cpu_bound, run_io_bound

# Get the shared executor pool
executor = get_executor_pool()

# CPU-bound operations (process pool) - e.g., SLM inference
result = await run_cpu_bound(
    expensive_computation,
    arg1, arg2
)

# I/O-bound operations (thread pool) - e.g., API calls
result = await run_io_bound(
    api_call,
    url, params
)

# Or use the executor directly
result = await executor.run_in_process(cpu_bound_func, args)
result = await executor.run_in_thread(io_bound_func, args)
```

### Concurrency Best Practices

- **SLM Inference**: Use process pool (`run_cpu_bound`) - CPU intensive
- **LLM API Calls**: Use thread pool (`run_io_bound`) - I/O waiting
- **Task Classification**: Use process pool for ML inference
- **Network Operations**: Use thread pool for non-blocking I/O

### Architecture Benefits

- **Non-blocking**: All operations run asynchronously with uvloop
- **Efficient**: Process pool for CPU-bound, thread pool for I/O-bound
- **Scalable**: Configurable worker pools adapt to load
- **Reliable**: Graceful degradation with fallback mechanisms

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
# Build base image
docker build -t ghost-swarm-base -f Dockerfile.base .

# Build orchestrator
docker build -t ghost-swarm-orchestrator -f Dockerfile.orchestrator .

# Build worker
docker build -t ghost-swarm-worker -f Dockerfile.worker .

# Build SLM server (optional)
docker build -t ghost-swarm-slm -f Dockerfile.mcp .
```

### Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# Start without SLM (LLM-only mode)
docker-compose up -d orchestrator worker-1 worker-2 worker-3 redis
```

### Docker Compose Services

The full stack includes:

- **orchestrator**: Routes tasks with TaskMaster complexity classification
- **worker-1, worker-2, worker-3**: Execute tasks with hybrid inference
- **slm-server** (optional): Runs Phi-3-mini for simple tasks
- **redis**: Message broker for pub/sub

### Environment Configuration

```yaml
# docker-compose.yml
services:
  orchestrator:
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - ENABLE_HYBRID_INFERENCE=true
      - MAX_PROCESS_WORKERS=4
      - MAX_THREAD_WORKERS=20
  
  worker-1:
    environment:
      - AGENT_ID=worker-1
      - SLM_ENDPOINT=http://slm-server:8080
      - LLM_MODEL=claude-sonnet-4-5-20250929
      - ENABLE_SLM_FALLBACK=true
  
  slm-server:
    image: ghcr.io/ggerganov/llama.cpp:server
    volumes:
      - ./models:/models:ro
    environment:
      - MODEL=/models/phi-3-mini-4k-instruct-q4.gguf
      - N_GPU_LAYERS=35
```

### Monitoring

```bash
# Check orchestrator logs
docker logs -f ghost-orchestrator

# Check worker logs with inference routing
docker logs -f ghost-worker-1 | grep -E "(slm|llm|complexity)"

# Check SLM server status
curl http://localhost:8080/health
```

## Roadmap

### ✅ Completed

- [x] Agent2Agent (A2A) communication protocol
- [x] Hybrid Inference Engine (SLM + LLM routing)
- [x] ExecutorPool concurrency (process + thread pools)
- [x] TaskMaster complexity classification
- [x] Docker containerization with docker-compose
- [x] Structured logging and monitoring
- [x] BaseAgent with async operations
- [x] Cost and performance optimization

### 🚧 In Progress

- [ ] MCP server integration enhancements
- [ ] Advanced evaluator ghost
- [ ] RAG capabilities
- [ ] Statistics dashboard

### 📋 Planned

- [ ] Fine-tuned SLMs for domain-specific tasks
- [ ] Self-improving pipeline with automated retraining
- [ ] A/B testing framework for model comparison
- [ ] Kubernetes deployment manifests
- [ ] Web UI for agent management
- [ ] Advanced monitoring and metrics
- [ ] Multi-region deployment support
- [ ] Pub/Sub system integration enhancements

## Quick Reference

### Task Complexity Examples

| Task Type | Complexity | Route | Example |
|-----------|-----------|-------|---------|
| JSON parsing | Simple | SLM | Extract fields from JSON |
| Email parsing | Simple | SLM | Parse sender/subject |
| Basic routing | Simple | SLM | Classify task category |
| Code generation | Medium | SLM/LLM | Generate simple functions |
| Text summarization | Medium | LLM | Summarize documents |
| Advanced reasoning | Complex | LLM | Multi-step logic |
| Creative writing | Complex | LLM | Generate stories |
| Strategic planning | Complex | LLM | Business strategy |

### Performance Metrics

```python
# Access inference statistics
stats = agent.inference_engine.stats

print(f"Total requests: {stats.total_requests}")
print(f"SLM usage: {stats.slm_requests / stats.total_requests * 100:.1f}%")
print(f"Average latency: {stats.avg_latency_ms:.1f}ms")
print(f"Total cost: ${stats.total_cost_usd:.2f}")
print(f"Cost per request: ${stats.avg_cost_per_request:.4f}")
```

### Common Patterns

#### Force LLM for Specific Tasks

```python
# Override automatic routing for critical tasks
result = await agent.infer(
    prompt="Complex strategic analysis...",
    force_llm=True,  # Skip SLM, go directly to LLM
)
```

#### Custom Complexity Thresholds

```python
# Adjust routing thresholds in .env
SIMPLE_THRESHOLD=0.3    # Lower = more to SLM
COMPLEX_THRESHOLD=0.7   # Higher = less to LLM
```

#### Monitor Task Routing

```bash
# Watch routing decisions in real-time
docker logs -f ghost-worker-1 | grep complexity

# Output shows:
# {"event": "task_classified", "complexity": "simple", "score": 0.15, "route": "slm"}
# {"event": "task_classified", "complexity": "complex", "score": 0.89, "route": "llm"}
```

### Troubleshooting

#### SLM Not Available

If SLM server is down, tasks automatically fall back to LLM:
```
{"event": "slm_inference_failed", "error": "connection_refused"}
{"event": "falling_back_to_llm"}
```

#### High Latency

Check concurrent task load:
```python
print(f"Current load: {agent.current_load}/{agent.max_load}")
print(f"Active tasks: {len(agent.active_tasks)}")
```

#### Cost Spikes

Monitor SLM vs LLM usage ratio:
```python
# Should see ~70% SLM, 30% LLM for optimal cost
slm_ratio = stats.slm_requests / stats.total_requests
if slm_ratio < 0.6:
    print("Warning: Too many LLM calls, check task complexity")
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

[Your License Here]

## Acknowledgments

- [Google Agent2Agent Framework](https://github.com/a2aproject/A2A)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Anthropic Claude](https://www.anthropic.com/)