# Ghost Swarm Quick Reference 🚀

## Start the System

```bash
# 1. A2A Server (Terminal 1)
python -c "import asyncio; from common.communication.a2a import A2AServer; asyncio.run(A2AServer().start())"

# 2. Orchestrator (Terminal 2)
python -m ghosts.orchestrator.orchestrator

# 3. Workers (Terminals 3-5)
python -m ghosts.worker.worker

# 4. MCP Server (Terminal 6)
python mcp_server/server.py

# 5. MCP Client (Terminal 7)
python mcp_server/client.py mcp_server/server.py ollama llama3.2
```

## LLM Provider Cheat Sheet

### Free & Local
```bash
# Ollama (Best for: Privacy, offline, experimentation)
ollama pull llama3.2
python mcp_server/client.py mcp_server/server.py ollama llama3.2
```

### Paid API (Good Pricing)
```bash
# OpenRouter (Best for: Variety, some free models)
export OPENAI_API_KEY=your-openrouter-key
python mcp_server/client.py mcp_server/server.py openrouter mixtral-8x7b

# Groq (Best for: Speed - 500+ tokens/sec!)
export GROQ_API_KEY=your-groq-key  
python mcp_server/client.py mcp_server/server.py groq llama-3.1-70b
```

### Premium
```bash
# Claude (Best for: Capability, reasoning)
python mcp_server/client.py mcp_server/server.py anthropic
```

## Worker with Different Providers

```python
# Start worker with specific LLM
from ghosts.worker.worker import WorkerAgent
import asyncio

# Ollama worker (free)
worker = WorkerAgent(llm_provider="ollama")

# Claude worker (powerful)
worker = WorkerAgent(llm_provider="anthropic")

# OpenRouter worker (flexible)
worker = WorkerAgent(llm_provider="openrouter")

await worker.start()
```

## MCP Tools Quick Reference

```python
# LLM Inference
await session.call_tool("llm_inference", {
    "prompt": "Explain quantum computing",
    "max_tokens": 500
})

# Data Analysis
await session.call_tool("analyze_data", {
    "data": '{"sales": [100, 150, 200]}',
    "analysis_type": "statistical"
})

# Task Distribution
await session.call_tool("distribute_task", {
    "task_description": "Analyze sentiment",
    "task_parameters": {"text": "Great product!"}
})
```

## MCP Resources

```python
# List agents
agents = await session.read_resource("ghost://agents/list")

# Get capabilities
caps = await session.read_resource("ghost://agents/capabilities")
```

## Unified LLM Client (Python)

```python
from common.llm.client import LLMFactory

# Claude
client = LLMFactory.create_claude_client()

# Ollama
client = LLMFactory.create_ollama_client("llama3.2")

# OpenRouter
client = LLMFactory.create_openrouter_client(
    api_key="key",
    model="mixtral-8x7b"
)

# Generate
response = await client.generate("Hello, world!")
print(response.content)
```

## Cost Comparison

| Provider | Model | Cost/1M tokens | Speed |
|----------|-------|----------------|-------|
| Ollama | Llama 3.2 | **Free** | Medium |
| OpenRouter | Gemma 2 9B | **Free** | Fast |
| OpenRouter | Mixtral 8x7B | $0.24 | Fast |
| Groq | Llama 3.1 70B | $0.59 | **Ultra Fast** |
| Claude | Sonnet 4.5 | $3.00 | Fast |

## Common Commands

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Check types
mypy .

# Format code
black .

# Lint
ruff check .
```

## File Locations

```
mcp_server/server.py          # MCP server
mcp_server/client.py           # MCP client
common/llm/client.py           # Unified LLM client
ghosts/worker/worker.py        # Worker agent
ghosts/orchestrator/orchestrator.py  # Orchestrator
docs/MCP_GUIDE.md              # Full MCP guide
```

## Environment Variables

```bash
# Claude
ANTHROPIC_API_KEY=sk-ant-api03-...

# OpenRouter
OPENAI_API_KEY=your-openrouter-key
OPENAI_API_BASE=https://openrouter.ai/api/v1

# Ollama (local, no key needed)
# Just: ollama serve

# A2A
A2A_HOST=0.0.0.0
A2A_PORT=8765
```

## Troubleshooting

```bash
# Ollama not working?
ollama serve
curl http://localhost:11434/api/tags

# A2A connection failed?
netstat -an | grep 8765

# Check logs
export LOG_LEVEL=DEBUG
python mcp_server/server.py
```

## Next Steps

1. Try Ollama for free local LLM
2. Mix providers (cheap + powerful workers)
3. Add custom MCP tools
4. Build custom agents
5. Deploy with Docker

## Resources

- **Full Guide**: `docs/MCP_GUIDE.md`
- **Architecture**: `docs/architecture/ARCHITECTURE.md`
- **Development**: `docs/DEVELOPMENT.md`
- **Quick Start**: `docs/QUICKSTART.md`

---

**Pro Tip**: Start with Ollama (free) to experiment, then add Claude for complex tasks!
