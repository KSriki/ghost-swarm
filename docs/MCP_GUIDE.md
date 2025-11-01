# MCP Integration Guide

## Overview

Ghost Swarm integrates the **Model Context Protocol (MCP)** to provide standardized tool access for AI agents using **Claude AI** (not OpenAI), fully integrated with our **Agent2Agent (A2A) framework**.

## What is MCP?

MCP is an open standard that connects AI systems to external tools and data sources:

- **Tools**: Model-controlled actions (file operations, API calls, etc.)
- **Resources**: Application-controlled context (documents, databases, etc.)
- **Prompts**: User-controlled interaction templates

## Ghost Swarm MCP Architecture

```
┌─────────────────┐
│  Ghost Agent    │  ← BaseAgent with A2A
│  (MCP Client)   │
└────────┬────────┘
         │
    ┌────┴─────┐
    │  Claude  │     ← Claude API (not OpenAI)
    │   API    │
    └────┬─────┘
         │
    ┌────┴──────────────┐
    │                   │
┌───▼────────┐    ┌────▼──────┐
│Filesystem  │    │  Agent    │  ← MCP Servers
│MCP Server  │    │MCP Server │
└────────────┘    └───────────┘
```

## Quick Start

### 1. Basic MCP Usage

```python
from mcp_server.filesystem import FilesystemMCPServer
from mcp_server.claude_client import ClaudeMCPClient

# Create MCP server
fs_server = FilesystemMCPServer(allowed_directories=["./data"])
await fs_server.setup()

# Create Claude client with MCP
claude = ClaudeMCPClient()
claude.register_mcp_server(fs_server)

# Use Claude with tools
response = await claude.chat(
    message="List all Python files in the data directory",
    use_tools=True,
)
```

### 2. MCP-Enabled Ghost Agent

```python
from mcp_server.examples import MCPEnabledWorker
from common import TaskRequest
from uuid import uuid4

# Create agent with MCP servers
worker = MCPEnabledWorker(mcp_servers=[fs_server])
await worker.start()

# Process task using MCP tools
task = TaskRequest(
    task_id=uuid4(),
    task_type="analysis",
    description="Analyze all data files",
    parameters={},
)

result = await worker.process_task(task)
```

## Built-in MCP Servers

### Filesystem Server

```python
from mcp_server.filesystem import FilesystemMCPServer

fs_server = FilesystemMCPServer(
    allowed_directories=["./data", "./docs"],
    readonly=False,
)
```

**Tools:**
- `read_file`: Read file contents
- `write_file`: Write to file
- `list_directory`: List contents
- `search_files`: Find files by pattern

### Agent Management Server

```python
from mcp_server.agents import AgentManagementMCPServer

agent_server = AgentManagementMCPServer()
```

**Tools:**
- `list_agents`: List Ghost agents
- `get_agent_info`: Get agent details
- `send_task`: Send task to agent
- `find_best_agent`: Find optimal agent

## Creating Custom MCP Servers

```python
from mcp_server.base import BaseMCPServer

class MyMCPServer(BaseMCPServer):
    def __init__(self):
        super().__init__(name="my-server", version="1.0.0")
    
    async def setup(self):
        # Register tools
        self.register_tool(
            name="my_tool",
            description="My custom tool",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                },
                "required": ["input"],
            },
            handler=self._my_tool_handler,
        )
    
    async def _my_tool_handler(self, input: str) -> dict:
        # Tool logic
        return {"result": f"Processed: {input}"}
```

## Why Claude + MCP (Not OpenAI)

Ghost Swarm uses **Claude with MCP** because:

1. ✅ **Open Standard**: MCP is vendor-neutral
2. ✅ **Claude API**: Superior reasoning with tools
3. ✅ **A2A Integration**: Works with our Agent2Agent framework
4. ✅ **No Vendor Lock-in**: Open source approach
5. ✅ **Better Tool Handling**: More flexible than OpenAI's approach

## Integration with A2A Framework

MCP works seamlessly with A2A:

```python
from common import BaseAgent, AgentRole

class MCPAgent(BaseAgent):
    def __init__(self, mcp_servers):
        super().__init__(role=AgentRole.WORKER)
        
        # Claude with MCP
        self.claude = ClaudeMCPClient()
        for server in mcp_servers:
            self.claude.register_mcp_server(server)
    
    async def process_task(self, task):
        # Claude uses MCP tools
        response = await self.claude.chat(
            message=task.description,
            use_tools=True,
        )
        
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            result={"response": response},
        )
```

## Examples

Run examples:

```bash
# All MCP examples
python -m mcp_server.examples

# Specific servers
python -m mcp_server.filesystem
python -m mcp_server.agents
```

## Best Practices

### Security

```python
# Restrict filesystem access
fs_server = FilesystemMCPServer(
    allowed_directories=["./safe/path"],
    readonly=True,
)
```

### Error Handling

```python
try:
    response = await claude.chat(message, use_tools=True)
except Exception as e:
    logger.error("mcp_error", error=str(e))
```

### Performance

```python
# Use async for parallel processing
results = await asyncio.gather(*[
    claude.chat(msg, use_tools=True)
    for msg in messages
])
```

## Resources

- **MCP Spec**: https://spec.modelcontextprotocol.io/
- **Claude API**: https://docs.anthropic.com/
- **Code**: `mcp_server/` directory
- **Examples**: `mcp_server/examples.py`

## Next Steps

1. Try the examples: `python -m mcp_server.examples`
2. Create custom MCP servers for your needs
3. Integrate with existing Ghost agents
4. Combine multiple MCP servers
5. Deploy with proper security

MCP makes Ghost Swarm agents incredibly powerful while staying fully open source! 🚀