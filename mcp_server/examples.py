"""
Example: Integrating MCP with Ghost Swarm Agents.

Demonstrates how to:
1. Create MCP servers with tools
2. Connect agents to MCP servers
3. Use Claude with MCP tools
4. Integrate with A2A framework
"""

import asyncio
from typing import Any
from uuid import uuid4

from common import (
    configure_logging,
    get_logger,
    BaseAgent,
    AgentRole,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from mcp_server.filesystem import FilesystemMCPServer
from mcp_server.agents import AgentManagementMCPServer
from mcp_server.claude_client import ClaudeMCPClient

logger = get_logger(__name__)


class MCPEnabledWorker(BaseAgent):
    """
    Worker agent with MCP capabilities.
    
    Combines:
    - BaseAgent (for A2A communication)
    - ClaudeMCPClient (for Claude + MCP tools)
    """
    
    def __init__(
        self,
        agent_id: str | None = None,
        mcp_servers: list[Any] | None = None,
    ) -> None:
        """
        Initialize MCP-enabled worker.
        
        Args:
            agent_id: Unique agent identifier
            mcp_servers: List of MCP servers to connect to
        """
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.WORKER,
            capabilities=[
                "llm_inference",
                "mcp_tools",
                "file_operations",
                "agent_management",
            ],
        )
        
        # Claude MCP client
        self.claude = ClaudeMCPClient()
        
        # Register MCP servers
        if mcp_servers:
            for server in mcp_servers:
                self.claude.register_mcp_server(server)
        
        logger.info(
            "mcp_enabled_worker_initialized",
            agent_id=self.agent_id,
            mcp_servers=len(mcp_servers) if mcp_servers else 0,
        )
    
    async def initialize(self) -> None:
        """Initialize agent."""
        logger.info("initializing_mcp_worker", agent_id=self.agent_id)
    
    async def cleanup(self) -> None:
        """Clean up agent."""
        logger.info("cleaning_up_mcp_worker", agent_id=self.agent_id)
    
    async def process_task(self, task: TaskRequest) -> TaskResult:
        """
        Process task using Claude with MCP tools.
        
        Args:
            task: Task request
            
        Returns:
            Task result
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(
                "processing_task_with_mcp",
                task_id=str(task.task_id),
                task_type=task.task_type,
            )
            
            # Build task message for Claude
            task_message = f"""Task: {task.description}

Type: {task.task_type}
Parameters: {task.parameters}

Please accomplish this task using the available tools."""
            
            # Process with Claude + MCP tools
            response = await self.claude.chat(
                message=task_message,
                use_tools=True,
            )
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result={
                    "response": response,
                    "tools_used": True,
                },
                execution_time=execution_time,
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "mcp_task_error",
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


async def example_basic_mcp_usage():
    """Example 1: Basic MCP tool usage with Claude."""
    logger.info("=== Example 1: Basic MCP Usage ===")
    
    # Create filesystem MCP server
    fs_server = FilesystemMCPServer(
        allowed_directories=["./data", "./docs"],
        readonly=False,
    )
    await fs_server.setup()
    
    # Create Claude MCP client
    claude = ClaudeMCPClient()
    claude.register_mcp_server(fs_server)
    
    # Use Claude with MCP tools
    response = await claude.chat(
        message="List all markdown files in the docs directory",
        use_tools=True,
    )
    
    logger.info("claude_response", response=response)


async def example_mcp_enabled_agent():
    """Example 2: MCP-enabled Ghost agent."""
    logger.info("=== Example 2: MCP-Enabled Agent ===")
    
    # Create MCP servers
    fs_server = FilesystemMCPServer(
        allowed_directories=["./data"],
        readonly=False,
    )
    await fs_server.setup()
    
    agent_server = AgentManagementMCPServer()
    await agent_server.setup()
    
    # Create MCP-enabled worker
    worker = MCPEnabledWorker(
        mcp_servers=[fs_server, agent_server],
    )
    
    await worker.start()
    
    # Process a task
    task = TaskRequest(
        task_id=uuid4(),
        task_type="file_analysis",
        description="Analyze all Python files in the data directory and summarize their purpose",
        parameters={},
    )
    
    result = await worker.process_task(task)
    
    logger.info(
        "task_completed",
        status=result.status.value,
        execution_time=result.execution_time,
    )
    
    await worker.stop()


async def example_multi_server_workflow():
    """Example 3: Multi-server MCP workflow."""
    logger.info("=== Example 3: Multi-Server Workflow ===")
    
    # Create multiple MCP servers
    fs_server = FilesystemMCPServer(
        allowed_directories=["./"],
        readonly=False,
    )
    await fs_server.setup()
    
    agent_server = AgentManagementMCPServer()
    await agent_server.setup()
    
    # Create agent with multiple servers
    worker = MCPEnabledWorker(
        mcp_servers=[fs_server, agent_server],
    )
    await worker.start()
    
    # Complex task using multiple tools
    task = TaskRequest(
        task_id=uuid4(),
        task_type="complex_analysis",
        description="""
        1. List all Python files in the current directory
        2. Read the content of each file
        3. Identify which files define agents
        4. For each agent file, extract the agent's capabilities
        5. Create a summary report
        """,
        parameters={},
    )
    
    result = await worker.process_task(task)
    
    logger.info("workflow_completed", result=result.result)
    
    await worker.stop()


async def example_streaming_response():
    """Example 4: Streaming responses with MCP."""
    logger.info("=== Example 4: Streaming Response ===")
    
    # Create Claude client
    claude = ClaudeMCPClient()
    
    # Stream response
    logger.info("streaming_started")
    
    async for chunk in claude.stream_chat(
        message="Explain how the Ghost Swarm architecture works in detail",
    ):
        print(chunk, end="", flush=True)
    
    print()  # New line
    logger.info("streaming_completed")


async def main():
    """Run all examples."""
    configure_logging()
    
    logger.info("=== MCP Integration Examples ===")
    
    try:
        # Example 1: Basic usage
        await example_basic_mcp_usage()
        
        print("\n" + "="*50 + "\n")
        
        # Example 2: MCP-enabled agent
        await example_mcp_enabled_agent()
        
        print("\n" + "="*50 + "\n")
        
        # Example 3: Multi-server workflow
        await example_multi_server_workflow()
        
        print("\n" + "="*50 + "\n")
        
        # Example 4: Streaming
        await example_streaming_response()
        
    except Exception as e:
        logger.error("example_error", error=str(e), exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())