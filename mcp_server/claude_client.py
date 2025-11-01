"""
Claude-powered MCP Client for Ghost Swarm Agents.

Enables agents to use Claude AI with MCP tools integrated via our A2A framework.
Uses Claude API instead of OpenAI.
"""

import asyncio
import json
from typing import Any, AsyncIterator

import structlog
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message as AnthropicMessage, TextBlock, ToolUseBlock

from common import get_logger, get_settings
from mcp_server.base import MCPToolProvider

logger = get_logger(__name__)


class ClaudeMCPClient:
    """
    Claude-powered MCP client.
    
    Integrates:
    - Claude API for LLM inference
    - MCP tools for extended capabilities
    - A2A framework for agent communication
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> None:
        """
        Initialize Claude MCP client.
        
        Args:
            model: Claude model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        settings = get_settings()
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize Claude clients
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.async_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        
        # Tool provider for MCP tools
        self.tool_provider = MCPToolProvider()
        
        # Conversation history
        self.messages: list[dict[str, Any]] = []
        
        logger.info(
            "claude_mcp_client_initialized",
            model=model,
            max_tokens=max_tokens,
        )
    
    def register_mcp_server(self, server: Any) -> None:
        """
        Register an MCP server with the client.
        
        Args:
            server: MCP server instance
        """
        self.tool_provider.register_server(server)
        logger.info("mcp_server_registered", server_name=server.name)
    
    def _convert_tools_to_claude_format(self) -> list[dict[str, Any]]:
        """Convert MCP tools to Claude's tool format."""
        claude_tools = []
        
        for server in self.tool_provider.servers.values():
            for tool_name, handler in server.tools.items():
                # Get tool schema from server
                # For now, use a basic format
                claude_tools.append({
                    "name": f"{server.name}_{tool_name}",
                    "description": f"Tool from {server.name} server: {tool_name}",
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                })
        
        return claude_tools
    
    async def chat(
        self,
        message: str,
        system_prompt: str | None = None,
        use_tools: bool = True,
    ) -> str:
        """
        Send a message to Claude and get a response.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            use_tools: Whether to enable tool use
            
        Returns:
            Claude's response
        """
        # Add user message to history
        self.messages.append({
            "role": "user",
            "content": message,
        })
        
        # Prepare tools
        tools = self._convert_tools_to_claude_format() if use_tools else None
        
        # Call Claude
        try:
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt or "You are a helpful AI assistant.",
                messages=self.messages,
                tools=tools if tools else [],
            )
            
            # Process response
            assistant_message = await self._process_response(response)
            
            # Add to history
            self.messages.append({
                "role": "assistant",
                "content": assistant_message,
            })
            
            return assistant_message
            
        except Exception as e:
            logger.error("claude_chat_error", error=str(e), exc_info=True)
            raise
    
    async def _process_response(self, response: AnthropicMessage) -> str:
        """
        Process Claude's response, handling tool calls if needed.
        
        Args:
            response: Claude's response message
            
        Returns:
            Final response text
        """
        response_text = ""
        tool_calls_made = False
        
        for block in response.content:
            if isinstance(block, TextBlock):
                response_text += block.text
            
            elif isinstance(block, ToolUseBlock):
                tool_calls_made = True
                
                # Execute tool
                tool_result = await self._execute_tool_call(block)
                
                # Add tool result to conversation
                response_text += f"\n[Tool: {block.name}]\n{tool_result}\n"
        
        # If tools were called, make another call to Claude with results
        if tool_calls_made and response.stop_reason == "tool_use":
            # Continue conversation with tool results
            followup = await self.async_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=self.messages + [
                    {
                        "role": "assistant",
                        "content": response_text,
                    }
                ],
            )
            
            # Get final response
            for block in followup.content:
                if isinstance(block, TextBlock):
                    response_text += block.text
        
        return response_text
    
    async def _execute_tool_call(self, tool_block: ToolUseBlock) -> str:
        """
        Execute a tool call from Claude.
        
        Args:
            tool_block: Tool use block from Claude
            
        Returns:
            Tool execution result
        """
        tool_name = tool_block.name
        arguments = tool_block.input
        
        try:
            # Parse server name and tool name
            if "_" in tool_name:
                server_name, tool_name = tool_name.split("_", 1)
            else:
                server_name = list(self.tool_provider.servers.keys())[0]
            
            # Execute tool
            result = await self.tool_provider.execute_tool(
                tool_name=tool_name,
                server_name=server_name,
                arguments=arguments,
            )
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(
                "tool_execution_error",
                tool_name=tool_name,
                error=str(e),
                exc_info=True,
            )
            return f"Error executing tool: {str(e)}"
    
    async def stream_chat(
        self,
        message: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a chat response from Claude.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            
        Yields:
            Response chunks
        """
        # Add user message to history
        self.messages.append({
            "role": "user",
            "content": message,
        })
        
        try:
            # Stream response
            async with self.async_client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt or "You are a helpful AI assistant.",
                messages=self.messages,
            ) as stream:
                full_response = ""
                
                async for text in stream.text_stream:
                    full_response += text
                    yield text
                
                # Add to history
                self.messages.append({
                    "role": "assistant",
                    "content": full_response,
                })
                
        except Exception as e:
            logger.error("claude_stream_error", error=str(e), exc_info=True)
            raise
    
    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self.messages.clear()
        logger.info("conversation_reset")
    
    def get_conversation_history(self) -> list[dict[str, Any]]:
        """Get conversation history."""
        return self.messages.copy()


class MCPAgent:
    """
    Ghost agent with MCP and Claude integration.
    
    Combines:
    - Claude AI for reasoning
    - MCP tools for actions
    - A2A for agent communication
    """
    
    def __init__(
        self,
        agent_id: str,
        system_prompt: str | None = None,
    ) -> None:
        """
        Initialize MCP-enabled agent.
        
        Args:
            agent_id: Unique agent identifier
            system_prompt: System prompt for Claude
        """
        self.agent_id = agent_id
        self.system_prompt = system_prompt or (
            "You are an AI agent in the Ghost Swarm system. "
            "You have access to various tools via MCP servers. "
            "Use these tools to help accomplish tasks effectively."
        )
        
        # Claude MCP client
        self.claude = ClaudeMCPClient()
        
        logger.info("mcp_agent_initialized", agent_id=agent_id)
    
    def add_mcp_server(self, server: Any) -> None:
        """
        Add an MCP server to this agent.
        
        Args:
            server: MCP server instance
        """
        self.claude.register_mcp_server(server)
    
    async def process_task(self, task_description: str) -> str:
        """
        Process a task using Claude and available MCP tools.
        
        Args:
            task_description: Description of task to accomplish
            
        Returns:
            Task result
        """
        logger.info("processing_task", agent_id=self.agent_id)
        
        response = await self.claude.chat(
            message=task_description,
            system_prompt=self.system_prompt,
            use_tools=True,
        )
        
        return response
    
    async def chat(self, message: str) -> str:
        """
        Chat with the agent.
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        return await self.claude.chat(
            message=message,
            system_prompt=self.system_prompt,
        )


async def main() -> None:
    """Example usage of Claude MCP client."""
    from common import configure_logging
    from mcp_server.filesystem import FilesystemMCPServer
    
    configure_logging()
    
    # Create MCP agent
    agent = MCPAgent(
        agent_id="demo-agent",
        system_prompt="You are a helpful file management assistant.",
    )
    
    # Add filesystem MCP server
    fs_server = FilesystemMCPServer(
        allowed_directories=["./data"],
        readonly=False,
    )
    await fs_server.setup()
    agent.add_mcp_server(fs_server)
    
    # Process a task
    result = await agent.process_task(
        "List all Python files in the data directory"
    )
    
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())