"""
MCP Server Base Implementation for Ghost Swarm.

Provides a foundation for creating MCP servers that expose tools, resources,
and prompts to AI agents using Claude, integrated with our A2A framework.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine

import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

from common import get_logger

logger = get_logger(__name__)


class BaseMCPServer(ABC):
    """
    Base MCP Server for Ghost Swarm.
    
    Provides standardized interface for:
    - Tools: Model-controlled actions
    - Resources: Application-controlled context  
    - Prompts: User-controlled interactions
    """
    
    def __init__(self, name: str, version: str = "1.0.0") -> None:
        """
        Initialize MCP server.
        
        Args:
            name: Server name
            version: Server version
        """
        self.name = name
        self.version = version
        self.server = Server(name)
        
        # Tool registry
        self.tools: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        
        # Resource registry
        self.resources: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        
        # Prompt templates
        self.prompts: dict[str, dict[str, Any]] = {}
        
        logger.info("mcp_server_initialized", name=name, version=version)
    
    @abstractmethod
    async def setup(self) -> None:
        """Setup server - register tools, resources, and prompts."""
        pass
    
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """
        Register a tool with the MCP server.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: JSON schema for parameters
            handler: Async function to handle tool execution
        """
        self.tools[name] = handler
        
        tool = Tool(
            name=name,
            description=description,
            inputSchema=parameters,
        )
        
        # Register with MCP server
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool call from MCP client."""
            if name not in self.tools:
                raise ValueError(f"Unknown tool: {name}")
            
            handler = self.tools[name]
            result = await handler(**arguments)
            
            return [
                TextContent(
                    type="text",
                    text=str(result),
                )
            ]
        
        logger.info("tool_registered", tool_name=name)
    
    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str,
        handler: Callable[..., Coroutine[Any, Any, str]],
    ) -> None:
        """
        Register a resource with the MCP server.
        
        Args:
            uri: Resource URI
            name: Resource name
            description: Resource description
            mime_type: MIME type of resource
            handler: Async function to retrieve resource content
        """
        self.resources[uri] = handler
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Handle resource read from MCP client."""
            if uri not in self.resources:
                raise ValueError(f"Unknown resource: {uri}")
            
            handler = self.resources[uri]
            content = await handler()
            return content
        
        logger.info("resource_registered", uri=uri, name=name)
    
    def register_prompt(
        self,
        name: str,
        description: str,
        arguments: list[dict[str, Any]],
        template: str,
    ) -> None:
        """
        Register a prompt template with the MCP server.
        
        Args:
            name: Prompt name
            description: Prompt description
            arguments: List of argument definitions
            template: Prompt template string
        """
        self.prompts[name] = {
            "description": description,
            "arguments": arguments,
            "template": template,
        }
        
        @self.server.get_prompt()
        async def handle_get_prompt(
            name: str,
            arguments: dict[str, Any] | None = None,
        ) -> str:
            """Handle prompt get from MCP client."""
            if name not in self.prompts:
                raise ValueError(f"Unknown prompt: {name}")
            
            prompt_data = self.prompts[name]
            template = prompt_data["template"]
            
            # Simple template substitution
            if arguments:
                for key, value in arguments.items():
                    template = template.replace(f"{{{key}}}", str(value))
            
            return template
        
        logger.info("prompt_registered", prompt_name=name)
    
    async def run(self) -> None:
        """Run the MCP server."""
        logger.info("starting_mcp_server", name=self.name)
        
        # Setup server
        await self.setup()
        
        # Run stdio server
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


class MCPToolProvider:
    """
    Provides tools from MCP servers to Ghost agents.
    
    Bridges MCP servers with our Agent2Agent framework.
    """
    
    def __init__(self) -> None:
        """Initialize tool provider."""
        self.servers: dict[str, BaseMCPServer] = {}
        logger.info("mcp_tool_provider_initialized")
    
    def register_server(self, server: BaseMCPServer) -> None:
        """
        Register an MCP server.
        
        Args:
            server: MCP server instance
        """
        self.servers[server.name] = server
        logger.info("mcp_server_registered", server_name=server.name)
    
    async def list_tools(self) -> list[dict[str, Any]]:
        """
        List all available tools from registered servers.
        
        Returns:
            List of tool specifications
        """
        tools = []
        
        for server in self.servers.values():
            for tool_name in server.tools.keys():
                tools.append({
                    "name": tool_name,
                    "server": server.name,
                })
        
        return tools
    
    async def execute_tool(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Execute a tool from a registered server.
        
        Args:
            tool_name: Name of tool to execute
            server_name: Name of server providing the tool
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if server_name not in self.servers:
            raise ValueError(f"Unknown server: {server_name}")
        
        server = self.servers[server_name]
        
        if tool_name not in server.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        handler = server.tools[tool_name]
        result = await handler(**arguments)
        
        logger.info(
            "tool_executed",
            tool_name=tool_name,
            server_name=server_name,
        )
        
        return result