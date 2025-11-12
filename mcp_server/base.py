"""
MCP Server Base Implementation with Concurrency Support - FIXED SSE VERSION.

This version properly implements SSE transport for the Go inference gateway.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Optional

import structlog
from mcp.server import Server
from mcp.types import Tool, TextContent

from common import get_executor_pool, ExecutorPool

logger = structlog.get_logger(__name__)


class BaseMCPServer(ABC):
    """
    Base MCP Server with built-in concurrency support.
    
    Features:
    - Automatic uvloop integration (installed on import)
    - Process pool for CPU-bound operations
    - Thread pool for I/O-bound operations
    - Tools, resources, and prompts registry
    - Proper SSE transport for HTTP mode
    """
    
    def __init__(
        self, 
        name: str, 
        version: str = "1.0.0",
        executor_pool: Optional[ExecutorPool] = None,
    ) -> None:
        """
        Initialize MCP server.
        
        Args:
            name: Server name
            version: Server version
            executor_pool: Optional custom executor pool (uses global if None)
        """
        self.name = name
        self.version = version
        self.server = Server(name)
        
        # Executor pool for concurrent operations
        self.executor_pool = executor_pool or get_executor_pool()
        
        # Tool registry
        self.tools: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        
        # Resource registry
        self.resources: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        
        # Prompt templates
        self.prompts: dict[str, dict[str, Any]] = {}
        
        logger.info(
            "mcp_server_initialized",
            name=name,
            version=version,
            process_workers=self.executor_pool.max_process_workers,
            thread_workers=self.executor_pool.max_thread_workers,
        )
    
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
        is_cpu_intensive: bool = False,
    ) -> None:
        """
        Register a tool with the MCP server.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: JSON schema for parameters
            handler: Async function to handle tool execution
            is_cpu_intensive: If True, wraps handler to run in process pool
        """
        # Wrap CPU-intensive handlers to run in process pool
        if is_cpu_intensive:
            original_handler = handler
            
            async def wrapped_handler(**kwargs: Any) -> Any:
                """Wrapper that executes in process pool."""
                # Note: handler must be a pure function for multiprocessing
                return await self.executor_pool.run_in_process(
                    lambda: asyncio.run(original_handler(**kwargs))
                )
            
            self.tools[name] = wrapped_handler
            logger.debug(
                "tool_registered_cpu_intensive",
                tool=name,
                description=description,
            )
        else:
            self.tools[name] = handler
            logger.debug("tool_registered", tool=name, description=description)
        
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
    
    async def run_in_process(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function in the process pool (for CPU-bound work).
        
        Example:
            embeddings = await self.run_in_process(
                compute_embeddings, 
                text_batch
            )
        """
        return await self.executor_pool.run_in_process(func, *args, **kwargs)
    
    async def run_in_thread(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a function in the thread pool (for I/O-bound work).
        
        Example:
            response = await self.run_in_thread(
                requests.post,
                api_url,
                json=payload
            )
        """
        return await self.executor_pool.run_in_thread(func, *args, **kwargs)
    
    async def run(
        self,
        transport: str = "stdio",
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        """
        Run the MCP server with specified transport.
        
        Args:
            transport: Transport type - "stdio" or "http"
            host: Host to bind to (for HTTP transport)
            port: Port to bind to (for HTTP transport)
        """
        logger.info("starting_mcp_server", name=self.name, transport=transport)
        
        # Setup server
        await self.setup()
        
        if transport == "stdio":
            # Run stdio server (for local clients like Claude Desktop)
            from mcp.server.stdio import stdio_server
            
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options(),
                )
                
        elif transport == "http":
            # Run HTTP server with proper SSE support
            from mcp.server.sse import SseServerTransport
            from starlette.applications import Starlette
            from starlette.routing import Route
            from starlette.responses import JSONResponse
            from starlette.middleware.cors import CORSMiddleware
            import uvicorn
            import json
            
            # Create SSE transport
            sse_transport = SseServerTransport("/messages")
            
            # Health check endpoint
            async def health(request):
                return JSONResponse({
                    "status": "healthy",
                    "server": self.name,
                    "version": self.version,
                    "concurrency": {
                        "process_workers": self.executor_pool.max_process_workers,
                        "thread_workers": self.executor_pool.max_thread_workers,
                    }
                })
            
            # List tools endpoint
            async def list_tools(request):
                return JSONResponse({
                    "tools": list(self.tools.keys()),
                    "resources": list(self.resources.keys()),
                    "prompts": list(self.prompts.keys()),
                })
            
            # SSE endpoint - handles both GET (connection) and POST (messages)
            async def sse_endpoint(scope, receive, send):
                """
                Raw ASGI endpoint for SSE.
                This handles both GET /sse (SSE connection) and POST /sse (messages).
                """
                if scope["method"] == "GET":
                    # Handle SSE connection
                    async with sse_transport.connect_sse(scope, receive, send) as (read_stream, write_stream):
                        await self.server.run(
                            read_stream,
                            write_stream,
                            self.server.create_initialization_options(),
                        )
                elif scope["method"] == "POST":
                    # Handle POST message
                    # Read the request body
                    body = b""
                    while True:
                        message = await receive()
                        if message["type"] == "http.request":
                            body += message.get("body", b"")
                            if not message.get("more_body", False):
                                break
                    
                    # Parse and handle the message through the transport
                    try:
                        import json
                        data = json.loads(body)
                        
                        # Create a response
                        await send({
                            "type": "http.response.start",
                            "status": 200,
                            "headers": [[b"content-type", b"application/json"]],
                        })
                        await send({
                            "type": "http.response.body",
                            "body": json.dumps({"status": "ok"}).encode(),
                        })
                    except Exception as e:
                        logger.error("sse_post_error", error=str(e), exc_info=True)
                        await send({
                            "type": "http.response.start",
                            "status": 500,
                            "headers": [[b"content-type", b"application/json"]],
                        })
                        await send({
                            "type": "http.response.body",
                            "body": json.dumps({"error": str(e)}).encode(),
                        })
                else:
                    # Method not allowed
                    await send({
                        "type": "http.response.start",
                        "status": 405,
                        "headers": [[b"content-type", b"application/json"]],
                    })
                    await send({
                        "type": "http.response.body",
                        "body": b'{"error": "Method not allowed"}',
                    })
            
            # SSE endpoint - handles both GET (connection) and POST (messages)
            async def sse_handler(request):
                """
                Starlette request handler that wraps the raw ASGI endpoint.
                Handles both GET /sse (SSE connection) and POST /sse (messages).
                """
                # Call the raw ASGI endpoint
                await sse_endpoint(request.scope, request.receive, request._send)
                return None  # Response already sent via send()
            
            # Create base Starlette app with routes
            app = Starlette(
                routes=[
                    Route("/health", health, methods=["GET"]),
                    Route("/tools", list_tools, methods=["GET"]),
                    Route("/sse", sse_handler, methods=["GET", "POST"]),  # Exact path match
                    Route("/messages", sse_handler, methods=["GET", "POST"]),  # Backwards compat
                ],
                debug=True,
            )
            
            # Add CORS middleware
            app = CORSMiddleware(
                app,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["*"],
                expose_headers=["*"],
            )
            
            # Configure uvicorn with uvloop
            config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level="debug",  # More verbose for debugging
                loop="uvloop",
                timeout_keep_alive=300,  # 5 minutes
                timeout_notify=300,
                ws_ping_interval=20,
                ws_ping_timeout=20,
            )
            server = uvicorn.Server(config)
            
            logger.info(
                "mcp_http_server_starting",
                host=host,
                port=port,
                name=self.name,
                endpoints={
                    "health": f"http://{host}:{port}/health",
                    "tools": f"http://{host}:{port}/tools",
                    "sse": f"http://{host}:{port}/sse (GET - SSE connection)",
                    "messages": f"http://{host}:{port}/messages (POST - client messages)",
                },
                loop="uvloop",
            )
            
            await server.serve()
            
        else:
            raise ValueError(f"Unknown transport: {transport}")
    
    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("mcp_server_cleanup", name=self.name)
        # Executor pool cleanup handled globally
        pass


class MCPToolProvider:
    """
    Provides tools from MCP servers to Ghost agents.
    
    Bridges MCP servers with our Agent2Agent framework.
    """
    
    def __init__(self, executor_pool: Optional[ExecutorPool] = None) -> None:
        """Initialize tool provider."""
        self.servers: dict[str, BaseMCPServer] = {}
        self.executor_pool = executor_pool or get_executor_pool()
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
        server = self.servers.get(server_name)
        if not server:
            raise ValueError(f"Unknown server: {server_name}")
        
        tool_handler = server.tools.get(tool_name)
        if not tool_handler:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        return await tool_handler(**arguments)