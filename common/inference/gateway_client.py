"""
AgentGateway Client for Ghost Swarm Agents.

Provides a lightweight client for agents to communicate with the centralized
AgentGateway for LLM inference, MCP tool calls, and A2A mesh networking.

Architecture:
    Agent → AgentGateway → {SLM, LLM, MCP Servers}
"""

import os
from typing import Any, Optional
import httpx
import structlog

logger = structlog.get_logger(__name__)


class AgentGatewayClient:
    """
    Client for communicating with AgentGateway.
    
    Handles:
    - LLM inference routing (SLM vs Claude)
    - MCP tool discovery and execution
    - Agent mesh communication
    """
    
    def __init__(
        self,
        gateway_url: str | None = None,
        agent_id: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        """
        Initialize AgentGateway client.
        
        Args:
            gateway_url: URL of the AgentGateway (default: http://agentgateway:3000)
            agent_id: ID of the calling agent (for tracing)
            timeout: Request timeout in seconds
        """
        self.gateway_url = gateway_url or os.getenv(
            "AGENTGATEWAY_URL",
            "http://agentgateway:3000"
        )
        self.agent_id = agent_id or "unknown"
        self.timeout = timeout
        
        # HTTP client for gateway communication
        self.http_client = httpx.AsyncClient(
            base_url=self.gateway_url,
            timeout=timeout,
            headers={
                "User-Agent": f"ghost-swarm-agent/{self.agent_id}",
            },
        )
        
        logger.info(
            "gateway_client_initialized",
            agent_id=self.agent_id,
            gateway_url=self.gateway_url,
        )
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.http_client.aclose()
    
    async def infer(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        complexity: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform LLM inference via AgentGateway.
        
        The gateway will route to SLM or LLM based on complexity.
        
        Args:
            messages: Conversation messages
            system: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            complexity: Task complexity hint ("simple", "moderate", "complex")
            **kwargs: Additional model parameters
            
        Returns:
            Dict with:
                - content: Generated text
                - model: Model used
                - provider: Provider used (anthropic, llama_cpp, etc.)
                - usage: Token usage stats
                - cost: Estimated cost
        """
        try:
            # Prepare request payload
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            }
            
            if system:
                payload["system"] = system
            
            # Add complexity hint for gateway routing
            if complexity:
                payload["metadata"] = {"complexity": complexity}
            
            # Send to gateway LLM endpoint
            response = await self.http_client.post(
                "/v1/chat/completions",
                json=payload,
                headers={
                    "X-Agent-ID": self.agent_id,
                    "X-Complexity": complexity or "unknown",
                },
            )
            response.raise_for_status()
            
            result = response.json()
            
            logger.info(
                "gateway_inference_complete",
                agent_id=self.agent_id,
                complexity=complexity,
                model=result.get("model"),
                provider=result.get("provider"),
            )
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(
                "gateway_inference_error",
                agent_id=self.agent_id,
                status_code=e.response.status_code,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "gateway_client_error",
                agent_id=self.agent_id,
                error=str(e),
            )
            raise
    
    async def list_mcp_tools(self) -> list[dict[str, Any]]:
        """
        List available MCP tools via gateway.
        
        Returns:
            List of available tools
        """
        try:
            response = await self.http_client.get("/mcp/tools")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(
                "list_mcp_tools_error",
                agent_id=self.agent_id,
                error=str(e),
            )
            raise
    
    async def call_mcp_tool(
        self,
        tool: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute an MCP tool via gateway.
        
        Args:
            tool: Tool name (e.g., "filesystem.read_file")
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            response = await self.http_client.post(
                "/mcp/execute",
                json={
                    "tool": tool,
                    "arguments": arguments,
                },
                headers={"X-Agent-ID": self.agent_id},
            )
            response.raise_for_status()
            
            result = response.json()
            
            logger.info(
                "mcp_tool_executed",
                agent_id=self.agent_id,
                tool=tool,
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "mcp_tool_error",
                agent_id=self.agent_id,
                tool=tool,
                error=str(e),
            )
            raise
    
    async def health_check(self) -> bool:
        """
        Check if gateway is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = await self.http_client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False