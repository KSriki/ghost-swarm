"""
Inference Gateway Client for Ghost Swarm.

Provides a unified interface to the Inference Gateway for routing requests
to appropriate models (SLM/LLM) based on TaskMaster classification.

Features:
- OpenAI-compatible API client
- Automatic model routing based on task complexity
- Fallback handling
- Performance tracking
- Non-blocking async operations
"""

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, AsyncIterator

import httpx
import structlog

from common.communication.concurrency import run_io_bound

logger = structlog.get_logger(__name__)


class ModelTier(str, Enum):
    """Model tiers for routing."""
    
    SLM = "slm"  # Small Language Model (fast, cheap)
    LLM = "llm"  # Large Language Model (powerful, expensive)


@dataclass
class InferenceGatewayResponse:
    """Response from Inference Gateway."""
    
    content: str
    model: str
    model_tier: ModelTier
    tokens_used: int
    latency_ms: float
    cost_usd: float
    metadata: dict[str, Any]


class InferenceGatewayClient:
    """
    Client for Inference Gateway with intelligent model routing.
    
    Routes requests to SLM or LLM based on TaskMaster complexity classification.
    Provides OpenAI-compatible API interface for seamless integration.
    
    Example:
        client = InferenceGatewayClient()
        
        # Automatic routing based on complexity
        response = await client.chat(
            messages=[{"role": "user", "content": "Parse this JSON: {...}"}],
            complexity="simple",  # Routes to SLM
        )
        
        # Force specific model
        response = await client.chat(
            messages=[{"role": "user", "content": "Strategic analysis..."}],
            model="claude-sonnet-4-5-20250929",  # Force LLM
        )
    """
    
    def __init__(
        self,
        gateway_url: Optional[str] = None,
        slm_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize Inference Gateway client.
        
        Args:
            gateway_url: Gateway endpoint (defaults from env)
            slm_model: SLM model name (defaults from env)
            llm_model: LLM model name (defaults from env)
            timeout: Request timeout in seconds
        """
        self.gateway_url = gateway_url or os.getenv("AGENTGATEWAY_URL")
        
        # Model configuration
        self.slm_model = slm_model or os.getenv("SLM_MODEL", "phi-3-mini")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "claude-sonnet-4-5-20250929")
        
        # HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "slm_requests": 0,
            "llm_requests": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "total_latency_ms": 0.0,
            "errors": 0,
        }
        
        logger.info(
            "agentgateway_client_initialized",
            gateway_url=self.gateway_url,
            slm_model=self.slm_model,
            llm_model=self.llm_model,
        )
    
    def _select_model(
        self,
        complexity: Optional[str] = None,
        model: Optional[str] = None,
    ) -> tuple[str, ModelTier]:
        """
        Select model based on complexity or explicit model name.
        
        Args:
            complexity: Task complexity from TaskMaster ("simple", "moderate", "complex")
            model: Explicit model name (overrides complexity)
            
        Returns:
            Tuple of (model_name, model_tier)
        """
        # Explicit model override
        if model:
            # Determine tier based on model name
            if "claude" in model.lower() or "gpt-4" in model.lower():
                return model, ModelTier.LLM
            else:
                return model, ModelTier.SLM
        
        # Route based on complexity
        if complexity in ["simple", "moderate"]:
            return self.slm_model, ModelTier.SLM
        else:
            # Default to LLM for complex or unclassified tasks
            return self.llm_model, ModelTier.LLM
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        complexity: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> InferenceGatewayResponse:
        """
        Send chat completion request to Inference Gateway.
        
        Args:
            messages: List of message dicts with "role" and "content"
            complexity: Task complexity from TaskMaster ("simple", "moderate", "complex")
            model: Optional explicit model name (overrides complexity routing)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response (not yet implemented)
            **kwargs: Additional parameters for the model
            
        Returns:
            InferenceGatewayResponse with content and metadata
        """
        start_time = time.time()
        
        # Select model based on complexity or explicit choice
        selected_model, model_tier = self._select_model(complexity, model)
        
        # Update stats
        self.stats["total_requests"] += 1
        if model_tier == ModelTier.SLM:
            self.stats["slm_requests"] += 1
        else:
            self.stats["llm_requests"] += 1
        
        logger.debug(
            "inference_request",
            model=selected_model,
            tier=model_tier.value,
            complexity=complexity,
            messages_count=len(messages),
        )
        
        # Prepare request payload (OpenAI-compatible format)
        payload = {
            "model": selected_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        
        try:
            # Make request to gateway
            response = await self.client.post(
                self.gateway_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract response data (OpenAI format)
            choice = data["choices"][0]
            content = choice["message"]["content"]
            
            # Extract usage statistics
            usage = data.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            
            # Calculate cost (approximate)
            cost = self._calculate_cost(total_tokens, model_tier)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Update stats
            self.stats["total_tokens"] += total_tokens
            self.stats["total_cost_usd"] += cost
            self.stats["total_latency_ms"] += latency_ms
            
            logger.info(
                "inference_completed",
                model=selected_model,
                tier=model_tier.value,
                tokens=total_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
            )
            
            return InferenceGatewayResponse(
                content=content,
                model=selected_model,
                model_tier=model_tier,
                tokens_used=total_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                metadata={
                    "complexity": complexity,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": usage,
                },
            )
            
        except httpx.HTTPStatusError as e:
            self.stats["errors"] += 1
            logger.error(
                "agentgateway_http_error",
                status_code=e.response.status_code,
                error=str(e),
                model=selected_model,
            )
            raise
        
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(
                "agentgateway_error",
                error=str(e),
                model=selected_model,
                exc_info=True,
            )
            raise
    
    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        complexity: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream chat completion response from Inference Gateway.
        
        Args:
            messages: List of message dicts
            complexity: Task complexity
            model: Optional explicit model
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Yields:
            Response chunks as they arrive
        """
        selected_model, model_tier = self._select_model(complexity, model)
        
        payload = {
            "model": selected_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }
        
        try:
            async with self.client.stream(
                "POST",
                self.gateway_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                
                async for chunk in response.aiter_lines():
                    if chunk.startswith("data: "):
                        data = chunk[6:]  # Remove "data: " prefix
                        
                        if data == "[DONE]":
                            break
                        
                        try:
                            import json
                            chunk_data = json.loads(data)
                            delta = chunk_data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            
                            if content:
                                yield content
                        
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning("stream_parse_error", error=str(e))
                            continue
        
        except Exception as e:
            logger.error(
                "stream_error",
                error=str(e),
                model=selected_model,
                exc_info=True,
            )
            raise
    
    def _calculate_cost(self, tokens: int, model_tier: ModelTier) -> float:
        """
        Calculate approximate cost based on tokens and model tier.
        
        Args:
            tokens: Total tokens used
            model_tier: Model tier (SLM or LLM)
            
        Returns:
            Estimated cost in USD
        """
        if model_tier == ModelTier.SLM:
            # SLM is typically free (self-hosted) or very cheap
            return 0.0
        else:
            # LLM approximate cost (Claude Sonnet 4.5)
            # $3/M input, $15/M output - average ~$9/M tokens
            return (tokens / 1_000_000) * 9.0
    
    async def health_check(self) -> dict[str, Any]:
        """
        Check Inference Gateway health.
        
        Returns:
            Health status dict
        """
        try:
            # Most gateways expose a health endpoint
            health_url = self.gateway_url.replace("/v1/chat/completions", "/health")
            
            response = await self.client.get(health_url)
            response.raise_for_status()
            
            return {
                "status": "healthy",
                "gateway_url": self.gateway_url,
                "response": response.json(),
            }
        
        except Exception as e:
            logger.warning("health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "gateway_url": self.gateway_url,
                "error": str(e),
            }
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get client statistics.
        
        Returns:
            Statistics dict with usage metrics
        """
        total_requests = self.stats["total_requests"]
        
        avg_latency = (
            self.stats["total_latency_ms"] / total_requests
            if total_requests > 0
            else 0.0
        )
        
        slm_percentage = (
            self.stats["slm_requests"] / total_requests * 100
            if total_requests > 0
            else 0.0
        )
        
        return {
            **self.stats,
            "avg_latency_ms": avg_latency,
            "slm_percentage": slm_percentage,
            "llm_percentage": 100 - slm_percentage,
            "avg_cost_per_request": (
                self.stats["total_cost_usd"] / total_requests
                if total_requests > 0
                else 0.0
            ),
        }
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.stats = {
            "total_requests": 0,
            "slm_requests": 0,
            "llm_requests": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "total_latency_ms": 0.0,
            "errors": 0,
        }
        logger.info("stats_reset")
    
    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        await self.client.aclose()
        logger.info("inference_gateway_client_closed")
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()


# Convenience functions for backwards compatibility with existing code

async def infer_with_gateway(
    prompt: str,
    system: Optional[str] = None,
    complexity: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    gateway_client: Optional[InferenceGatewayClient] = None,
) -> InferenceGatewayResponse:
    """
    Convenience function for simple inference with gateway.
    
    Args:
        prompt: User prompt
        system: System prompt
        complexity: Task complexity from TaskMaster
        model: Optional explicit model
        max_tokens: Maximum tokens
        temperature: Sampling temperature
        gateway_client: Optional client instance (creates new if None)
        
    Returns:
        InferenceGatewayResponse
    """
    # Build messages
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    # Use provided client or create new one
    if gateway_client:
        return await gateway_client.chat(
            messages=messages,
            complexity=complexity,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        async with InferenceGatewayClient() as client:
            return await client.chat(
                messages=messages,
                complexity=complexity,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )