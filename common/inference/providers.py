"""
Inference Providers for SLM and LLM.

Supports multiple backends:
- SLM: llama.cpp (local), Ollama (local)
- LLM: Anthropic Claude (cloud)
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class InferenceProvider(str, Enum):
    """Types of inference providers."""
    
    SLM_LLAMA_CPP = "slm_llama_cpp"
    SLM_OLLAMA = "slm_ollama"
    LLM_ANTHROPIC = "llm_anthropic"
    LLM_OPENAI = "llm_openai"


@dataclass
class InferenceResponse:
    """Response from an inference provider."""
    
    content: str
    provider: InferenceProvider
    model: str
    tokens_used: int
    latency_ms: float
    cost: float = 0.0
    metadata: dict[str, Any] | None = None


class BaseInferenceProvider(ABC):
    """Base class for all inference providers."""
    
    def __init__(self, provider_type: InferenceProvider):
        self.provider_type = provider_type
        self.total_calls = 0
        self.total_tokens = 0
        self.total_latency_ms = 0.0
        self.total_cost = 0.0
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> InferenceResponse:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if provider is available."""
        pass
    
    def get_stats(self) -> dict[str, Any]:
        """Get provider statistics."""
        avg_latency = self.total_latency_ms / self.total_calls if self.total_calls > 0 else 0
        
        return {
            "provider": self.provider_type.value,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": avg_latency,
            "total_cost": self.total_cost,
        }


class SLMProvider(BaseInferenceProvider):
    """Base class for SLM providers (local inference)."""
    
    def __init__(self, provider_type: InferenceProvider, model_name: str):
        super().__init__(provider_type)
        self.model_name = model_name


class LLMProvider(BaseInferenceProvider):
    """Base class for LLM providers (cloud inference)."""
    
    def __init__(self, provider_type: InferenceProvider, api_key: str):
        super().__init__(provider_type)
        self.api_key = api_key


class LlamaCppProvider(SLMProvider):
    """
    llama.cpp provider for local SLM inference.
    
    Requires llama.cpp server running (can be in Docker).
    """
    
    def __init__(
        self,
        model_name: str = "phi-3-mini",
        endpoint: str = "http://slm-server:8080",
    ):
        super().__init__(InferenceProvider.SLM_LLAMA_CPP, model_name)
        self.endpoint = endpoint
        
        logger.info(
            "llama_cpp_provider_initialized",
            model=model_name,
            endpoint=endpoint,
        )
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> InferenceResponse:
        """Generate text using llama.cpp."""
        import httpx
        
        start_time = time.time()
        
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Format for llama.cpp
        formatted_prompt = self._format_prompt(messages)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.endpoint}/completion",
                    json={
                        "prompt": formatted_prompt,
                        "n_predict": max_tokens,
                        "temperature": temperature,
                        "stop": ["</s>", "<|end|>", "<|endoftext|>"],
                        **kwargs,
                    },
                )
                response.raise_for_status()
                data = response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            content = data.get("content", "")
            tokens = data.get("tokens_predicted", len(content.split()))
            
            # Update stats
            self.total_calls += 1
            self.total_tokens += tokens
            self.total_latency_ms += latency_ms
            
            logger.debug(
                "slm_inference_completed",
                provider="llama_cpp",
                model=self.model_name,
                tokens=tokens,
                latency_ms=latency_ms,
            )
            
            return InferenceResponse(
                content=content,
                provider=self.provider_type,
                model=self.model_name,
                tokens_used=tokens,
                latency_ms=latency_ms,
                cost=0.0,  # Local inference = $0
                metadata=data,
            )
            
        except Exception as e:
            logger.error(
                "slm_inference_failed",
                provider="llama_cpp",
                error=str(e),
                exc_info=True,
            )
            raise
    
    async def is_available(self) -> bool:
        """Check if llama.cpp server is available."""
        import httpx
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.endpoint}/health")
                return response.status_code == 200
        except Exception:
            return False
    
    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        """Format messages for llama.cpp (simple chat template)."""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\nAssistant:"
        return formatted


class OllamaProvider(SLMProvider):
    """
    Ollama provider for local SLM inference.
    
    Requires Ollama running (can be in Docker).
    """
    
    def __init__(
        self,
        model_name: str = "phi3:mini",
        endpoint: str = "http://ollama:11434",
    ):
        super().__init__(InferenceProvider.SLM_OLLAMA, model_name)
        self.endpoint = endpoint
        
        logger.info(
            "ollama_provider_initialized",
            model=model_name,
            endpoint=endpoint,
        )
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> InferenceResponse:
        """Generate text using Ollama."""
        import httpx
        
        start_time = time.time()
        
        # Build messages for Ollama format
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.endpoint}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            content = data.get("message", {}).get("content", "")
            
            # Ollama provides token counts
            eval_count = data.get("eval_count", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)
            tokens = eval_count + prompt_eval_count
            
            # Update stats
            self.total_calls += 1
            self.total_tokens += tokens
            self.total_latency_ms += latency_ms
            
            logger.debug(
                "slm_inference_completed",
                provider="ollama",
                model=self.model_name,
                tokens=tokens,
                latency_ms=latency_ms,
            )
            
            return InferenceResponse(
                content=content,
                provider=self.provider_type,
                model=self.model_name,
                tokens_used=tokens,
                latency_ms=latency_ms,
                cost=0.0,  # Local inference = $0
                metadata=data,
            )
            
        except Exception as e:
            logger.error(
                "slm_inference_failed",
                provider="ollama",
                error=str(e),
                exc_info=True,
            )
            raise
    
    async def is_available(self) -> bool:
        """Check if Ollama server is available."""
        import httpx
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.endpoint}/api/tags")
                return response.status_code == 200
        except Exception:
            return False


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude provider for cloud LLM inference.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
    ):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        super().__init__(InferenceProvider.LLM_ANTHROPIC, api_key)
        self.model = model
        
        # Initialize Anthropic client
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        
        # Pricing (per million tokens)
        self.pricing = {
            "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
            "claude-opus-4-20241229": {"input": 15.0, "output": 75.0},
        }
        
        logger.info(
            "anthropic_provider_initialized",
            model=model,
        )
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> InferenceResponse:
        """Generate text using Anthropic Claude."""
        from common.concurrency import run_io_bound
        
        start_time = time.time()
        
        try:
            # Build messages
            messages = [{"role": "user", "content": prompt}]
            
            # Call Claude API in thread pool (blocking I/O)
            response = await run_io_bound(
                self.client.messages.create,
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a helpful AI assistant.",
                messages=messages,
                **kwargs,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            content = response.content[0].text
            
            # Calculate token usage and cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens
            
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            # Update stats
            self.total_calls += 1
            self.total_tokens += total_tokens
            self.total_latency_ms += latency_ms
            self.total_cost += cost
            
            logger.debug(
                "llm_inference_completed",
                provider="anthropic",
                model=self.model,
                tokens=total_tokens,
                latency_ms=latency_ms,
                cost=cost,
            )
            
            return InferenceResponse(
                content=content,
                provider=self.provider_type,
                model=self.model,
                tokens_used=total_tokens,
                latency_ms=latency_ms,
                cost=cost,
                metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "stop_reason": response.stop_reason,
                },
            )
            
        except Exception as e:
            logger.error(
                "llm_inference_failed",
                provider="anthropic",
                error=str(e),
                exc_info=True,
            )
            raise
    
    async def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        return bool(self.api_key)
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD."""
        pricing = self.pricing.get(self.model, self.pricing["claude-sonnet-4-5-20250929"])
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost