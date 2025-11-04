"""
Hybrid Inference Engine.

Intelligently routes inference requests to SLM or LLM based on:
1. Task complexity (from TaskMaster classification)
2. Manual override (force_llm parameter)
3. Fallback on SLM failure

Features:
- Automatic SLM/LLM routing
- Fallback handling
- Performance tracking
- Cost optimization
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Optional

import structlog

from .classifier import TaskClassifier, TaskComplexity, create_classifier
from .providers import (
    InferenceProvider,
    InferenceResponse,
    AnthropicProvider,
    LlamaCppProvider,
    OllamaProvider,
)

logger = structlog.get_logger(__name__)


@dataclass
class HybridInferenceResult:
    """Result from hybrid inference."""
    
    content: str
    provider: InferenceProvider
    model: str
    complexity: TaskComplexity
    tokens_used: int
    latency_ms: float
    cost: float = 0.0
    fallback_occurred: bool = False
    metadata: dict[str, Any] | None = None


class HybridInferenceEngine:
    """
    Hybrid inference engine that routes to SLM or LLM.
    
    Decision flow:
    1. Check for force_llm override
    2. Check for pre-classified complexity (from TaskMaster)
    3. Classify complexity locally if needed
    4. Route to SLM (simple/moderate) or LLM (complex)
    5. Fallback to LLM if SLM fails
    
    Example:
        engine = HybridInferenceEngine(llm_client)
        result = await engine.infer("Parse this JSON: {...}")
        # Routes to SLM automatically
        
        result = await engine.infer("Develop a strategic plan...", force_llm=True)
        # Routes to LLM explicitly
    """
    
    def __init__(
        self,
        llm_client: Any,
        slm_provider: str = "llama_cpp",
        slm_endpoint: Optional[str] = None,
        enable_fallback: bool = True,
        classifier: Optional[TaskClassifier] = None,
    ):
        """
        Initialize hybrid inference engine.
        
        Args:
            llm_client: Anthropic client for LLM inference
            slm_provider: SLM provider type ("llama_cpp" or "ollama")
            slm_endpoint: SLM server endpoint (defaults from env)
            enable_fallback: Enable fallback from SLM to LLM on failure
            classifier: Task complexity classifier (created if not provided)
        """
        self.llm_client = llm_client
        self.enable_fallback = enable_fallback
        
        # Initialize classifier
        self.classifier = classifier or create_classifier()
        
        # Initialize providers
        self.llm_provider = self._init_llm_provider()
        self.slm_provider = self._init_slm_provider(slm_provider, slm_endpoint)
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "slm_requests": 0,
            "llm_requests": 0,
            "fallbacks": 0,
            "total_cost": 0.0,
            "total_latency_ms": 0.0,
            "complexity_distribution": {
                "simple": 0,
                "moderate": 0,
                "complex": 0,
            },
        }
        
        logger.info(
            "hybrid_inference_engine_initialized",
            slm_provider=slm_provider,
            llm_model=self.llm_provider.model if self.llm_provider else None,
            fallback_enabled=enable_fallback,
        )
    
    def _init_llm_provider(self) -> AnthropicProvider:
        """Initialize LLM provider (Anthropic)."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model = os.getenv("LLM_MODEL", "claude-sonnet-4-5-20250929")
        
        return AnthropicProvider(api_key=api_key, model=model)
    
    def _init_slm_provider(
        self,
        provider_type: str,
        endpoint: Optional[str],
    ) -> LlamaCppProvider | OllamaProvider:
        """Initialize SLM provider."""
        if provider_type == "llama_cpp":
            endpoint = endpoint or os.getenv("SLM_ENDPOINT", "http://slm-server:8080")
            model = os.getenv("SLM_MODEL", "phi-3-mini")
            return LlamaCppProvider(model_name=model, endpoint=endpoint)
        
        elif provider_type == "ollama":
            endpoint = endpoint or os.getenv("OLLAMA_ENDPOINT", "http://ollama:11434")
            model = os.getenv("SLM_MODEL", "phi3:mini")
            return OllamaProvider(model_name=model, endpoint=endpoint)
        
        else:
            raise ValueError(f"Unknown SLM provider: {provider_type}")
    
    async def infer(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        force_llm: bool = False,
        pre_classified_complexity: Optional[str] = None,
        **kwargs: Any,
    ) -> HybridInferenceResult:
        """
        Perform hybrid inference with automatic SLM/LLM routing.
        
        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            force_llm: Force use of LLM regardless of complexity
            pre_classified_complexity: Pre-classified complexity from TaskMaster
            **kwargs: Additional model-specific parameters
            
        Returns:
            HybridInferenceResult with content and metadata
        """
        self.stats["total_requests"] += 1
        
        # Determine complexity
        if pre_classified_complexity:
            # Use pre-classification from TaskMaster (orchestrator)
            complexity = TaskComplexity(pre_classified_complexity)
            logger.debug(
                "using_preclassified_complexity",
                complexity=complexity.value,
            )
        else:
            # Classify locally
            classification = self.classifier.classify(prompt, context=system)
            complexity = classification.complexity
            logger.debug(
                "classified_complexity",
                complexity=complexity.value,
                confidence=classification.confidence,
            )
        
        # Update complexity distribution
        self.stats["complexity_distribution"][complexity.value] += 1
        
        # Decide routing
        use_llm = force_llm or complexity == TaskComplexity.COMPLEX
        
        if use_llm:
            # Route to LLM
            return await self._infer_llm(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                complexity=complexity,
                **kwargs,
            )
        else:
            # Route to SLM (with optional fallback to LLM)
            return await self._infer_slm_with_fallback(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                complexity=complexity,
                **kwargs,
            )
    
    async def _infer_llm(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        complexity: TaskComplexity,
        **kwargs: Any,
    ) -> HybridInferenceResult:
        """Perform LLM inference."""
        self.stats["llm_requests"] += 1
        
        logger.debug(
            "routing_to_llm",
            complexity=complexity.value,
        )
        
        try:
            response = await self.llm_provider.generate(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            
            # Update stats
            self.stats["total_cost"] += response.cost
            self.stats["total_latency_ms"] += response.latency_ms
            
            return HybridInferenceResult(
                content=response.content,
                provider=response.provider,
                model=response.model,
                complexity=complexity,
                tokens_used=response.tokens_used,
                latency_ms=response.latency_ms,
                cost=response.cost,
                fallback_occurred=False,
                metadata=response.metadata,
            )
            
        except Exception as e:
            logger.error(
                "llm_inference_failed",
                error=str(e),
                exc_info=True,
            )
            raise
    
    async def _infer_slm_with_fallback(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        complexity: TaskComplexity,
        **kwargs: Any,
    ) -> HybridInferenceResult:
        """Perform SLM inference with optional fallback to LLM."""
        self.stats["slm_requests"] += 1
        
        logger.debug(
            "routing_to_slm",
            complexity=complexity.value,
        )
        
        try:
            # Check SLM availability
            if not await self.slm_provider.is_available():
                logger.warning("slm_not_available", provider=self.slm_provider.provider_type.value)
                if self.enable_fallback:
                    return await self._fallback_to_llm(
                        prompt, system, max_tokens, temperature, complexity, **kwargs
                    )
                else:
                    raise RuntimeError("SLM not available and fallback disabled")
            
            # Perform SLM inference
            response = await self.slm_provider.generate(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            
            # Update stats
            self.stats["total_latency_ms"] += response.latency_ms
            
            return HybridInferenceResult(
                content=response.content,
                provider=response.provider,
                model=response.model,
                complexity=complexity,
                tokens_used=response.tokens_used,
                latency_ms=response.latency_ms,
                cost=0.0,  # SLM is free (self-hosted)
                fallback_occurred=False,
                metadata=response.metadata,
            )
            
        except Exception as e:
            logger.warning(
                "slm_inference_failed",
                error=str(e),
                will_fallback=self.enable_fallback,
            )
            
            if self.enable_fallback:
                return await self._fallback_to_llm(
                    prompt, system, max_tokens, temperature, complexity, **kwargs
                )
            else:
                raise
    
    async def _fallback_to_llm(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        complexity: TaskComplexity,
        **kwargs: Any,
    ) -> HybridInferenceResult:
        """Fallback to LLM when SLM fails."""
        self.stats["fallbacks"] += 1
        self.stats["llm_requests"] += 1
        
        logger.info("falling_back_to_llm")
        
        response = await self.llm_provider.generate(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        
        # Update stats
        self.stats["total_cost"] += response.cost
        self.stats["total_latency_ms"] += response.latency_ms
        
        return HybridInferenceResult(
            content=response.content,
            provider=response.provider,
            model=response.model,
            complexity=complexity,
            tokens_used=response.tokens_used,
            latency_ms=response.latency_ms,
            cost=response.cost,
            fallback_occurred=True,
            metadata=response.metadata,
        )
    
    def get_stats(self) -> dict[str, Any]:
        """Get inference statistics."""
        total = self.stats["total_requests"]
        avg_latency = (
            self.stats["total_latency_ms"] / total
            if total > 0
            else 0
        )
        
        slm_percentage = (
            self.stats["slm_requests"] / total * 100
            if total > 0
            else 0
        )
        
        return {
            **self.stats,
            "avg_latency_ms": avg_latency,
            "slm_percentage": slm_percentage,
            "llm_percentage": 100 - slm_percentage,
            "slm_provider": self.slm_provider.provider_type.value,
            "llm_provider": self.llm_provider.provider_type.value,
        }
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.stats = {
            "total_requests": 0,
            "slm_requests": 0,
            "llm_requests": 0,
            "fallbacks": 0,
            "total_cost": 0.0,
            "total_latency_ms": 0.0,
            "complexity_distribution": {
                "simple": 0,
                "moderate": 0,
                "complex": 0,
            },
        }