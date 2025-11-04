"""
Hybrid Inference System for Ghost Swarm.

Provides intelligent routing between Small Language Models (SLMs) and 
Large Language Models (LLMs) based on task complexity.

Components:
- TaskClassifier: Classifies task complexity (simple/moderate/complex)
- HybridInferenceEngine: Routes inference to SLM or LLM
- SLM Providers: Local SLM inference (llama.cpp, Ollama)
- LLM Providers: Cloud LLM inference (Anthropic Claude)
"""

from .classifier import (
    TaskClassifier,
    TaskComplexity,
    ClassificationResult,
    create_classifier,
)
from .engine import (
    HybridInferenceEngine,
    HybridInferenceResult,
    InferenceProvider,
)
from .providers import (
    SLMProvider,
    LLMProvider,
    LlamaCppProvider,
    OllamaProvider,
    AnthropicProvider,
)

__all__ = [
    # Classifier
    "TaskClassifier",
    "TaskComplexity",
    "ClassificationResult",
    "create_classifier",
    # Engine
    "HybridInferenceEngine",
    "HybridInferenceResult",
    "InferenceProvider",
    # Providers
    "SLMProvider",
    "LLMProvider",
    "LlamaCppProvider",
    "OllamaProvider",
    "AnthropicProvider",
]