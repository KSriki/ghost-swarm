"""
Task Complexity Classifier.

Analyzes incoming tasks and classifies them as SIMPLE, MODERATE, or COMPLEX
to enable intelligent routing to SLM or LLM.

Uses heuristic-based classification with optional ML-based classifier.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class TaskComplexity(str, Enum):
    """Task complexity levels."""
    
    SIMPLE = "simple"  # SLM can handle easily
    MODERATE = "moderate"  # SLM can handle with fine-tuning
    COMPLEX = "complex"  # Requires LLM


@dataclass
class ClassificationResult:
    """Result of task classification."""
    
    complexity: TaskComplexity
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Why this classification was chosen
    features: dict[str, float]  # Feature scores used in classification


class TaskClassifier:
    """
    Heuristic-based task complexity classifier.
    
    Analyzes task prompts to determine if they can be handled by SLM or need LLM.
    
    Classification rules (based on NVIDIA research):
    - SIMPLE: Routing, parsing, JSON generation, status updates, simple Q&A
    - MODERATE: Summaries, structured generation, data extraction, simple reasoning
    - COMPLEX: Multi-step reasoning, creative writing, strategic planning, open-ended tasks
    """
    
    def __init__(
        self,
        simple_threshold: float = 0.3,
        complex_threshold: float = 0.7,
    ):
        """
        Initialize classifier.
        
        Args:
            simple_threshold: Score below this = SIMPLE
            complex_threshold: Score above this = COMPLEX
            Between thresholds = MODERATE
        """
        self.simple_threshold = simple_threshold
        self.complex_threshold = complex_threshold
        
        # Keywords indicating complexity
        self.simple_keywords = {
            "parse", "extract", "format", "convert", "validate",
            "route", "classify", "categorize", "filter", "sort",
            "json", "status", "update", "check", "verify",
        }
        
        self.complex_keywords = {
            "analyze", "evaluate", "assess", "critique", "design",
            "strategy", "plan", "brainstorm", "creative", "novel",
            "explain why", "compare and contrast", "pros and cons",
            "implications", "consequences", "trade-offs",
        }
        
        logger.info(
            "classifier_initialized",
            simple_threshold=simple_threshold,
            complex_threshold=complex_threshold,
        )
    
    def classify(self, prompt: str, context: Optional[str] = None) -> ClassificationResult:
        """
        Classify task complexity.
        
        Args:
            prompt: Task prompt to classify
            context: Optional additional context
            
        Returns:
            ClassificationResult with complexity and reasoning
        """
        # Extract features
        features = self._extract_features(prompt, context)
        
        # Calculate complexity score (0.0 = simple, 1.0 = complex)
        score = self._calculate_complexity_score(features)
        
        # Determine complexity level
        if score < self.simple_threshold:
            complexity = TaskComplexity.SIMPLE
            reasoning = self._explain_simple(features)
        elif score > self.complex_threshold:
            complexity = TaskComplexity.COMPLEX
            reasoning = self._explain_complex(features)
        else:
            complexity = TaskComplexity.MODERATE
            reasoning = self._explain_moderate(features)
        
        # Calculate confidence (how far from thresholds)
        if complexity == TaskComplexity.SIMPLE:
            confidence = 1.0 - (score / self.simple_threshold)
        elif complexity == TaskComplexity.COMPLEX:
            confidence = (score - self.complex_threshold) / (1.0 - self.complex_threshold)
        else:
            # Moderate: confidence based on distance from both thresholds
            dist_to_simple = abs(score - self.simple_threshold)
            dist_to_complex = abs(score - self.complex_threshold)
            confidence = min(dist_to_simple, dist_to_complex) * 2  # Normalize
        
        confidence = min(1.0, max(0.0, confidence))  # Clamp to [0, 1]
        
        logger.debug(
            "task_classified",
            complexity=complexity.value,
            score=score,
            confidence=confidence,
            prompt_length=len(prompt),
        )
        
        return ClassificationResult(
            complexity=complexity,
            confidence=confidence,
            reasoning=reasoning,
            features=features,
        )
    
    def _extract_features(self, prompt: str, context: Optional[str] = None) -> dict[str, float]:
        """Extract features from prompt for classification."""
        text = prompt.lower()
        if context:
            text += " " + context.lower()
        
        features = {}
        
        # Length features (longer = more complex)
        features["length"] = min(1.0, len(prompt) / 1000)  # Normalize to [0, 1]
        features["word_count"] = min(1.0, len(prompt.split()) / 200)
        
        # Keyword features
        features["simple_keywords"] = self._count_keywords(text, self.simple_keywords)
        features["complex_keywords"] = self._count_keywords(text, self.complex_keywords)
        
        # Question features
        features["question_marks"] = min(1.0, text.count("?") / 3)
        features["multi_part"] = 1.0 if ("first" in text and "second" in text) or ("1." in text and "2." in text) else 0.0
        
        # Reasoning indicators
        features["reasoning_required"] = 1.0 if any(
            word in text for word in ["why", "how", "explain", "because", "therefore"]
        ) else 0.0
        
        # Structured output indicators (simpler)
        features["structured_output"] = 1.0 if any(
            word in text for word in ["json", "format", "structure", "template", "schema"]
        ) else 0.0
        
        # Code-related (moderate complexity)
        features["code_related"] = 1.0 if any(
            word in text for word in ["code", "function", "class", "implement", "debug"]
        ) else 0.0
        
        # Creative/open-ended (complex)
        features["creative"] = 1.0 if any(
            word in text for word in ["creative", "imagine", "design", "invent", "novel", "brainstorm"]
        ) else 0.0
        
        return features
    
    def _count_keywords(self, text: str, keywords: set[str]) -> float:
        """Count keyword matches, normalized."""
        matches = sum(1 for keyword in keywords if keyword in text)
        return min(1.0, matches / 3)  # Normalize
    
    def _calculate_complexity_score(self, features: dict[str, float]) -> float:
        """Calculate complexity score from features."""
        # Weighted sum of features
        weights = {
            "length": 0.1,
            "word_count": 0.1,
            "simple_keywords": -0.3,  # Negative = simpler
            "complex_keywords": 0.3,
            "question_marks": 0.1,
            "multi_part": 0.2,
            "reasoning_required": 0.3,
            "structured_output": -0.2,  # Negative = simpler
            "code_related": 0.15,
            "creative": 0.4,
        }
        
        score = 0.5  # Start at middle
        for feature, weight in weights.items():
            score += features.get(feature, 0.0) * weight
        
        # Clamp to [0, 1]
        return min(1.0, max(0.0, score))
    
    def _explain_simple(self, features: dict[str, float]) -> str:
        """Explain why task is simple."""
        reasons = []
        
        if features.get("simple_keywords", 0) > 0:
            reasons.append("contains simple task keywords (parse, format, extract)")
        if features.get("structured_output", 0) > 0:
            reasons.append("requests structured output (JSON, format)")
        if features.get("length", 0) < 0.3:
            reasons.append("short prompt")
        
        if not reasons:
            reasons.append("no complex reasoning or creativity required")
        
        return "Task classified as SIMPLE: " + ", ".join(reasons)
    
    def _explain_moderate(self, features: dict[str, float]) -> str:
        """Explain why task is moderate."""
        reasons = []
        
        if features.get("code_related", 0) > 0:
            reasons.append("code-related task")
        if features.get("reasoning_required", 0) > 0:
            reasons.append("requires some reasoning")
        if 0.3 < features.get("length", 0) < 0.7:
            reasons.append("medium-length task")
        
        if not reasons:
            reasons.append("balanced complexity")
        
        return "Task classified as MODERATE: " + ", ".join(reasons)
    
    def _explain_complex(self, features: dict[str, float]) -> str:
        """Explain why task is complex."""
        reasons = []
        
        if features.get("creative", 0) > 0:
            reasons.append("creative/open-ended task")
        if features.get("complex_keywords", 0) > 0:
            reasons.append("contains complex reasoning keywords (analyze, evaluate, strategy)")
        if features.get("multi_part", 0) > 0:
            reasons.append("multi-part question")
        if features.get("length", 0) > 0.7:
            reasons.append("long, detailed prompt")
        
        if not reasons:
            reasons.append("requires sophisticated reasoning")
        
        return "Task classified as COMPLEX: " + ", ".join(reasons)


def create_classifier(
    simple_threshold: float = 0.3,
    complex_threshold: float = 0.7,
) -> TaskClassifier:
    """
    Factory function to create a task classifier.
    
    Args:
        simple_threshold: Score below this = SIMPLE
        complex_threshold: Score above this = COMPLEX
        
    Returns:
        Configured TaskClassifier instance
    """
    return TaskClassifier(
        simple_threshold=simple_threshold,
        complex_threshold=complex_threshold,
    )