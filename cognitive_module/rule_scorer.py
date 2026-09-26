"""
cognitive_module.rule_scorer
────────────────────────────
Rule-based cognitive scorer. Consumes FeatureExtractor output
and produces a CognitiveProfile using hand-crafted heuristics.

This is the "always available" scorer — no ML dependencies required.
It also serves as the training-data labeller for fine-tuning DistilBERT.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .models import CognitiveProfile, TaskType
from .config import CognitiveConfig, DEFAULT_CONFIG
from .feature_extractor import (
    FeatureExtractor,
    ERROR_TRACE_RE,
    REASONING_MARKERS,
    CREATIVE_MARKERS,
    PRECISION_MARKERS,
)


class RuleBasedScorer:
    """
    Heuristic cognitive scorer.
    Extracts features, applies rules, and returns a CognitiveProfile.
    """

    def __init__(self, config: CognitiveConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self._extractor = FeatureExtractor()

    def score(self, prompt: str) -> CognitiveProfile:
        """Analyse a prompt and return a full CognitiveProfile."""
        features = self._extractor.extract(prompt)
        lower = prompt.strip().lower()
        signals: List[str] = []

        # ── Map features → 6 dimensions ──────────────────────────────────

        # 1. Reasoning Depth
        reasoning = self._calc_reasoning(features, lower, signals)

        # 2. Domain Specificity
        domain = int(features["domain_specificity_score"])
        if domain > 0:
            signals.append(f"domain_score:{domain}")

        # 3. Code Complexity
        code = int(features["code_complexity_score"])
        if code > 0:
            signals.append(f"code_score:{code}")

        # 4. Creative Demand
        creative = int(features["creative_demand_score"])
        if creative > 0:
            signals.append(f"creative_score:{creative}")

        # 5. Precision Required
        precision = int(features["precision_required_score"])
        if precision > 0:
            signals.append(f"precision_score:{precision}")

        # 6. Structural Complexity
        structure = int(features["structural_complexity_score"])
        if structure > 0:
            signals.append(f"structure_score:{structure}")

        # Build profile
        profile = CognitiveProfile(
            reasoning_depth=min(reasoning, 100),
            domain_specificity=min(domain, 100),
            code_complexity=min(code, 100),
            creative_demand=min(creative, 100),
            precision_required=min(precision, 100),
            structural_complexity=min(structure, 100),
            signals=signals[:self.config.max_signals],
            scorer_used="rule",
        )

        # Detect task type
        task_type, confidence = self._detect_task_type(profile, lower, features)
        profile.task_type = task_type
        profile.confidence = confidence

        return profile

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _calc_reasoning(
        self, features: Dict[str, float], lower: str, signals: List[str]
    ) -> int:
        """Calculate reasoning depth from features."""
        score = int(features["reasoning_keyword_score"])

        # Question depth
        score += int(features["question_depth_score"])

        # Sentence complexity
        avg_words = features["avg_words_per_sentence"]
        if avg_words > 25:
            score += 14
            signals.append("complex_sentences")
        elif avg_words > 18:
            score += 10
            signals.append("moderate_sentences")
        elif avg_words > 12:
            score += 5

        # Conditionals
        conds = int(features["conditional_count"])
        if conds >= 4:
            score += 18
            signals.append(f"conditionals:{conds}")
        elif conds >= 2:
            score += 10
            signals.append(f"conditionals:{conds}")
        elif conds >= 1:
            score += 6

        # Word count bonus — longer prompts tend to be more complex
        word_count = features.get("word_count", 0)
        if word_count >= 150:
            score += 12
        elif word_count >= 80:
            score += 8
        elif word_count >= 30:
            score += 4

        if score > 0:
            signals.append(f"reasoning_score:{min(score, 100)}")

        return score

    def _detect_task_type(
        self,
        profile: CognitiveProfile,
        lower: str,
        features: Dict[str, float],
    ) -> Tuple[TaskType, float]:
        """Determine primary task type and classification confidence."""

        scores: Dict[TaskType, float] = {t: 0.0 for t in TaskType}

        # Code / Debugging
        if profile.code_complexity >= 25:
            has_error = bool(ERROR_TRACE_RE.search(lower))
            has_debug = features.get("has_debug_markers", 0) > 0
            if has_error or has_debug:
                scores[TaskType.DEBUGGING] = profile.code_complexity * 1.2
            else:
                scores[TaskType.CODE] = profile.code_complexity * 1.0

        # Mathematical
        if profile.precision_required >= 25:
            if features.get("has_math_task_markers", 0) > 0:
                scores[TaskType.MATHEMATICAL] = profile.precision_required * 1.1

        # Creative
        if profile.creative_demand >= 20:
            scores[TaskType.CREATIVE] = profile.creative_demand * 1.0

        # Analytical
        if profile.reasoning_depth >= 25:
            scores[TaskType.ANALYTICAL] = profile.reasoning_depth * 0.9

        # Multi-step
        if profile.structural_complexity >= 25:
            scores[TaskType.MULTI_STEP] = profile.structural_complexity * 0.85

        # Factual — but NOT when code signals are also strong
        if (profile.domain_specificity >= 20 and profile.reasoning_depth < 30
                and profile.code_complexity < 20):
            scores[TaskType.FACTUAL] = profile.domain_specificity * 0.7

        # Conversational (default — only when ALL scores are trivially low)
        dim_avg = sum(profile.dimension_vector()) / 6
        max_dim = max(profile.dimension_vector())
        if dim_avg < 8 and max_dim < 15:
            scores[TaskType.CONVERSATIONAL] = max(30, 60 - dim_avg)

        # Pick highest
        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_type]

        # Confidence = how dominant the top type is vs the runner-up
        sorted_scores = sorted(scores.values(), reverse=True)
        if sorted_scores[0] > 0 and len(sorted_scores) > 1:
            gap = 1 - (sorted_scores[1] / sorted_scores[0]) if sorted_scores[0] else 1.0
            confidence = min(0.5 + gap * 0.5, 1.0)
        else:
            confidence = 0.5 if best_score > 0 else 0.3

        return best_type, round(confidence, 2)

    @property
    def extractor(self) -> FeatureExtractor:
        """Expose the feature extractor for external use (training data, etc.)."""
        return self._extractor
