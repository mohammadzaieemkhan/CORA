"""
CORA Cognitive Module — Multi-Dimensional Prompt Analysis Engine

Replaces the simple keyword-matching scorer with a 6-dimensional cognitive
complexity analyser that classifies prompts across:

  1. Reasoning Depth       — multi-step logic, causal chains
  2. Domain Specificity     — technical jargon, specialised knowledge
  3. Code Complexity        — code patterns, language detection, debugging
  4. Creative Demand        — open-ended generation, originality
  5. Precision Required     — factual accuracy, mathematical rigor
  6. Structural Complexity  — multi-part tasks, nested questions, constraints
"""

from __future__ import annotations

import re
import math
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

# ────────────────────────────────────────────────────────────────────────────────
#  Task Types
# ────────────────────────────────────────────────────────────────────────────────
class TaskType(str, Enum):
    FACTUAL        = "factual"
    ANALYTICAL     = "analytical"
    CODE           = "code"
    DEBUGGING      = "debugging"
    CREATIVE       = "creative"
    CONVERSATIONAL = "conversational"
    MATHEMATICAL   = "mathematical"
    MULTI_STEP     = "multi_step"


TASK_TYPE_META: Dict[TaskType, Dict] = {
    TaskType.FACTUAL:        {"icon": "📖", "label": "Factual",        "boost": 0},
    TaskType.ANALYTICAL:     {"icon": "🧠", "label": "Analytical",     "boost": 10},
    TaskType.CODE:           {"icon": "💻", "label": "Code",           "boost": 15},
    TaskType.DEBUGGING:      {"icon": "🔧", "label": "Debugging",      "boost": 20},
    TaskType.CREATIVE:       {"icon": "🎨", "label": "Creative",       "boost": 5},
    TaskType.CONVERSATIONAL: {"icon": "💬", "label": "Conversational", "boost": -5},
    TaskType.MATHEMATICAL:   {"icon": "📐", "label": "Mathematical",   "boost": 12},
    TaskType.MULTI_STEP:     {"icon": "📋", "label": "Multi-Step",     "boost": 15},
}


# ────────────────────────────────────────────────────────────────────────────────
#  Cognitive Profile (result dataclass)
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class CognitiveProfile:
    reasoning_depth: int        = 0
    domain_specificity: int     = 0
    code_complexity: int        = 0
    creative_demand: int        = 0
    precision_required: int     = 0
    structural_complexity: int  = 0
    task_type: TaskType         = TaskType.CONVERSATIONAL
    confidence: float           = 0.0   # 0.0 – 1.0
    signals: List[str]          = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["task_type"] = self.task_type.value
        return d


# ────────────────────────────────────────────────────────────────────────────────
#  Dimension Weights (for budget-score aggregation)
# ────────────────────────────────────────────────────────────────────────────────
DIMENSION_WEIGHTS = {
    "reasoning_depth":       0.25,
    "code_complexity":       0.25,
    "domain_specificity":    0.15,
    "structural_complexity": 0.15,
    "creative_demand":       0.10,
    "precision_required":    0.10,
}


# ────────────────────────────────────────────────────────────────────────────────
#  Vocabulary / Pattern Banks
# ────────────────────────────────────────────────────────────────────────────────

# — Reasoning —
REASONING_MARKERS = {
    # causal / logical connectors
    "because": 6, "therefore": 8, "consequently": 8, "implies": 7,
    "hence": 7, "thus": 7, "as a result": 8,
    # deep-analysis verbs
    "analyze": 12, "analyse": 12, "evaluate": 11, "compare": 10,
    "contrast": 10, "synthesize": 14, "synthesise": 14,
    "assess": 10, "justify": 11, "critique": 12,
    # reasoning-depth signals
    "step by step": 14, "reasoning": 10, "logic": 8,
    "trade-off": 10, "tradeoff": 10, "pros and cons": 12,
    "why does": 10, "why do": 10, "why is": 8, "how does": 10,
    "how do": 9, "what causes": 10, "root cause": 12,
    "implications": 10, "impact of": 8,
    "debug": 14, "recursive": 12, "dynamic programming": 15,
    "implement": 10, "optimize": 12, "fix": 8, "solve": 10,
    "algorithm": 12, "complexity": 10, "explain": 8,
    "geopolitical": 12, "impact": 8, "regulation": 10,
    "policy": 10, "difference between": 10,
    "implications of": 12, "across": 6, "impact of": 10,
    "effects of": 10, "consequences": 10,
    "relationship between": 12, "role of": 8,
    "prove": 15, "proof": 15, "formal": 12, "theorem": 14,
    "derive": 12, "design": 8, "construct": 10,
    "red-black": 14, "balancing": 10, "insertion": 8,
    "deletion": 8, "np": 12, "complexity theory": 15,
    "formal proof": 16,
}

# — Domain-specific vocabulary banks —
DOMAIN_VOCAB: Dict[str, List[str]] = {
    "computer_science": [
        "algorithm", "data structure", "binary tree", "hash map", "linked list",
        "recursion", "dynamic programming", "big o", "time complexity",
        "space complexity", "polymorphism", "inheritance", "encapsulation",
        "microservice", "kubernetes", "docker", "rest api", "graphql",
        "database", "sql", "nosql", "tcp", "http", "websocket",
        "machine learning", "neural network", "transformer", "attention mechanism",
        "gradient descent", "backpropagation", "convolution", "embedding",
        "recursive", "memoization", "backtracking", "greedy", "divide and conquer",
        "sorting", "searching", "geopolitical", "regulation", "policy", "implications",
        "red-black tree", "avl tree", "b-tree", "trie", "heap",
        "insertion", "deletion", "balancing", "traversal",
        "np", "np-hard", "np-complete", "complexity theory",
        "formal", "proof", "theorem", "lemma",
        "c++", "rust", "golang", "typescript",
        "implement", "red-black",
    ],
    "medicine": [
        "diagnosis", "symptom", "pathology", "pharmacology", "dosage",
        "clinical trial", "prognosis", "etiology", "epidemiology",
        "cardiac", "neurological", "oncology", "metabolic",
    ],
    "law": [
        "statute", "jurisdiction", "tort", "liability", "precedent",
        "plaintiff", "defendant", "contract law", "intellectual property",
        "compliance", "regulation", "amendment",
    ],
    "science": [
        "hypothesis", "experiment", "variable", "quantum", "thermodynamics",
        "molecular", "photosynthesis", "genome", "evolution", "entropy",
        "electromagnetic", "relativity", "particle physics",
    ],
    "finance": [
        "portfolio", "derivative", "hedge", "equity", "bond", "yield",
        "amortization", "compound interest", "black-scholes", "volatility",
        "market cap", "dividend", "ipo", "valuation",
    ],
}

# — Code patterns —
CODE_KEYWORDS = {
    "function", "def ", "class ", "import ", "const ", "let ", "var ",
    "return ", "if ", "else ", "for ", "while ", "try ", "catch ",
    "except ", "raise ", "throw ", "async ", "await ", "yield ",
    "public ", "private ", "static ", "void ", "int ", "string ",
    "print(", "console.log", "fmt.println", "system.out",
    "recursive", "recursion", "dynamic programming", "dp", "algorithm",
    "implement", "function that", "write a function", "write a program",
    "debug", "fix this", "binary search", "sorting", "linked list",
    "stack", "queue", "tree", "graph", "hash",
    "red-black", "insertion", "deletion", "balancing",
    "in c++", "in rust", "in java", "in python",
}

CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")
ERROR_TRACE_PATTERN = re.compile(
    r"(Traceback|Error:|Exception:|FATAL|panic:|Segmentation fault"
    r"|TypeError|ValueError|KeyError|IndexError|SyntaxError"
    r"|NullPointerException|undefined is not|Cannot read propert)",
    re.IGNORECASE,
)
FILE_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:\\|/(?:usr|etc|home|var|opt)/)[\w/\\.\-]+"
)

# — Creative —
CREATIVE_MARKERS = {
    "write a story": 15, "write a poem": 15, "creative": 10,
    "imagine": 10, "brainstorm": 12, "come up with": 10,
    "generate ideas": 12, "fiction": 12, "narrative": 10,
    "design a": 8, "invent": 10, "original": 6,
    "compose": 10, "draft": 8, "rewrite": 6,
}

# — Precision —
PRECISION_MARKERS = {
    "exact": 8, "precisely": 10, "accurate": 8, "citation": 12,
    "source": 6, "reference": 6, "peer-reviewed": 14,
    "calculate": 14, "compute": 10, "derive": 12,
    "proof": 14, "prove": 12, "verify": 10, "prove that": 16, "p != np": 20,
    "np-complete": 16, "formal proof": 16, "complexity theory": 15,
    "theorem": 14, "lemma": 14, "corollary": 14,
    "significant figures": 10, "decimal places": 8,
    "how many": 10, "how much": 10, "how far": 10, "how long": 10,
    "total": 8, "per hour": 10, "miles": 8, "kilometers": 8,
    "percent": 12, "percentage": 12, "what is the": 6, "how old": 10,
    "if x": 10, "solve for": 14, "word problem": 12,
}

MATH_PATTERN = re.compile(
    r"(?:\d+\s*[\+\-\*\/\=\^]\s*\d+|∫|∑|∏|√|≈|≠|≤|≥|lim|log|sin|cos|tan"
    r"|matrix|determinant|eigenvalue|∂|∇|dx|dy)",
    re.IGNORECASE,
)

# — Structural —
NUMBERED_LIST_PATTERN = re.compile(r"^\s*\d+[\.\)]\s", re.MULTILINE)
BULLET_LIST_PATTERN = re.compile(r"^\s*[-\*•]\s", re.MULTILINE)
CONSTRAINT_MARKERS = [
    "must", "should", "ensure", "constraint", "requirement",
    "at least", "at most", "no more than", "between",
    "given that", "assuming", "if and only if",
]
MULTI_PART_MARKERS = [
    "first", "second", "third", "finally", "additionally",
    "moreover", "furthermore", "also", "next", "then",
    "part 1", "part 2", "part a", "part b",
]


# ────────────────────────────────────────────────────────────────────────────────
#  Individual Dimension Scorers
# ────────────────────────────────────────────────────────────────────────────────

def _score_reasoning_depth(text: str, lower: str, sentences: List[str]) -> Tuple[int, List[str]]:
    """Score 0-100: How much multi-step/causal reasoning is required."""
    score = 0
    signals: List[str] = []

    # Keyword hits
    for phrase, pts in REASONING_MARKERS.items():
        if phrase in lower:
            score += pts
            signals.append(f"reasoning_keyword:{phrase}")

    # Question depth — multiple questions = deeper reasoning needed
    q_count = text.count("?")
    if q_count >= 4:
        score += 18
        signals.append(f"deep_questioning:{q_count}q")
    elif q_count >= 2:
        score += 10
        signals.append(f"multi_question:{q_count}q")
    elif q_count >= 1:
        score += 4

    # Sentence complexity — longer sentences tend to encode more reasoning
    if sentences:
        avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_words > 25:
            score += 12
            signals.append("complex_sentences")
        elif avg_words > 15:
            score += 6

    # Conditional structures
    conditionals = len(re.findall(r"\b(?:if|when|unless|whether|suppose|assuming)\b", lower))
    if conditionals >= 3:
        score += 14
        signals.append(f"conditionals:{conditionals}")
    elif conditionals >= 1:
        score += 6

    return min(score, 100), signals


def _score_domain_specificity(lower: str) -> Tuple[int, List[str]]:
    """Score 0-100: How much specialised domain knowledge is needed."""
    score = 0
    signals: List[str] = []
    domain_hits: Dict[str, int] = {}

    for domain, vocab in DOMAIN_VOCAB.items():
        hits = sum(1 for term in vocab if term in lower)
        if hits > 0:
            domain_hits[domain] = hits

    if not domain_hits:
        return 0, []

    # Primary domain
    primary = max(domain_hits, key=domain_hits.get)  # type: ignore[arg-type]
    hit_count = domain_hits[primary]

    score += min(hit_count * 12, 60)
    signals.append(f"domain:{primary}({hit_count} terms)")

    # Cross-domain breadth bonus
    if len(domain_hits) >= 3:
        score += 20
        signals.append(f"cross_domain:{len(domain_hits)} fields")
    elif len(domain_hits) >= 2:
        score += 10

    return min(score, 100), signals


def _score_code_complexity(text: str, lower: str) -> Tuple[int, List[str]]:
    """Score 0-100: How much code-related reasoning is needed."""
    score = 0
    signals: List[str] = []

    # Code blocks
    code_blocks = CODE_BLOCK_PATTERN.findall(text)
    if code_blocks:
        total_lines = sum(block.count("\n") for block in code_blocks)
        score += min(20 + total_lines * 2, 50)
        signals.append(f"code_blocks:{len(code_blocks)}({total_lines}lines)")

    # Inline code
    inline = INLINE_CODE_PATTERN.findall(text)
    if inline:
        score += min(len(inline) * 4, 20)
        signals.append(f"inline_code:{len(inline)}")

    # Code keywords (even without backticks)
    kw_hits = sum(1 for kw in CODE_KEYWORDS if kw in lower)
    if kw_hits >= 5:
        score += 25
        signals.append(f"code_keywords:{kw_hits}")
    elif kw_hits >= 2:
        score += 12
        signals.append(f"code_keywords:{kw_hits}")
    elif kw_hits >= 1:
        score += 6

    # Error traces / debugging patterns
    if ERROR_TRACE_PATTERN.search(text):
        score += 20
        signals.append("error_trace_detected")

    # File paths
    if FILE_PATH_PATTERN.search(text):
        score += 8
        signals.append("file_paths_detected")

    return min(score, 100), signals


def _score_creative_demand(lower: str) -> Tuple[int, List[str]]:
    """Score 0-100: How much creative/generative output is expected."""
    score = 0
    signals: List[str] = []

    for phrase, pts in CREATIVE_MARKERS.items():
        if phrase in lower:
            score += pts
            signals.append(f"creative_marker:{phrase}")

    # Open-ended indicators
    open_ended = ["what if", "could you", "how might", "suggest", "propose"]
    hits = sum(1 for m in open_ended if m in lower)
    if hits >= 2:
        score += 12
        signals.append(f"open_ended:{hits}")
    elif hits >= 1:
        score += 6

    return min(score, 100), signals


def _score_precision_required(text: str, lower: str) -> Tuple[int, List[str]]:
    """Score 0-100: How much factual/mathematical precision is expected."""
    score = 0
    signals: List[str] = []

    for phrase, pts in PRECISION_MARKERS.items():
        if phrase in lower:
            score += pts
            signals.append(f"precision_marker:{phrase}")

    # Math symbols / expressions
    math_hits = MATH_PATTERN.findall(text)
    if math_hits:
        score += min(len(math_hits) * 8, 40)
        signals.append(f"math_expressions:{len(math_hits)}")

    # Numeric density — lots of numbers suggest precision-oriented tasks
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    if len(numbers) >= 6:
        score += 15
        signals.append(f"numeric_density:{len(numbers)}")
    elif len(numbers) >= 3:
        score += 8

    return min(score, 100), signals


def _score_structural_complexity(text: str, lower: str, sentences: List[str]) -> Tuple[int, List[str]]:
    """Score 0-100: How structurally complex / multi-part the prompt is."""
    score = 0
    signals: List[str] = []

    # Numbered lists
    numbered = NUMBERED_LIST_PATTERN.findall(text)
    if numbered:
        score += min(len(numbered) * 8, 30)
        signals.append(f"numbered_list:{len(numbered)} items")

    # Bullet lists
    bullets = BULLET_LIST_PATTERN.findall(text)
    if bullets:
        score += min(len(bullets) * 6, 24)
        signals.append(f"bullet_list:{len(bullets)} items")

    # Multi-part markers
    part_hits = sum(1 for m in MULTI_PART_MARKERS if m in lower)
    if part_hits >= 3:
        score += 18
        signals.append(f"multi_part:{part_hits}")
    elif part_hits >= 1:
        score += 8

    # Constraints
    constraint_hits = sum(1 for c in CONSTRAINT_MARKERS if c in lower)
    if constraint_hits >= 3:
        score += 16
        signals.append(f"constraints:{constraint_hits}")
    elif constraint_hits >= 1:
        score += 6

    # Sentence count
    if len(sentences) >= 8:
        score += 15
        signals.append(f"long_prompt:{len(sentences)} sentences")
    elif len(sentences) >= 4:
        score += 8

    # Total word count
    words = len(text.split())
    if words >= 200:
        score += 12
        signals.append(f"word_count:{words}")
    elif words >= 80:
        score += 6

    return min(score, 100), signals


# ────────────────────────────────────────────────────────────────────────────────
#  Task Type Detection
# ────────────────────────────────────────────────────────────────────────────────

def _detect_task_type(profile: CognitiveProfile, lower: str) -> Tuple[TaskType, float]:
    """Determine the primary task type and confidence from dimension scores."""

    scores: Dict[TaskType, float] = {t: 0.0 for t in TaskType}

    # Code / Debugging
    if profile.code_complexity >= 5:
        has_error = bool(ERROR_TRACE_PATTERN.search(lower))
        debug_words = any(w in lower for w in ["fix", "bug", "error", "debug", "not working", "issue", "crash"])
        code_words = any(w in lower for w in [
            "implement", "write a function", "write a program", "red-black"
        ])
        
        if (has_error or debug_words) and not code_words:
            scores[TaskType.DEBUGGING] = max(profile.code_complexity * 1.2, 50.0)
        else:
            scores[TaskType.CODE] = max(profile.code_complexity * 1.5, 55.0)

    # Mathematical
    if profile.precision_required >= 10:
        math_words = any(w in lower for w in [
            "calculate", "solve", "equation", "formula", "integral",
            "derivative", "proof", "prove", "theorem", "np", "complexity theory",
            "formal", "lemma"
        ])
        if math_words:
            scores[TaskType.MATHEMATICAL] = profile.precision_required * 1.1

    # Creative
    if profile.creative_demand >= 30:
        scores[TaskType.CREATIVE] = profile.creative_demand * 1.0

    # Analytical
    if profile.reasoning_depth >= 20 and profile.code_complexity < 15:
        scores[TaskType.ANALYTICAL] = profile.reasoning_depth * 0.9
    elif profile.reasoning_depth >= 20 and profile.code_complexity >= 15:
        scores[TaskType.ANALYTICAL] = profile.reasoning_depth * 0.5  # reduced when code pres

    # Multi-step
    if profile.structural_complexity >= 40:
        scores[TaskType.MULTI_STEP] = profile.structural_complexity * 0.85

    # Factual
    if profile.domain_specificity >= 25 and profile.reasoning_depth < 40:
        scores[TaskType.FACTUAL] = profile.domain_specificity * 0.7

    # Conversational (default — low scores everywhere)
    dim_avg = (
        profile.reasoning_depth + profile.domain_specificity +
        profile.code_complexity + profile.creative_demand +
        profile.precision_required + profile.structural_complexity
    ) / 6
    if dim_avg < 20:
        scores[TaskType.CONVERSATIONAL] = max(30, 60 - dim_avg)

    # Pick the highest-scoring type
    best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_type]

    # Confidence: how dominant is the top type vs the second?
    sorted_scores = sorted(scores.values(), reverse=True)
    if sorted_scores[0] > 0 and len(sorted_scores) > 1:
        gap_ratio = 1 - (sorted_scores[1] / sorted_scores[0]) if sorted_scores[0] != 0 else 1.0
        confidence = min(0.5 + gap_ratio * 0.5, 1.0)
    else:
        confidence = 0.5 if best_score > 0 else 0.3

    return best_type, round(confidence, 2)


# ────────────────────────────────────────────────────────────────────────────────
#  Main Analysis Entry Point
# ────────────────────────────────────────────────────────────────────────────────

def analyze_prompt(prompt: str) -> CognitiveProfile:
    """
    Perform full cognitive analysis on a prompt.
    Returns a CognitiveProfile with 6 dimension scores, detected task type,
    confidence, and a list of signals that contributed to the scoring.
    """
    text = prompt.strip()
    lower = text.lower()
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    # Score each dimension
    reasoning, sig_r = _score_reasoning_depth(text, lower, sentences)
    domain, sig_d    = _score_domain_specificity(lower)
    code, sig_c      = _score_code_complexity(text, lower)
    creative, sig_cr = _score_creative_demand(lower)
    precision, sig_p = _score_precision_required(text, lower)
    structure, sig_s = _score_structural_complexity(text, lower, sentences)

    all_signals = sig_r + sig_d + sig_c + sig_cr + sig_p + sig_s

    profile = CognitiveProfile(
        reasoning_depth=reasoning,
        domain_specificity=domain,
        code_complexity=code,
        creative_demand=creative,
        precision_required=precision,
        structural_complexity=structure,
        signals=all_signals,
    )

    # Detect task type
    task_type, confidence = _detect_task_type(profile, lower)
    profile.task_type = task_type
    profile.confidence = confidence

    return profile


# ────────────────────────────────────────────────────────────────────────────────
#  Budget Score Aggregation
# ────────────────────────────────────────────────────────────────────────────────

def profile_to_budget_score(profile: CognitiveProfile) -> int:
    """
    Convert a CognitiveProfile into a single 0-100 budget score
    using weighted dimension aggregation + task-type boost.
    """
    weighted = (
        profile.reasoning_depth       * DIMENSION_WEIGHTS["reasoning_depth"]
        + profile.code_complexity     * DIMENSION_WEIGHTS["code_complexity"]
        + profile.domain_specificity  * DIMENSION_WEIGHTS["domain_specificity"]
        + profile.structural_complexity * DIMENSION_WEIGHTS["structural_complexity"]
        + profile.creative_demand     * DIMENSION_WEIGHTS["creative_demand"]
        + profile.precision_required  * DIMENSION_WEIGHTS["precision_required"]
    )

    # Task-type boost (soft override)
    boost = TASK_TYPE_META[profile.task_type]["boost"]
    final = weighted + boost

    return min(max(int(round(final)), 1), 100)


# ────────────────────────────────────────────────────────────────────────────────
#  Tier Assignment
# ────────────────────────────────────────────────────────────────────────────────

def score_to_tier(score: int) -> str:
    """Map a budget score to a model tier."""
    if score <= 20:
        return "Tier 0"
    elif score <= 45:
        return "Tier 1"
    elif score <= 70:
        return "Tier 2"
    elif score <= 88:
        return "Tier 3"
    else:
        return "Tier 4"


# ────────────────────────────────────────────────────────────────────────────────
#  Routing Reason Generator
# ────────────────────────────────────────────────────────────────────────────────

def generate_routing_reason(profile: CognitiveProfile, tier: str, model: str) -> str:
    """
    Generate a human-readable explanation of why a prompt was routed
    to a particular tier and model.
    """
    meta = TASK_TYPE_META[profile.task_type]
    task_label = f"{meta['icon']} {meta['label']}"

    # Find the dominant dimension
    dims = {
        "Reasoning Depth":       profile.reasoning_depth,
        "Domain Specificity":    profile.domain_specificity,
        "Code Complexity":       profile.code_complexity,
        "Creative Demand":       profile.creative_demand,
        "Precision Required":    profile.precision_required,
        "Structural Complexity": profile.structural_complexity,
    }
    dominant = max(dims, key=dims.get)  # type: ignore[arg-type]
    dominant_score = dims[dominant]

    # Build explanation
    parts = [f"Detected task type: {task_label}"]

    if dominant_score >= 30:
        parts.append(f"Primary signal: {dominant} ({dominant_score}/100)")

    # Secondary dimensions
    secondary = [
        f"{k} ({v})"
        for k, v in sorted(dims.items(), key=lambda x: x[1], reverse=True)
        if k != dominant and v >= 20
    ]
    if secondary:
        parts.append(f"Contributing factors: {', '.join(secondary[:3])}")

    if profile.confidence >= 0.8:
        parts.append(f"Classification confidence: High ({int(profile.confidence * 100)}%)")
    elif profile.confidence >= 0.6:
        parts.append(f"Classification confidence: Medium ({int(profile.confidence * 100)}%)")
    else:
        parts.append(f"Classification confidence: Low ({int(profile.confidence * 100)}%)")

    boost = meta["boost"]
    if boost != 0:
        direction = "↑" if boost > 0 else "↓"
        parts.append(f"Task-type adjustment: {direction}{abs(boost)} pts")

    parts.append(f"Routed to {tier} → {model}")

    return " · ".join(parts)
