"""
cognitive_module.feature_extractor
──────────────────────────────────
Extracts a structured numerical feature vector from raw prompt text.

These features serve *dual purpose*:
  1. Rule-based scorer consumes them directly for heuristic scoring
  2. DistilBERT hybrid model can concatenate them with transformer embeddings

Total: 42 named features → Dict[str, float]
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


# ════════════════════════════════════════════════════════════════════════════════
#  VOCABULARY / PATTERN BANKS
# ════════════════════════════════════════════════════════════════════════════════

# ── Reasoning markers ────────────────────────────────────────────────────────
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
    # design / architecture / engineering
    "design": 12, "implement": 12, "architecture": 14, "architect": 12,
    "optimize": 12, "optimise": 12, "optimization": 12, "optimisation": 12,
    "scalable": 10, "scalability": 10, "distributed": 12,
    "concurrent": 10, "concurrency": 10, "parallel": 10,
    "fault-tolerant": 14, "fault tolerant": 14, "consensus": 12,
    "protocol": 10, "mechanism": 8,
    # academic / formal reasoning
    "from first principles": 16, "first principles": 14,
    "formally": 10, "formalize": 12, "formalise": 12,
    "derive": 12, "derivation": 12, "mathematically": 14,
    "prove that": 14, "proof that": 14, "demonstrate that": 12,
    "rigorously": 12, "rigorous": 10,
    # comparison / trade-off analysis  
    "versus": 8, "vs": 6, "advantages": 8, "disadvantages": 8,
    "benefits": 6, "drawbacks": 8, "limitations": 8,
    "strengths and weaknesses": 12, "compare and contrast": 14,
    # complexity / depth signals
    "in depth": 10, "in-depth": 10, "comprehensive": 10,
    "thorough": 8, "detailed": 8, "elaborate": 8,
    "explain why": 10, "explain how": 10,
    "what are the": 6, "describe the": 6,
    # common question/instruction verbs
    "explain": 8, "describe": 6, "outline": 8, "summarize": 6, "summarise": 6,
    "provide": 4, "list": 4, "define": 4, "illustrate": 8,
    "strategy": 8, "schema": 8, "structure": 6, "overview": 6,
    "difference between": 10, "relationship between": 10,
}

# ── Domain vocabularies ─────────────────────────────────────────────────────
DOMAIN_VOCAB: Dict[str, List[str]] = {
    "computer_science": [
        "algorithm", "data structure", "binary tree", "hash map", "linked list",
        "recursion", "dynamic programming", "big o", "time complexity",
        "space complexity", "polymorphism", "inheritance", "encapsulation",
        "microservice", "kubernetes", "docker", "rest api", "graphql",
        "database", "sql", "nosql", "tcp", "udp", "http", "websocket",
        "machine learning", "neural network", "transformer", "attention mechanism",
        "gradient descent", "backpropagation", "convolution", "embedding",
        "compiler", "parser", "lexer", "ast", "tokenizer",
        "distributed system", "consensus", "byzantine", "paxos", "raft",
        "cache", "load balancer", "sharding", "replication",
        "thread", "mutex", "semaphore", "deadlock", "race condition",
        "latency", "throughput", "bottleneck", "benchmark",
        "api", "endpoint", "middleware", "authentication", "authorization",
        "encryption", "hashing", "ssl", "tls", "oauth",
        "frontend", "backend", "full-stack", "devops", "ci/cd",
        "deep learning", "reinforcement learning", "fine-tuning", "inference",
        "gpu", "batch size", "epoch", "loss function", "optimizer",
        # Data structures & algorithms (from cognitive.py)
        "red-black tree", "avl tree", "b-tree", "trie", "heap",
        "insertion", "deletion", "balancing", "traversal",
        "np", "np-hard", "np-complete", "complexity theory",
        "implement", "red-black", "memoization", "backtracking",
        "greedy", "divide and conquer", "sorting", "searching",
        # Web development / frameworks
        "react", "angular", "vue", "svelte", "next.js", "nuxt",
        "useeffect", "usestate", "usememo", "usecallback", "useref",
        "hook", "hooks", "component", "render", "re-render",
        "dependency array", "context provider", "context api",
        "state management", "redux", "zustand", "mobx",
        "virtual dom", "dom", "jsx", "tsx", "props", "state",
        "strict mode", "production build", "dev mode",
        "webpack", "vite", "babel", "typescript", "javascript",
        "node.js", "express", "fastapi", "django", "flask",
        "async", "await", "promise", "callback", "event loop",
        # Debugging / runtime concepts
        "infinite loop", "memory leak", "stack overflow", "segfault",
        "null pointer", "undefined", "nan", "type error",
        "runtime error", "compile error", "syntax error",
        "breakpoint", "stack trace", "profiler", "debugger",
        "batch", "batched", "concurrent", "concurrency",
    ],
    "medicine": [
        "diagnosis", "symptom", "pathology", "pharmacology", "dosage",
        "clinical trial", "prognosis", "etiology", "epidemiology",
        "cardiac", "neurological", "oncology", "metabolic",
        "patient", "treatment", "therapy", "prescription", "contraindication",
    ],
    "law": [
        "statute", "jurisdiction", "tort", "liability", "precedent",
        "plaintiff", "defendant", "contract law", "intellectual property",
        "compliance", "regulation", "amendment",
        "hipaa", "gdpr", "legal", "lawsuit",
    ],
    "science": [
        "hypothesis", "experiment", "variable", "quantum", "thermodynamics",
        "molecular", "photosynthesis", "genome", "evolution", "entropy",
        "electromagnetic", "relativity", "particle physics",
        "physics", "chemistry", "biology", "neuroscience",
    ],
    "finance": [
        "portfolio", "derivative", "hedge", "equity", "bond", "yield",
        "amortization", "compound interest", "black-scholes", "volatility",
        "market cap", "dividend", "ipo", "valuation",
        "revenue", "profit margin", "roi", "cash flow",
    ],
    "mathematics": [
        "matrix", "vector", "eigenvalue", "determinant", "linear algebra",
        "calculus", "integral", "differential", "topology",
        "probability", "statistics", "bayesian", "stochastic",
        "graph theory", "combinatorics", "number theory",
        "polynomial", "fourier", "laplace", "markov",
        # Common math task terms
        "equation", "solve", "quadratic", "coefficient", "variable",
        "factor", "factoring", "root", "solution", "inequality",
        "prime", "prime number", "divisible", "remainder", "modulo",
        "trigonometry", "logarithm", "exponent", "function",
    ],
}

# ── Code patterns ────────────────────────────────────────────────────────────
CODE_KEYWORDS = {
    "function", "def ", "class ", "import ", "const ", "let ", "var ",
    "return ", "if ", "else ", "for ", "while ", "try ", "catch ",
    "except ", "raise ", "throw ", "async ", "await ", "yield ",
    "public ", "private ", "static ", "void ", "int ", "string ",
    "print(", "console.log", "fmt.println", "system.out",
    # Implementation / algorithm keywords (from cognitive.py)
    "implement", "write a function", "write a program", "function that",
    "recursive", "recursion", "dynamic programming", "dp", "algorithm",
    "debug", "fix this", "binary search", "sorting", "linked list",
    "stack", "queue", "tree", "graph", "hash",
    "red-black", "insertion", "deletion", "balancing",
    "in c++", "in rust", "in java", "in python", "in javascript",
    # Web development / framework-specific debugging
    "useeffect", "usestate", "usememo", "usecallback", "useref",
    "re-render", "render loop", "dependency array", "context provider",
    "component", "hook", "hooks", "middleware", "endpoint",
    "production build", "dev mode", "strict mode",
    "webpack", "vite", "babel", "npm", "yarn",
    "infinite loop", "memory leak", "stack overflow",
    "react", "angular", "vue", "svelte",
    # Common code-related verbs
    "refactor", "optimize", "deploy", "compile", "build",
    "write code", "code that", "program that",
}

CODE_BLOCK_RE    = re.compile(r"```[\s\S]*?```", re.MULTILINE)
INLINE_CODE_RE   = re.compile(r"`[^`]+`")
ERROR_TRACE_RE   = re.compile(
    r"(Traceback|Error:|Exception:|FATAL|panic:|Segmentation fault"
    r"|TypeError|ValueError|KeyError|IndexError|SyntaxError"
    r"|NullPointerException|undefined is not|Cannot read propert)",
    re.IGNORECASE,
)
FILE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\|/(?:usr|etc|home|var|opt)/)[\w/\\.\-]+"
)

# ── Creative markers ────────────────────────────────────────────────────────
CREATIVE_MARKERS = {
    "write a story": 15, "write a poem": 15, "creative": 10,
    "imagine": 10, "brainstorm": 12, "come up with": 10,
    "generate ideas": 12, "fiction": 12, "narrative": 10,
    "invent": 10, "original": 6,
    "compose": 10, "draft": 8, "rewrite": 6,
    "write a song": 14, "write a script": 12, "creative writing": 14,
    "worldbuilding": 12, "character development": 10,
}

OPEN_ENDED_MARKERS = ["what if", "could you", "how might", "suggest", "propose"]

# ── Precision markers ───────────────────────────────────────────────────────
PRECISION_MARKERS = {
    "exact": 8, "precisely": 10, "accurate": 8, "citation": 12,
    "source": 6, "reference": 6, "peer-reviewed": 14,
    "calculate": 14, "compute": 10, "derive": 12,
    "proof": 14, "prove": 12, "verify": 10,
    "significant figures": 10, "decimal places": 8,
    # math & formal precision
    "mathematically": 12, "formal proof": 14, "derivation": 10,
    "complexity analysis": 14, "time complexity": 12, "space complexity": 12,
    "big o": 12, "asymptotic": 12, "bound": 8,
    "theorem": 12, "lemma": 10, "corollary": 10,
    "equation": 8, "formula": 8, "expression": 6,
    "parameter count": 10, "exact number": 10,
    "benchmark": 8, "measure": 6, "quantify": 10,
    "how many": 10, "how much": 10, "how far": 10, "how long": 10,
    "total": 8, "per hour": 10, "miles": 8, "kilometers": 8,
    "percent": 12, "percentage": 12, "what is the": 6, "how old": 10,
    "if x": 10, "solve for": 14, "word problem": 12,
}

MATH_RE = re.compile(
    r"(?:\d+\s*[\+\-\*\/\=\^]\s*\d+|∫|∑|∏|√|≈|≠|≤|≥|lim|log|sin|cos|tan"
    r"|matrix|determinant|eigenvalue|∂|∇|dx|dy|O\()",
    re.IGNORECASE,
)

# ── Structural markers ──────────────────────────────────────────────────────
NUMBERED_LIST_RE = re.compile(r"^\s*\d+[\.)\]]\s", re.MULTILINE)
BULLET_LIST_RE   = re.compile(r"^\s*[-\*•]\s", re.MULTILINE)

CONSTRAINT_MARKERS = [
    "must", "should", "ensure", "constraint", "requirement",
    "at least", "at most", "no more than", "between",
    "given that", "assuming", "if and only if",
    "with the following", "following constraints",
    "needs to", "has to", "required to", "necessary",
    "make sure", "comply", "compliance",
    "without", "only", "except", "limited to",
]

MULTI_PART_MARKERS = [
    "first", "second", "third", "fourth", "fifth",
    "finally", "additionally", "lastly",
    "moreover", "furthermore", "also", "next", "then",
    "part 1", "part 2", "part 3", "part a", "part b",
    "step 1", "step 2", "step 3",
    "1.", "2.", "3.", "4.", "5.",
    "phase 1", "phase 2",
]

CONDITIONAL_RE = re.compile(r"\b(?:if|when|unless|whether|suppose|assuming|whereas|provided that)\b", re.IGNORECASE)

# ── Debugging markers ───────────────────────────────────────────────────────
DEBUG_MARKERS = ["fix", "bug", "error", "debug", "not working", "issue", "crash"]

# ── Math task markers ───────────────────────────────────────────────────────
MATH_TASK_MARKERS = ["calculate", "solve", "equation", "formula", "integral", "derivative", "proof"]

# ── Filler/Padding Detection ────────────────────────────────────────────────
# Verbs that indicate an actionable instruction (imperative mood)
IMPERATIVE_VERBS = {
    "explain", "describe", "write", "create", "build", "make", "give",
    "tell", "show", "help", "find", "solve", "calculate", "implement",
    "design", "analyze", "analyse", "compare", "list", "define", "code",
    "fix", "debug", "optimize", "optimise", "generate", "summarize",
    "summarise", "translate", "convert", "compute", "prove", "derive",
    "refactor", "deploy", "develop", "outline", "evaluate", "assess",
    "suggest", "recommend", "provide", "draw", "plot", "diagram",
    "modify", "update", "change", "add", "remove", "delete", "insert",
}

# Phrase prefixes that signal an actionable request
ACTIONABLE_PREFIXES = [
    "give me", "can you", "could you", "would you", "please",
    "i need", "i want", "i'd like", "how do i", "how can i",
    "how to", "what is", "what are", "what's", "who is", "who are",
    "where is", "where are", "when is", "when did",
    "is it", "is there", "are there", "do you", "does it",
]

# Context-reference markers: if the actionable sentence references "the above",
# "the following", etc., the surrounding text is likely relevant context, not filler.
CONTEXT_REFERENCE_MARKERS = [
    "the above", "the following", "the below", "this text",
    "this passage", "this paragraph", "this code", "this error",
    "this snippet", "the given", "given the", "based on the",
    "from the", "in the above", "mentioned above", "shown above",
    "provided above", "attached", "here is", "here are",
    "the preceding", "the previous",
]


def _is_actionable_sentence(sentence: str) -> bool:
    """
    Check if a sentence contains a question or direct instruction.
    Returns True for questions and imperative statements.
    """
    s = sentence.strip()
    if not s:
        return False

    lower = s.lower()
    words = lower.split()
    if not words:
        return False

    # Questions (ends with ?)
    if s.rstrip().endswith("?"):
        return True

    # Starts with an imperative verb
    first_word = words[0].rstrip(".,;:!?")
    if first_word in IMPERATIVE_VERBS:
        return True

    # Starts with an actionable prefix
    if any(lower.startswith(p) for p in ACTIONABLE_PREFIXES):
        return True

    return False


def _has_context_references(actionable_text: str) -> bool:
    """
    Check if actionable sentences reference surrounding text
    (e.g., "summarize the above", "given the following code").
    If so, the surrounding text is essential context, not filler.
    """
    lower = actionable_text.lower()
    return any(marker in lower for marker in CONTEXT_REFERENCE_MARKERS)


def detect_filler(prompt: str) -> dict:
    """
    Analyse a prompt for filler/padding content.
    
    Returns a dict with:
      - filler_ratio: float (0.0–1.0), proportion of non-actionable sentences
      - actionable_sentence_count: int
      - total_sentence_count: int
      - is_filler_padded: bool
      - actionable_text: str (just the actionable sentences joined)
    """
    # Split into sentences (on period, exclamation, question mark)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', prompt.strip()) if s.strip()]
    
    # Also handle sentences that might not end with punctuation
    if not sentences:
        sentences = [prompt.strip()]
    
    total = len(sentences)
    if total <= 1:
        # Single sentence — no filler detection needed
        return {
            "filler_ratio": 0.0,
            "actionable_sentence_count": 1,
            "total_sentence_count": 1,
            "is_filler_padded": False,
            "actionable_text": prompt.strip(),
        }
    
    actionable = []
    passive = []
    
    for sent in sentences:
        if _is_actionable_sentence(sent):
            actionable.append(sent)
        else:
            passive.append(sent)
    
    actionable_count = len(actionable)
    filler_ratio = len(passive) / total if total > 0 else 0.0
    
    # Build the actionable-only text
    actionable_text = " ".join(actionable) if actionable else prompt.strip()
    
    # Determine if this is genuinely filler-padded:
    # - High filler ratio (>= 0.70 — most sentences are passive)
    # - At least 3 total sentences (short prompts aren't "padded")
    # - The actionable sentences don't reference the surrounding text as context
    is_padded = (
        filler_ratio >= 0.70
        and total >= 3
        and actionable_count >= 1
        and not _has_context_references(actionable_text)
    )
    
    return {
        "filler_ratio": round(filler_ratio, 3),
        "actionable_sentence_count": actionable_count,
        "total_sentence_count": total,
        "is_filler_padded": is_padded,
        "actionable_text": actionable_text,
    }

# ════════════════════════════════════════════════════════════════════════════════
#  FEATURE NAMES (in canonical order — 45 features)
# ════════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES: List[str] = [
    # ── Text statistics (7) ──
    "word_count",
    "char_count",
    "sentence_count",
    "avg_words_per_sentence",
    "question_mark_count",
    "unique_word_ratio",
    "avg_word_length",

    # ── Reasoning (5) ──
    "reasoning_keyword_score",
    "reasoning_keyword_count",
    "conditional_count",
    "causal_connector_count",
    "question_depth_score",

    # ── Domain (5) ──
    "domain_hit_count",
    "domain_count",
    "primary_domain_hits",
    "cross_domain_score",
    "domain_specificity_score",

    # ── Code (7) ──
    "code_block_count",
    "code_block_lines",
    "inline_code_count",
    "code_keyword_count",
    "has_error_trace",
    "has_file_paths",
    "code_complexity_score",

    # ── Creative (4) ──
    "creative_keyword_score",
    "creative_keyword_count",
    "open_ended_count",
    "creative_demand_score",

    # ── Precision (5) ──
    "precision_keyword_score",
    "precision_keyword_count",
    "math_expression_count",
    "numeric_density",
    "precision_required_score",

    # ── Structure (7) ──
    "numbered_list_count",
    "bullet_list_count",
    "multi_part_marker_count",
    "constraint_marker_count",
    "structural_complexity_score",
    "has_debug_markers",
    "has_math_task_markers",

    # ── Aggregate (2) ──
    "total_signal_count",
    "estimated_output_complexity",

    # ── Filler Detection (3) ──
    "filler_ratio",
    "actionable_sentence_count",
    "is_filler_padded",
]


# ════════════════════════════════════════════════════════════════════════════════
#  VOCABULARY NORMALIZATION (deflate ornamental synonyms before scoring)
# ════════════════════════════════════════════════════════════════════════════════

# Maps pretentious/academic synonyms to their plain equivalents.
# Applied to lowercased text BEFORE keyword matching so that fancy vocabulary
# for simple concepts doesn't inflate complexity scores.
VOCABULARY_DEFLATION: Dict[str, str] = {
    # Verbs — fancy → plain
    "elucidate": "explain",
    "explicate": "explain",
    "expound": "explain",
    "delineate": "describe",
    "articulate": "explain",
    "illuminate": "explain",
    "clarify": "explain",
    "enumerate": "list",
    "ascertain": "find out",
    "discern": "identify",
    "deliberate": "think about",
    "pontificate": "talk about",
    "ruminate": "think about",
    "contemplate": "think about",
    "cogitate": "think about",
    "endeavor": "try",
    "endeavour": "try",
    "utilize": "use",
    "utilise": "use",
    "facilitate": "help",
    "ameliorate": "improve",
    "commence": "start",
    "terminate": "end",
    "procure": "get",
    "disseminate": "share",
    "promulgate": "announce",
    "inaugurate": "start",
    "effectuate": "do",
    "peruse": "read",
    # Nouns — fancy → plain
    "paradigm": "concept",
    "methodology": "method",
    "framework": "approach",
    "modality": "way",
    "nomenclature": "naming",
    "vernacular": "language",
    "lexicon": "vocabulary",
    "discourse": "discussion",
    "juxtaposition": "comparison",
    "dichotomy": "contrast",
    "conundrum": "problem",
    "quandary": "problem",
    "predicament": "problem",
    "ramification": "consequence",
    "manifestation": "form",
    "instantiation": "example",
    "proclivity": "tendency",
    "propensity": "tendency",
    "predilection": "preference",
    # Adjectives — fancy → plain
    "quintessential": "essential",
    "fundamental": "basic",
    "paramount": "important",
    "indispensable": "essential",
    "ubiquitous": "common",
    "multifaceted": "complex",
    "comprehensive": "complete",
    "intrinsic": "basic",
    "extraneous": "extra",
    "superfluous": "extra",
    "succinct": "short",
    "concise": "short",
    "verbose": "wordy",
    "efficacious": "effective",
    "perspicacious": "sharp",
    "sagacious": "wise",
    "magnanimous": "generous",
    "ostentatious": "showy",
    "surreptitious": "secret",
    "perfunctory": "careless",
    "meticulous": "careful",
    # Adverbs / phrases
    "in a succinct manner": "briefly",
    "in a comprehensive manner": "fully",
    "in a meticulous manner": "carefully",
    "in an expeditious manner": "quickly",
    "in a manner": "",
    "with respect to": "about",
    "pertaining to": "about",
    "in regard to": "about",
    "in regards to": "about",
    "with regard to": "about",
    "vis-à-vis": "about",
    "vis-a-vis": "about",
    "apropos of": "about",
    "in the context of": "about",
    "in lieu of": "instead of",
    "notwithstanding": "despite",
    "henceforth": "from now on",
    "heretofore": "before",
    "aforementioned": "previous",
    "subsequently": "then",
    "consequently": "so",
    "furthermore": "also",
    "moreover": "also",
    "nevertheless": "but",
    "nonetheless": "but",
}

# Sort by descending length so longer phrases are matched first
_DEFLATION_PAIRS = sorted(VOCABULARY_DEFLATION.items(), key=lambda x: len(x[0]), reverse=True)


def _normalize_vocabulary(lower_text: str) -> str:
    """
    Replace ornamental/pretentious words with plain equivalents.
    Operates on already-lowercased text. Returns normalized text.
    """
    result = lower_text
    for fancy, plain in _DEFLATION_PAIRS:
        if fancy in result:
            result = result.replace(fancy, plain)
    return result


# ════════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """
    Extracts a structured feature dict from a raw prompt string.

    Usage:
        extractor = FeatureExtractor()
        features = extractor.extract("Explain quantum entanglement step by step")
        # → Dict[str, float] with 42 named features
    """

    def extract(self, prompt: str) -> Dict[str, float]:
        """Extract all 42 features from a prompt. Returns named feature dict."""
        text = prompt.strip()
        lower = text.lower()

        # Normalize vocabulary: deflate ornamental synonyms so fancy-but-simple
        # prompts don't trigger false complexity signals
        lower = _normalize_vocabulary(lower)

        words = text.split()
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        features: Dict[str, float] = {}

        # ── Text statistics ──────────────────────────────────────────────
        features["word_count"] = len(words)
        features["char_count"] = len(text)
        features["sentence_count"] = len(sentences)
        features["avg_words_per_sentence"] = (
            sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        )
        features["question_mark_count"] = text.count("?")
        unique_words = set(w.lower() for w in words)
        features["unique_word_ratio"] = (
            len(unique_words) / max(len(words), 1)
        )
        features["avg_word_length"] = (
            sum(len(w) for w in words) / max(len(words), 1)
        )

        # ── Reasoning ────────────────────────────────────────────────────
        r_score = 0
        r_count = 0
        causal_count = 0
        for phrase, pts in REASONING_MARKERS.items():
            if phrase in lower:
                r_score += pts
                r_count += 1
                if phrase in ("because", "therefore", "consequently", "implies",
                              "hence", "thus", "as a result"):
                    causal_count += 1

        cond_count = len(CONDITIONAL_RE.findall(lower))

        q_count = features["question_mark_count"]
        q_depth = 0
        if q_count >= 4:
            q_depth = 18
        elif q_count >= 2:
            q_depth = 10
        elif q_count >= 1:
            q_depth = 4

        features["reasoning_keyword_score"] = r_score
        features["reasoning_keyword_count"] = r_count
        features["conditional_count"] = cond_count
        features["causal_connector_count"] = causal_count
        features["question_depth_score"] = q_depth

        # ── Domain ───────────────────────────────────────────────────────
        domain_hits: Dict[str, int] = {}
        total_domain_hits = 0
        for domain, vocab in DOMAIN_VOCAB.items():
            hits = sum(1 for term in vocab if term in lower)
            if hits > 0:
                domain_hits[domain] = hits
                total_domain_hits += hits

        primary_hits = max(domain_hits.values()) if domain_hits else 0
        domain_count = len(domain_hits)
        cross_domain = 20 if domain_count >= 3 else (10 if domain_count >= 2 else 0)
        domain_score = min(primary_hits * 12, 60) + cross_domain if domain_hits else 0

        features["domain_hit_count"] = total_domain_hits
        features["domain_count"] = domain_count
        features["primary_domain_hits"] = primary_hits
        features["cross_domain_score"] = cross_domain
        features["domain_specificity_score"] = min(domain_score, 100)

        # ── Code ─────────────────────────────────────────────────────────
        code_blocks = CODE_BLOCK_RE.findall(text)
        block_lines = sum(b.count("\n") for b in code_blocks)
        inline_codes = INLINE_CODE_RE.findall(text)
        kw_hits = sum(1 for kw in CODE_KEYWORDS if kw in lower)
        has_error = 1 if ERROR_TRACE_RE.search(text) else 0
        has_paths = 1 if FILE_PATH_RE.search(text) else 0

        code_score = 0
        if code_blocks:
            code_score += min(20 + block_lines * 2, 50)
        if inline_codes:
            code_score += min(len(inline_codes) * 4, 20)
        if kw_hits >= 5:
            code_score += 25
        elif kw_hits >= 2:
            code_score += 12
        elif kw_hits >= 1:
            code_score += 6
        if has_error:
            code_score += 20
        if has_paths:
            code_score += 8

        features["code_block_count"] = len(code_blocks)
        features["code_block_lines"] = block_lines
        features["inline_code_count"] = len(inline_codes)
        features["code_keyword_count"] = kw_hits
        features["has_error_trace"] = has_error
        features["has_file_paths"] = has_paths
        features["code_complexity_score"] = min(code_score, 100)

        # ── Creative ─────────────────────────────────────────────────────
        cr_score = 0
        cr_count = 0
        for phrase, pts in CREATIVE_MARKERS.items():
            if phrase in lower:
                cr_score += pts
                cr_count += 1

        open_ended = sum(1 for m in OPEN_ENDED_MARKERS if m in lower)
        if open_ended >= 2:
            cr_score += 12
        elif open_ended >= 1:
            cr_score += 6

        features["creative_keyword_score"] = cr_score
        features["creative_keyword_count"] = cr_count
        features["open_ended_count"] = open_ended
        features["creative_demand_score"] = min(cr_score, 100)

        # ── Precision ────────────────────────────────────────────────────
        p_score = 0
        p_count = 0
        for phrase, pts in PRECISION_MARKERS.items():
            if phrase in lower:
                p_score += pts
                p_count += 1

        math_hits = MATH_RE.findall(text)
        if math_hits:
            p_score += min(len(math_hits) * 8, 40)

        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        num_density = len(numbers)
        if num_density >= 6:
            p_score += 15
        elif num_density >= 3:
            p_score += 8

        features["precision_keyword_score"] = p_score
        features["precision_keyword_count"] = p_count
        features["math_expression_count"] = len(math_hits)
        features["numeric_density"] = num_density
        features["precision_required_score"] = min(p_score, 100)

        # ── Structure ────────────────────────────────────────────────────
        numbered = NUMBERED_LIST_RE.findall(text)
        bullets = BULLET_LIST_RE.findall(text)
        part_hits = sum(1 for m in MULTI_PART_MARKERS if m in lower)
        constraint_hits = sum(1 for c in CONSTRAINT_MARKERS if c in lower)

        s_score = 0
        if numbered:
            s_score += min(len(numbered) * 8, 40)
        if bullets:
            s_score += min(len(bullets) * 6, 30)
        if part_hits >= 5:
            s_score += 28
        elif part_hits >= 3:
            s_score += 20
        elif part_hits >= 1:
            s_score += 10
        if constraint_hits >= 5:
            s_score += 24
        elif constraint_hits >= 3:
            s_score += 16
        elif constraint_hits >= 1:
            s_score += 8
        if len(sentences) >= 10:
            s_score += 22
        elif len(sentences) >= 6:
            s_score += 15
        elif len(sentences) >= 4:
            s_score += 10
        if len(words) >= 200:
            s_score += 20
        elif len(words) >= 100:
            s_score += 14
        elif len(words) >= 50:
            s_score += 8
        elif len(words) >= 30:
            s_score += 4

        features["numbered_list_count"] = len(numbered)
        features["bullet_list_count"] = len(bullets)
        features["multi_part_marker_count"] = part_hits
        features["constraint_marker_count"] = constraint_hits
        features["structural_complexity_score"] = min(s_score, 100)

        # ── Task-type helper flags ───────────────────────────────────────
        features["has_debug_markers"] = (
            1 if any(w in lower for w in DEBUG_MARKERS) else 0
        )
        features["has_math_task_markers"] = (
            1 if any(w in lower for w in MATH_TASK_MARKERS) else 0
        )

        # ── Aggregates ───────────────────────────────────────────────────
        signal_count = (
            r_count + total_domain_hits + len(code_blocks)
            + len(inline_codes) + cr_count + p_count
            + len(numbered) + len(bullets) + part_hits + constraint_hits
        )
        features["total_signal_count"] = signal_count

        # Estimated output complexity: a rough proxy for how "big" the answer
        # needs to be, based on structural + reasoning signals
        features["estimated_output_complexity"] = min(
            features["structural_complexity_score"] * 0.4
            + features["reasoning_keyword_score"] * 0.3
            + features["code_complexity_score"] * 0.3,
            100,
        )

        return features

    def extract_vector(self, prompt: str) -> List[float]:
        """
        Extract features as a flat list in canonical FEATURE_NAMES order.
        Suitable for feeding into ML models.
        """
        feat_dict = self.extract(prompt)
        return [feat_dict.get(name, 0.0) for name in FEATURE_NAMES]

    @staticmethod
    def feature_names() -> List[str]:
        """Return the canonical feature names in order."""
        return list(FEATURE_NAMES)
