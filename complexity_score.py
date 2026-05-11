import math
import re
from typing import Tuple, Dict

from cognitive_module import CognitiveProfile, TaskType

# ── Dimension weights ─────────────────────────────────────────────────────────
# Increased reasoning weight and reduced creativity dominance so that
# analytical/factual prompts score higher.
W_I = {
    "reasoning":   0.30,   # was 0.25 — reasoning is the strongest quality signal
    "domain":      0.20,   # was 0.15 — domain expertise matters for tier selection
    "creativity":  0.20,   # was 0.35 — was disproportionately high
    "constraints": 0.15,
    "contextual":  0.10,   # was 0.05 — precision matters more than before
    "fewshots":    0.05,
}

# ── Scaling exponents ─────────────────────────────────────────────────────────
ALPHA_I = {
    "reasoning":   2.21,   # updated via ordinal regression (was 1.50)
    "domain":      1.40,   # within 15% divergence — kept at 1.40
    "creativity":  0.06,   # updated via ordinal regression (was 0.80)
    "constraints": 0.07,   # updated via ordinal regression (was 1.10)
    "contextual":  4.51,   # updated via ordinal regression (was 1.10)
    "fewshots":    0.80,
}

# ── Task-type multipliers ────────────────────────────────────────────────────
# These amplify the final score based on the detected task type.
TAU = {
    TaskType.CODE:           1.30,
    TaskType.DEBUGGING:      1.30,
    TaskType.MATHEMATICAL:   1.40,  # was 1.25 — GSM8K proves math needs higher tier
    TaskType.ANALYTICAL:     1.15,   # was 1.10
    TaskType.MULTI_STEP:     1.05,   # was 1.00
    TaskType.FACTUAL:        1.00,
    TaskType.CREATIVE:       0.90,
    TaskType.CONVERSATIONAL: 0.95,   # lowered from 1.10 — reduces over-routing of simple chats
}

Z = 1.045

def detect_few_shots(text: str) -> float:
    # Use regex to detect Q/A, Input/Output, Example:, Human/Assistant, and numbered example patterns.
    # Returns 0.0/0.25/0.50/0.75/1.00 for 0/1/2/3/4+ examples found.
    qa = len(re.findall(r"(?i)\bq:.*?\ba:", text, flags=re.DOTALL))
    io = len(re.findall(r"(?i)input:.*?output:", text, flags=re.DOTALL))
    ex = len(re.findall(r"(?i)example\s*(?:\d+)?\s*:", text))
    ha = len(re.findall(r"(?i)human:.*?assistant:", text, flags=re.DOTALL))
    
    count = qa + io + ex + ha
    if count >= 4: return 1.00
    if count == 3: return 0.75
    if count == 2: return 0.50
    if count == 1: return 0.25
    return 0.00

def stretch(val, cap=20):
    """Normalise a raw cognitive dimension score to [0, 1].
    Cap lowered from 25 -> 20 so even moderate signals push scores higher,
    creating better separation between Tier 1 and Tier 2.
    """
    return min(val / cap, 1.0)

def profile_to_nemo_scores(profile: CognitiveProfile, prompt: str) -> Dict[str, float]:
    return {
        "reasoning": stretch(profile.reasoning_depth),
        "domain": stretch(profile.domain_specificity),
        "creativity": stretch(profile.creative_demand),
        "constraints": stretch(profile.structural_complexity),
        "contextual": stretch(profile.precision_required),
        "fewshots": detect_few_shots(prompt),
    }

def get_score_breakdown(profile: CognitiveProfile, prompt: str) -> Dict[str, float]:
    x_i = profile_to_nemo_scores(profile, prompt)
    
    breakdown = {}
    for key in x_i:
        breakdown[key] = (W_I[key] * ALPHA_I[key] / Z) * x_i[key]
        
    return breakdown

def cora_complexity_score(profile: CognitiveProfile, prompt: str) -> float:
    breakdown = get_score_breakdown(profile, prompt)
    sum_components = sum(breakdown.values())
    
    # task type enum lookup, default to CONVERSATIONAL if not found
    tau_val = TAU.get(profile.task_type, 0.95)
    
    return tau_val * sum_components

def calibrate_confidence(raw_confidence: float, tier: str) -> float:
    """
    Apply temperature scaling per tier to reduce RS calibration error.
    Temperatures derived from GSM8K calibration run.
    """
    # Ensure raw_confidence is in (0, 1) range for logit calculation
    eps = 1e-9
    p = max(eps, min(1.0 - eps, raw_confidence))
    
    # Higher temperature = lower confidence = better calibration for overconfident models
    TEMPERATURE = {
        "Tier 0": 2.8,   # most overconfident on math
        "Tier 1": 2.2,
        "Tier 2": 1.6,
        "Tier 3": 1.2,
        "Tier 4": 1.0,   # frontier models are best calibrated
    }
    T = TEMPERATURE.get(tier, 1.5)
    
    # Softmax temperature scaling: scale logit then re-sigmoid
    logit = math.log(p / (1 - p))
    scaled_logit = logit / T
    return 1 / (1 + math.exp(-scaled_logit))

def score_to_tier(profile: CognitiveProfile, prompt: str) -> Tuple[str, float, int]:
    score = cora_complexity_score(profile, prompt)
    word_count = len(prompt.split())

    # ── Prompt-length complexity boost ─────────────────────────────────
    if score > 0 and word_count >= 20:
        score += 0.04

    # ── Minimal floor for zero-score prompts ────────────────────────
    if score == 0.0 and word_count >= 6:
        score = 0.18

    # ── Tier thresholds ───────────────────────────────────────────────
    THRESHOLDS = [0.35, 0.70, 1.15, 1.50]
    tier_idx = sum(1 for t in THRESHOLDS if score >= t)
    tier_idx = max(0, min(4, tier_idx))
    tier_label = f"Tier {tier_idx}"

    # ── Confidence Calibration ────────────────────────────────────────
    # Normalise 0-2 score to a 0-1 confidence-like difficulty score
    # then apply temperature scaling to fix overconfidence.
    norm_diff = min(score / 2.0, 1.0)
    calibrated_diff = calibrate_confidence(norm_diff, tier_label)
    
    budget_score = int(min(max(round(calibrated_diff * 100), 1), 100))

    return tier_label, score, budget_score
