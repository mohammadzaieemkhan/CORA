import sys
import math
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent))
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from cognitive_module import create_scorer
from cognitive_module import CognitiveProfile, TaskType

W_I = {
    "reasoning":   0.30,
    "domain":      0.20,
    "creativity":  0.20,
    "constraints": 0.15,
    "contextual":  0.10,
    "fewshots":    0.05,
}

ALPHA_I = {
    "reasoning":   2.21,
    "domain":      1.40,
    "creativity":  0.06,
    "constraints": 0.07,
    "contextual":  4.51,
    "fewshots":    0.80,
}

TAU = {
    TaskType.CODE:           1.30,
    TaskType.DEBUGGING:      1.30,
    TaskType.MATHEMATICAL:   1.40,
    TaskType.ANALYTICAL:     1.15,
    TaskType.MULTI_STEP:     1.05,
    TaskType.FACTUAL:        1.00,
    TaskType.CREATIVE:       0.90,
    TaskType.CONVERSATIONAL: 0.95,
}

Z = 1.045

def detect_few_shots(text: str) -> float:
    import re
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

def stretch(val, cap):
    return min(val / cap, 1.0)

def compute_score_for_cap(profile, prompt, cap):
    x_i = {
        "reasoning": stretch(profile.reasoning_depth, cap),
        "domain": stretch(profile.domain_specificity, cap),
        "creativity": stretch(profile.creative_demand, cap),
        "constraints": stretch(profile.structural_complexity, cap),
        "contextual": stretch(profile.precision_required, cap),
        "fewshots": detect_few_shots(prompt),
    }
    
    breakdown = {}
    for key in x_i:
        breakdown[key] = (W_I[key] * ALPHA_I[key] / Z) * x_i[key]
        
    sum_components = sum(breakdown.values())
    tau_val = TAU.get(profile.task_type, 0.95)
    score = tau_val * sum_components
    
    word_count = len(prompt.split())
    
    # Floor and boost logic for cap=100 and cap=20:
    if cap == 100:
        if score > 0 and word_count >= 20:
            score += 0.008
        if score == 0.0 and word_count >= 6:
            score = 0.03
    else:
        if score > 0 and word_count >= 20:
            score += 0.04
        if score == 0.0 and word_count >= 6:
            score = 0.18
            
    return score

scorer = create_scorer()
prompts = [
    "what is normal body temperature",
    "hi",
    "write a python function to check if a number is prime",
    "explain the difference between general relativity and quantum mechanics in detail",
    "solve the equation x^2 - 5x + 6 = 0",
    "how is this question supposed to be tier 1 what is normal temperature of human body? thier is some misinterpretation in tier logic",
    "what is normal temperature of human body?"
]

for p in prompts:
    profile = scorer.score(p)
    
    # Cap = 100
    score_100 = compute_score_for_cap(profile, p, 100)
    # Thresholds [0.04, 0.10, 0.18, 0.31]
    THRESHOLDS_100 = [0.04, 0.10, 0.18, 0.31]
    tier_idx_100 = sum(1 for t in THRESHOLDS_100 if score_100 >= t)
    tier_100 = f"Tier {max(0, min(4, tier_idx_100))}"
    
    # Cap = 20
    score_20 = compute_score_for_cap(profile, p, 20)
    # Thresholds [0.35, 0.70, 1.15, 1.50]
    THRESHOLDS_20 = [0.35, 0.70, 1.15, 1.50]
    tier_idx_20 = sum(1 for t in THRESHOLDS_20 if score_20 >= t)
    tier_20 = f"Tier {max(0, min(4, tier_idx_20))}"
    
    print(f"Prompt: {p}")
    print(f"  Profile: reasoning={profile.reasoning_depth}, domain={profile.domain_specificity}, creative={profile.creative_demand}, precision={profile.precision_required}, structural={profile.structural_complexity}")
    print(f"  Option A (cap=100, thresholds=[0.04, 0.10, 0.18, 0.31]): Score={score_100:.4f} -> {tier_100}")
    print(f"  Option B (cap=20,  thresholds=[0.35, 0.70, 1.15, 1.50]): Score={score_20:.4f} -> {tier_20}")
    print("-" * 60)
