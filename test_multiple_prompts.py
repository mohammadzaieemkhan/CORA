import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent))
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from cognitive_module import create_scorer
from complexity_score import score_to_tier, cora_complexity_score

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
    tier_label, score, budget_score = score_to_tier(profile, p)
    print(f"Prompt: {p}")
    print(f"  Scorer used: {getattr(profile, 'scorer_used', 'none')}")
    print(f"  Task type: {profile.task_type}")
    print(f"  Profile: reasoning={profile.reasoning_depth}, domain={profile.domain_specificity}, creative={profile.creative_demand}, precision={profile.precision_required}, structural={profile.structural_complexity}")
    print(f"  Complexity Score: {score:.4f}")
    print(f"  Tier Label: {tier_label}")
    print(f"  Budget Score: {budget_score}")
    print("-" * 50)
