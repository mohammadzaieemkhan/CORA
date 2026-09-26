import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent))
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from cognitive_module import create_scorer
from complexity_score import score_to_tier, cora_complexity_score

scorer = create_scorer()
prompt = "what is normal body temperature"
profile = scorer.score(prompt)
print("Scorer used:", getattr(profile, "scorer_used", "none"))
print("Profile dict:", profile.to_dict())
tier_label, score, budget_score = score_to_tier(profile, prompt)
print("Complexity Score:", score)
print("Tier Label:", tier_label)
print("Budget Score:", budget_score)
