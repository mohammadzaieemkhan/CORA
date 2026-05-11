import json
import pandas as pd
from cognitive_module import create_scorer
from complexity_score import cora_complexity_score

scorer = create_scorer("rule")

with open("routerbench_samples.json") as f:
    samples = json.load(f)

scored = []
for item in samples:
    prompt = item.get("prompt") or item.get("instruction") or item.get("question", "")
    if not prompt: continue
    profile = scorer.score(prompt)
    raw_score = cora_complexity_score(profile, prompt)
    word_count = len(prompt.split())
    boosted = raw_score
    if raw_score > 0 and word_count >= 20:
        boosted += 0.04
    if boosted == 0.0 and word_count >= 6:
        boosted = 0.18
    scored.append(boosted)

df = pd.DataFrame({"score": scored})
# To get 50% AIQ (50% in Tier 4), the Tier 4 threshold should be the 50th percentile
# To get 75% AIQ (25% in Tier 4), the Tier 4 threshold should be the 75th percentile

print("Score Percentiles:")
print("25th percentile (Tier 1 limit):", df["score"].quantile(0.25))
print("50th percentile (Tier 2 limit):", df["score"].quantile(0.50))
print("75th percentile (Tier 3 limit):", df["score"].quantile(0.75))
print("85th percentile (Tier 4 limit):", df["score"].quantile(0.85))
print("95th percentile:", df["score"].quantile(0.95))
