import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from cognitive_module.scorer import create_scorer
from cognitive_module.nemo_scorer import NeMoScorer
from cognitive_module.rule_scorer import RuleBasedScorer

prompt = (
    "Hey, can you help me debug this? My React app's useEffect is causing an infinite re-render loop "
    "and I think it's related to the dependency array, but I also have a context provider that might "
    "be re-creating objects on every render. The state updates are batched in React 18 but the issue "
    "only happens in production builds, not in dev mode with strict mode off."
)

rule_scorer = RuleBasedScorer()
nemo_scorer = NeMoScorer()

print("--- RULE SCORER ---")
rule_profile = rule_scorer.score(prompt)
print(f"Task: {rule_profile.task_type}")
print(f"R={rule_profile.reasoning_depth} D={rule_profile.domain_specificity} C={rule_profile.code_complexity}")
print(f"Cr={rule_profile.creative_demand} P={rule_profile.precision_required} S={rule_profile.structural_complexity}")

print("\n--- NEMO SCORER RAW (if ready) ---")
if nemo_scorer._ready:
    inputs = nemo_scorer._tokenizer(
        [prompt], 
        return_tensors="pt", 
        add_special_tokens=True, 
        max_length=512, 
        padding=True, 
        truncation=True
    )
    input_ids = inputs["input_ids"].to(nemo_scorer._device)
    attention_mask = inputs["attention_mask"].to(nemo_scorer._device)
    
    import torch
    with torch.no_grad():
        res = nemo_scorer._model(input_ids, attention_mask)
    
    print(f"Raw predictions:")
    for k, v in res.items():
        if k in ["task_type_1", "task_type_2", "task_type_prob", "creativity_scope", "reasoning", "domain_knowledge", "constraint_ct", "contextual_knowledge", "number_of_few_shots"]:
            print(f"  {k}: {v[0]}")
    
    final_profile = nemo_scorer.score(prompt)
    print(f"\n--- FINAL BLENDED NEMO ---")
    print(f"Task: {final_profile.task_type}")
    print(f"R={final_profile.reasoning_depth} D={final_profile.domain_specificity} C={final_profile.code_complexity}")
    print(f"Cr={final_profile.creative_demand} P={final_profile.precision_required} S={final_profile.structural_complexity}")
else:
    print("Nemo scorer is not ready!")
