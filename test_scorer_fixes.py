"""
Comprehensive CORA scoring test — tests both the vocabulary inflation fix
AND that legitimate complex prompts score correctly.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent))
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from cognitive_module import create_scorer
from complexity_score import score_to_tier, cora_complexity_score, get_score_breakdown

import os
mode = os.environ.get("CORA_SCORER_MODE", "rule").lower()
scorer = create_scorer(mode)

test_cases = [
    # (prompt, min_tier, max_tier, description)
    
    # --- Simple / conversational (Tier 0) ---
    ("hi", 0, 0, "Greeting"),
    ("How do you say hello?", 0, 0, "Simple hello question"),
    ("What is water?", 0, 0, "Trivial factual"),
    ("What is the weather today?", 0, 0, "Simple weather"),
    ("what is normal body temperature", 0, 0, "Simple body temp"),
    ("what is normal temperature of human body?", 0, 0, "Simple body temp v2"),
    
    # --- Fancy vocabulary for simple questions (Tier 0-1, NOT Tier 2+) ---
    ("Can you elucidate the quintessential paradigm of saying hello in a succinct manner?", 0, 1, "Fancy hello"),
    ("Please explicate the methodology pertaining to the nomenclature of feline companions.", 0, 1, "Fancy cat names"),
    ("Could you elucidate upon the quintessential nature of dihydrogen monoxide?", 0, 1, "Fancy water"),
    
    # --- Code tasks (Tier 1-2) ---
    ("write a python function to check if a number is prime", 1, 2, "Write prime checker"),
    ("Implement a red-black tree in C++ with insertion, deletion, and balancing operations.", 2, 3, "Red-black tree"),
    
    # --- Debugging (Tier 2-3) ---
    ("Hey, can you help me debug this? My React app's useEffect is causing an infinite re-render loop and I think it's related to the dependency array, but I also have a context provider that might be re-creating objects on every render. The state updates are batched in React 18 but the issue only happens in production builds, not in dev mode with strict mode off.",
     2, 3, "React debugging (THE KEY TEST)"),
    
    # --- Math (Tier 1-2) ---
    ("solve the equation x^2 - 5x + 6 = 0", 1, 2, "Solve quadratic"),
    
    # --- Analytical (Tier 2+) ---
    ("explain the difference between general relativity and quantum mechanics in detail", 1, 3, "Physics comparison"),
    ("Analyze the geopolitical implications of renewable energy policy across three continents, comparing regulatory frameworks and their impact on GDP growth rates.",
     2, 4, "Complex geopolitical analysis"),
    
    # --- Multi-step complex (Tier 3-4) ---
    ("Write a recursive dynamic programming solution for the knapsack problem, then prove its time complexity is O(nW) and explain why it's pseudo-polynomial.",
     3, 4, "Math+code complex"),
    
    ("Explain step by step how transformers work, including the attention mechanism, positional encoding, and compare self-attention vs cross-attention.",
     2, 4, "ML deep explanation"),
]

print("=" * 100)
print("CORA SCORING COMPREHENSIVE TEST")
print("=" * 100)

passed = 0
failed = 0

for prompt, min_tier, max_tier, description in test_cases:
    profile = scorer.score(prompt)
    tier_label, score, budget = score_to_tier(profile, prompt)
    tier_num = int(tier_label.split()[-1])
    
    in_range = min_tier <= tier_num <= max_tier
    if in_range:
        status = "PASS"
        passed += 1
    else:
        status = "FAIL"
        failed += 1
    
    tier_range = f"Tier {min_tier}-{max_tier}" if min_tier != max_tier else f"Tier {min_tier}"
    mark = "[OK]" if in_range else "[!!]"
    
    print(f"\n{mark} {status} | {description}")
    print(f"  Got: {tier_label} (score={score:.3f}) | Expected: {tier_range}")
    print(f"  Task: {profile.task_type.value} | R={profile.reasoning_depth} D={profile.domain_specificity} C={profile.code_complexity} Cr={profile.creative_demand} P={profile.precision_required} S={profile.structural_complexity}")
    
    if not in_range:
        breakdown = get_score_breakdown(profile, prompt)
        top = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Breakdown: {', '.join(f'{k}={v:.3f}' for k,v in top)}")
        print(f"  Signals: {profile.signals}")

print(f"\n{'=' * 100}")
print(f"RESULTS: {passed}/{passed + failed} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print(f"{failed} test(s) FAILED")
print("=" * 100)
