"""
Generate a .docx file with 25 misleading test prompts for CORA tier validation.
5 questions per tier (Tier 0 – Tier 4).

Each prompt is designed to MISLEAD the classifier:
  - Tier 0 prompts use fancy vocabulary but are trivially simple
  - Tier 1 prompts look complex on the surface but are straightforward
  - Tier 2 prompts disguise moderate complexity as easy or hard
  - Tier 3 prompts hide genuine difficulty behind casual language
  - Tier 4 prompts look deceptively simple but require frontier-level reasoning

Run:  python generate_test_docx.py
Output: CORA_Tier_Test_Questions.docx
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

# ── 25 Test Prompts ─────────────────────────────────────────────────────────
# Format: (prompt, expected_tier, mislead_explanation)

QUESTIONS = [
    # ══════════════════════════════════════════════════════════════════════════
    #  TIER 0 – Trivial / Conversational  (score < 0.35)
    #  Mislead strategy: use big words, academic tone, or domain jargon
    #  but the actual ask is trivially simple.
    # ══════════════════════════════════════════════════════════════════════════
    (
        "Greetings!",
        "Tier 0",
        "Single-word greeting. Fancy but trivial. Should NOT be promoted despite sounding formal."
    ),
    (
        "What's the weather like?",
        "Tier 0",
        "Simple conversational query. No reasoning, domain, or code. Extremely short."
    ),
    (
        "Can you elucidate the quintessential paradigm of saying hello in a succinct manner?",
        "Tier 0",
        "MISLEAD: Uses extremely academic vocabulary ('elucidate', 'quintessential', 'paradigm') "
        "but the underlying ask is just 'how do you say hello' — trivially simple."
    ),
    (
        "Tell me a joke",
        "Tier 0",
        "Very short, purely conversational. No analytical or domain requirements."
    ),
    (
        "Utilizing the epistemological framework of modern discourse, kindly furnish a brief salutation.",
        "Tier 0",
        "MISLEAD: Sounds like a PhD thesis prompt but is literally asking for 'say hi'. "
        "The academic language should NOT push this to Tier 1+."
    ),

    # ══════════════════════════════════════════════════════════════════════════
    #  TIER 1 – Simple factual / light analysis  (0.35 ≤ score < 0.70)
    #  Mislead strategy: embed simple questions in complex-sounding framing,
    #  or make them look like Tier 0 by being very casual.
    # ══════════════════════════════════════════════════════════════════════════
    (
        "yo what even is photosynthesis lol explain like im 5",
        "Tier 1",
        "MISLEAD: Extremely casual tone ('yo', 'lol', 'im 5') disguises a factual knowledge question "
        "that requires domain knowledge. Should NOT be demoted to Tier 0 despite slang."
    ),
    (
        "What is the capital of France? Also, what currency do they use and what language do they speak?",
        "Tier 1",
        "Multiple simple factual lookups. Slightly more than trivial but no reasoning needed."
    ),
    (
        "Within the overarching tapestry of European geopolitical history, what is the capital of France?",
        "Tier 1",
        "MISLEAD: Pompous framing around a simple factual question. The academic wrapper "
        "should NOT inflate this beyond Tier 1."
    ),
    (
        "Summarize the main idea of supply and demand in economics in 2-3 sentences.",
        "Tier 1",
        "Simple summarization of a well-known concept. Requires basic domain knowledge but no deep analysis."
    ),
    (
        "hmm idk maybe explain what DNA is? just the basics nothing crazy",
        "Tier 1",
        "MISLEAD: Ultra-casual tone with hedging language. Despite sounding like a Tier 0 chat, "
        "it requires basic biology domain knowledge to explain DNA properly."
    ),

    # ══════════════════════════════════════════════════════════════════════════
    #  TIER 2 – Moderate analysis / simple code  (0.70 ≤ score < 1.15)
    #  Mislead strategy: make it look like Tier 1 (too easy) or Tier 3 (too hard).
    # ══════════════════════════════════════════════════════════════════════════
    (
        "Write a Python function that takes a list of numbers and returns the second largest number. Handle edge cases like duplicates and empty lists.",
        "Tier 2",
        "Moderate coding task. Requires handling edge cases but is a well-known interview problem. "
        "Should NOT be pushed to Tier 3 despite 'edge cases' mention."
    ),
    (
        "just sort a list in python lol but also tell me the time complexity and why quicksort is usually faster than mergesort in practice",
        "Tier 2",
        "MISLEAD: Starts casually ('just sort... lol') but the second half requires analytical comparison "
        "of algorithm performance — genuine moderate complexity hidden behind casual tone."
    ),
    (
        "Compare and contrast the TCP and UDP protocols. When would you use each one? Give real-world examples.",
        "Tier 2",
        "Moderate analytical question requiring domain knowledge in networking. Comparison + examples "
        "elevates it above Tier 1 but it's standard textbook material, not Tier 3."
    ),
    (
        "Explain how a neural network learns through backpropagation. Keep it simple, no math needed.",
        "Tier 2",
        "MISLEAD: 'Keep it simple, no math' makes it sound easy, but explaining backpropagation "
        "conceptually still requires moderate domain depth and reasoning."
    ),
    (
        "I have a CSV file with 10,000 rows of sales data. Write a Python script using pandas to find the top 5 products by revenue per quarter and create a summary table.",
        "Tier 2",
        "Moderate data analysis + code task. Uses pandas (domain) with groupby/aggregation. "
        "Practical but well-documented pattern — not Tier 3 complexity."
    ),

    # ══════════════════════════════════════════════════════════════════════════
    #  TIER 3 – Complex reasoning / advanced code  (1.15 ≤ score < 1.50)
    #  Mislead strategy: use casual language or hide multi-step reasoning
    #  behind seemingly simple asks.
    # ══════════════════════════════════════════════════════════════════════════
    (
        "hey can u help me debug this? my React app's useEffect is causing an infinite re-render loop "
        "and I think it's related to the dependency array but I also have a context provider that "
        "might be re-creating objects on every render. the state updates are batched in React 18 "
        "but the issue only happens in production builds not in dev mode with strict mode off.",
        "Tier 3",
        "MISLEAD: Extremely casual tone ('hey can u help me') hides a genuinely complex debugging scenario "
        "involving React lifecycle, closures, context, production vs dev differences, and React 18 batching."
    ),
    (
        "Design a REST API for a multi-tenant SaaS application with role-based access control. "
        "Include endpoint specifications, authentication flow using JWT with refresh tokens, "
        "rate limiting strategy, and database schema for the authorization layer.",
        "Tier 3",
        "Multi-step system design requiring architecture, security, and database knowledge. "
        "Multiple interrelated components push this firmly into Tier 3."
    ),
    (
        "Prove that the sum of the first n odd numbers equals n². Use mathematical induction and "
        "then provide an alternative geometric proof.",
        "Tier 3",
        "MISLEAD: The statement sounds like a simple math fact, but requiring TWO different proof "
        "methods (induction + geometric) demands structured mathematical reasoning at Tier 3 level."
    ),
    (
        "Implement a thread-safe LRU cache in Python that supports TTL-based expiration, "
        "handles concurrent reads/writes without deadlocks, and includes eviction callbacks.",
        "Tier 3",
        "Advanced data structure + concurrency problem. Thread safety, TTL, and callbacks "
        "create genuine multi-dimensional complexity."
    ),
    (
        "walk me through how kubernetes handles pod scheduling when you have node affinity rules, "
        "taints and tolerations, resource requests vs limits, and priority classes all competing. "
        "what happens when there's a conflict?",
        "Tier 3",
        "MISLEAD: Casual 'walk me through' framing disguises a deeply technical systems question "
        "requiring knowledge of K8s scheduler internals and multi-constraint resolution."
    ),

    # ══════════════════════════════════════════════════════════════════════════
    #  TIER 4 – Frontier-level / research-grade  (score ≥ 1.50)
    #  Mislead strategy: make it look deceptively simple or frame it casually
    #  when it actually requires cutting-edge reasoning.
    # ══════════════════════════════════════════════════════════════════════════
    (
        "Given a distributed system with eventual consistency, design a conflict-free replicated "
        "data type (CRDT) for a collaborative text editor that supports concurrent insertions, "
        "deletions, and undo operations across multiple nodes with network partitions. "
        "Provide the mathematical specification using semilattice theory, prove convergence, "
        "implement it in Rust with proper memory management, and analyze the space complexity "
        "tradeoffs compared to OT-based approaches. Include benchmarks.",
        "Tier 4",
        "Multi-domain frontier problem: distributed systems theory, formal mathematics, "
        "systems programming in Rust, and performance analysis — all in one prompt."
    ),
    (
        "can you casually explain how to build a compiler from scratch? I need a tokenizer, parser "
        "with error recovery, type inference engine supporting parametric polymorphism and "
        "higher-kinded types, SSA-based IR with optimization passes (constant folding, dead code "
        "elimination, loop-invariant code motion), and LLVM backend code generation. oh and make "
        "it support incremental compilation.",
        "Tier 4",
        "MISLEAD: 'casually explain' and 'oh and make it' hide an enormous frontier-level compiler "
        "engineering task spanning lexing, parsing, type theory, optimization, and code generation."
    ),
    (
        "Develop a novel attention mechanism that achieves sub-quadratic complexity for transformer "
        "models while maintaining the expressiveness of full self-attention. Provide the theoretical "
        "foundation using kernel methods, prove the approximation bounds, implement it in PyTorch "
        "with custom CUDA kernels for the forward and backward passes, and design an ablation study "
        "comparing against FlashAttention, linear attention, and sparse attention on language "
        "modeling benchmarks.",
        "Tier 4",
        "Research-grade ML problem requiring novel algorithm design, mathematical proofs, "
        "GPU kernel programming, and rigorous experimental methodology."
    ),
    (
        "Write a formal verification proof in Coq or Lean 4 that a red-black tree implementation "
        "maintains its invariants (balanced height, no red-red edges, root is black) across all "
        "insertion and deletion operations. Then extract a certified executable and benchmark it "
        "against std::map in C++. Discuss the proof engineering trade-offs between Coq's tactic "
        "language and Lean 4's term-mode proofs.",
        "Tier 4",
        "Formal methods + systems programming crossover. Requires theorem proving expertise, "
        "data structure theory, performance engineering, and meta-level proof methodology comparison."
    ),
    (
        "so uh I have this weird problem... I need to solve the protein folding prediction for a "
        "novel enzyme using a combination of molecular dynamics simulation, graph neural networks "
        "on the residue contact map, and evolutionary sequence alignment with MSA transformer. "
        "the catch is the protein has non-standard amino acids and metal cofactors that break "
        "standard force fields. can you design the computational pipeline, suggest force field "
        "modifications, architect the GNN, and estimate computational cost on an A100 cluster?",
        "Tier 4",
        "MISLEAD: 'so uh I have this weird problem' is maximally casual but this is a cutting-edge "
        "computational biology problem requiring molecular dynamics, deep learning, bioinformatics, "
        "and HPC expertise — genuine frontier-level work."
    ),
]


# ── Build the .docx ────────────────────────────────────────────────────────────

def build_docx(output_path: str = "CORA_Tier_Test_Questions.docx"):
    doc = Document()

    # ── Title ──
    title = doc.add_heading("CORA Tier Classification – Test Questions", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ── Subtitle / description ──
    intro = doc.add_paragraph()
    intro.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = intro.add_run(
        "25 Misleading Test Prompts for Tier Validation\n"
        "Each prompt is designed to trick the classifier while having a clear correct tier."
    )
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_paragraph()  # spacer

    # ── Tier threshold reference ──
    ref = doc.add_heading("Tier Thresholds (complexity_score.py)", level=2)
    table = doc.add_table(rows=6, cols=3, style="Light Shading Accent 1")
    headers = ["Tier", "NeMo Score Range", "Budget Score Range"]
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
        for paragraph in table.rows[0].cells[i].paragraphs:
            for run in paragraph.runs:
                run.bold = True

    tiers_data = [
        ("Tier 0", "< 0.35", "≤ 20"),
        ("Tier 1", "0.35 – 0.69", "21 – 45"),
        ("Tier 2", "0.70 – 1.14", "46 – 70"),
        ("Tier 3", "1.15 – 1.49", "71 – 88"),
        ("Tier 4", "≥ 1.50", "89 – 100"),
    ]
    for row_idx, (tier, nemo, budget) in enumerate(tiers_data, start=1):
        table.rows[row_idx].cells[0].text = tier
        table.rows[row_idx].cells[1].text = nemo
        table.rows[row_idx].cells[2].text = budget

    doc.add_paragraph()  # spacer

    # ── Questions ──
    current_tier = None
    for q_num, (prompt, expected_tier, explanation) in enumerate(QUESTIONS, start=1):
        # Tier section header
        if expected_tier != current_tier:
            current_tier = expected_tier
            doc.add_heading(f"─── {current_tier} ───", level=1)

        # Question number + expected tier
        q_heading = doc.add_heading(f"Q{q_num}  │  Expected: {expected_tier}", level=2)

        # Prompt
        prompt_label = doc.add_paragraph()
        label_run = prompt_label.add_run("Prompt: ")
        label_run.bold = True
        label_run.font.size = Pt(11)

        prompt_para = doc.add_paragraph()
        prompt_para.style = "Quote"
        prompt_run = prompt_para.add_run(prompt)
        prompt_run.font.size = Pt(11)

        # Mislead explanation
        expl_para = doc.add_paragraph()
        expl_label = expl_para.add_run("Why this is misleading: ")
        expl_label.bold = True
        expl_label.font.size = Pt(10)
        expl_label.font.color.rgb = RGBColor(180, 60, 60)
        expl_text = expl_para.add_run(explanation)
        expl_text.font.size = Pt(10)
        expl_text.font.color.rgb = RGBColor(100, 100, 100)

        # Result fields (blank for filling in)
        result_para = doc.add_paragraph()
        result_run = result_para.add_run("Actual Tier: ________    Score: ________    ✓/✗: ________")
        result_run.font.size = Pt(10)
        result_run.font.color.rgb = RGBColor(60, 60, 180)

        doc.add_paragraph()  # spacer between questions

    # ── Summary table at the end ──
    doc.add_page_break()
    doc.add_heading("Results Summary", level=1)

    summary = doc.add_table(rows=26, cols=5, style="Light Shading Accent 1")
    summary_headers = ["Q#", "Expected Tier", "Actual Tier", "Score", "Pass/Fail"]
    for i, h in enumerate(summary_headers):
        summary.rows[0].cells[i].text = h
        for paragraph in summary.rows[0].cells[i].paragraphs:
            for run in paragraph.runs:
                run.bold = True

    for row_idx in range(1, 26):
        summary.rows[row_idx].cells[0].text = f"Q{row_idx}"
        summary.rows[row_idx].cells[1].text = QUESTIONS[row_idx - 1][1]

    # ── Save ──
    doc.save(output_path)
    print(f"[OK] Generated: {output_path}")
    print(f"     {len(QUESTIONS)} questions across 5 tiers (5 per tier)")


if __name__ == "__main__":
    build_docx()
