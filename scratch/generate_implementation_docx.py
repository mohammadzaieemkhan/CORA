"""
Generate a comprehensive DOCX of the CORA Implementation Chapter.
Run: python scratch/generate_implementation_docx.py
Output: scratch/CORA_Implementation_Chapter.docx
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

OUTPUT_PATH = r"C:\Users\Mohammad Zaieem Khan\Videos\CORA 2\CORA\scratch\CORA_Implementation_Chapter.docx"


def set_paragraph_spacing(para, before=0, after=6, line_spacing=None):
    pf = para.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if line_spacing:
        from docx.shared import Pt as DPt
        pf.line_spacing = DPt(line_spacing)


def add_heading(doc, text, level=1):
    heading = doc.add_heading(text, level=level)
    if level == 1:
        heading.runs[0].font.size = Pt(16)
        heading.runs[0].font.bold = True
        heading.runs[0].font.color.rgb = RGBColor(0x1F, 0x35, 0x64)
    elif level == 2:
        heading.runs[0].font.size = Pt(13)
        heading.runs[0].font.bold = True
        heading.runs[0].font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
    elif level == 3:
        heading.runs[0].font.size = Pt(11)
        heading.runs[0].font.bold = True
        heading.runs[0].font.color.rgb = RGBColor(0x1F, 0x35, 0x64)
    set_paragraph_spacing(heading, before=12, after=4)
    return heading


def add_body(doc, text, bold=False, italic=False):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.font.size = Pt(11)
    run.font.name = "Times New Roman"
    run.bold = bold
    run.italic = italic
    set_paragraph_spacing(para, before=0, after=6)
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return para


def add_caption(doc, text):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.font.size = Pt(10)
    run.font.italic = True
    run.font.name = "Times New Roman"
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_spacing(para, before=2, after=10)
    return para


def add_table_header(doc, headers, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for run in hdr_cells[i].paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(10)
        # shade header
        tc = hdr_cells[i]._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), '2E74B5')
        tcPr.append(shd)
        for run in hdr_cells[i].paragraphs[0].runs:
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    return table


def add_row(table, values):
    row_cells = table.add_row().cells
    for i, v in enumerate(values):
        row_cells[i].text = v
        for run in row_cells[i].paragraphs[0].runs:
            run.font.size = Pt(10)
    return row_cells


doc = Document()

# ── Page margins ──────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.17)
    section.right_margin = Cm(2.54)

# ── Default font ──────────────────────────────────────────────────────────────
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(11)

# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER TITLE
# ══════════════════════════════════════════════════════════════════════════════
ch = doc.add_heading("CHAPTER 5: IMPLEMENTATION", level=1)
ch.runs[0].font.size = Pt(18)
ch.runs[0].font.bold = True
ch.runs[0].font.color.rgb = RGBColor(0x1F, 0x35, 0x64)
ch.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_paragraph_spacing(ch, before=0, after=16)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# INTRODUCTION
# ══════════════════════════════════════════════════════════════════════════════
add_body(doc,
    "The implementation methodology of CORA (Cognitive Orchestration and Reasoning Allocator) was "
    "designed around the objective of building a real-time, cognition-aware middleware system capable "
    "of dynamically orchestrating multiple Large Language Models while maintaining low latency, high "
    "scalability, modular extensibility, and computational efficiency. The development process followed "
    "a structured engineering methodology that integrated frontend interface construction, asynchronous "
    "backend orchestration, engineered cognitive analysis, adaptive routing logic, streaming "
    "communication infrastructure, and persistent telemetry into a unified deployment pipeline."
)

add_body(doc,
    "Unlike conventional AI applications that directly invoke a single model endpoint, CORA required "
    "the coordinated implementation of multiple computational subsystems operating concurrently across "
    "distinct architectural layers. Each subsystem was developed independently with well-defined "
    "interfaces, and then integrated through asynchronous event-driven pipelines. The implementation "
    "therefore emphasized asynchronous execution, modular software engineering principles, provider "
    "abstraction, telemetry observability, and scalable orchestration patterns to ensure that the "
    "complete system could handle concurrent inference sessions without introducing processing "
    "bottlenecks or degrading routing accuracy."
)

add_body(doc,
    "The following sections describe each major implementation component in detail, covering the "
    "specific technologies employed, architectural decisions made, and engineering challenges "
    "addressed during development."
)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: FRONTEND IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "5.1  Frontend Implementation Methodology", level=2)

add_body(doc,
    "The CORA frontend was implemented using React 19 with Vite 8 as the build environment. "
    "React 19 was selected because its concurrent rendering architecture and compositional component "
    "model aligned well with the real-time interface requirements of a cognitive routing system. "
    "Vite 8 was chosen as the build tool because it provides near-instantaneous hot module "
    "replacement during development and produces highly optimised production bundles through "
    "native ES module handling and Rollup-based tree-shaking, resulting in faster load times "
    "compared to traditional Webpack-based configurations."
)

add_body(doc,
    "The frontend architecture follows a component-oriented design model in which independent "
    "interface modules communicate through centralised state management and asynchronous event "
    "handling. State was managed without external libraries such as Redux by exploiting React's "
    "built-in useState, useEffect, and useCallback hooks in conjunction with Context API for "
    "cross-component data sharing. This deliberate design decision kept the dependency tree "
    "minimal while preserving the reactivity required for real-time routing metadata display."
)

add_heading(doc, "5.1.1  Component Architecture", level=3)

add_body(doc,
    "The implemented frontend component hierarchy was structured to separate concerns across "
    "navigational, interactive, analytical, and decorative layers. The Navbar component provides "
    "session-aware navigation with dynamic authentication state rendering, toggling between "
    "logged-in and guest views based on token presence in local storage. The Hero Section serves "
    "as the primary entry point, combining animated text sequences with direct prompt submission "
    "capability, enabling users to initiate cognitive queries without navigating away from the "
    "landing view."
)

add_body(doc,
    "The PromptInput component was implemented with multi-line auto-expanding textarea logic, "
    "keyboard shortcut handling (Shift+Enter for newlines, Enter for submission), and integrated "
    "character budget indicators. The ResultsPanel was designed as a streaming-aware display "
    "module that consumes Server-Sent Event data in real time, progressively rendering token "
    "chunks as they arrive from the backend while simultaneously displaying cognitive profile "
    "metadata, tier assignment badges, and routing reason explanations received in the initial "
    "metadata event."
)

add_body(doc,
    "The Dashboard component aggregates all telemetry visualisations into a unified analytics "
    "view, consuming the /v1/stats and /v1/user/history API endpoints to display tier distribution "
    "charts, token savings summaries, and query history tables. The Sidebar provides session "
    "management and navigation shortcuts. The PipelineViz component renders an animated "
    "representation of the CORA processing pipeline, using sequenced Framer Motion animations "
    "to visualise the stages from prompt ingestion through cognitive analysis, budget scoring, "
    "tier routing, and response streaming."
)

add_body(doc,
    "Supporting components include the ParticleCanvas, which renders a GPU-accelerated WebGL "
    "particle system for background ambient animation, and the PromptOptimizer, which communicates "
    "with the /v1/optimize endpoint to display side-by-side comparisons of original and "
    "optimised prompt metrics including token counts, tier assignments, and projected cost savings. "
    "Authentication Modules handle registration and login flows with token persistence to "
    "localStorage. Statistics Panels render paginated query history with expandable cognitive "
    "profile breakdowns. Footer Components display system information and navigation links."
)

add_body(doc,
    "Table 5.1 summarises the implemented component hierarchy and the primary responsibility "
    "of each module."
)

# Table 5.1
t51 = add_table_header(doc, ["Component", "Layer", "Primary Responsibility"])
rows_51 = [
    ("Navbar", "Navigation", "Session-aware routing, authentication state toggle"),
    ("Hero Section", "Presentation", "Animated landing view with inline prompt submission"),
    ("PromptInput", "Interaction", "Multi-line input, keyboard shortcuts, character budget display"),
    ("ResultsPanel", "Streaming Display", "Real-time SSE token rendering, cognitive profile display"),
    ("Dashboard", "Analytics", "Telemetry visualisation, tier distribution, query history"),
    ("Sidebar", "Navigation", "Session management, quick-access navigation shortcuts"),
    ("PipelineViz", "Visualisation", "Animated rendering of the CORA processing pipeline stages"),
    ("ParticleCanvas", "Decoration", "WebGL ambient particle system background animation"),
    ("PromptOptimizer", "Tool", "Prompt efficiency comparison with token and tier metrics"),
    ("Authentication Modules", "Security", "User registration, login, token lifecycle management"),
    ("Statistics Panels", "Analytics", "Paginated history with expandable cognitive breakdowns"),
    ("Footer Components", "Navigation", "System information display and navigation links"),
]
for r in rows_51:
    add_row(t51, list(r))
doc.add_paragraph()
add_caption(doc, "Table 5.1: CORA Frontend Component Hierarchy and Responsibilities")

add_heading(doc, "5.1.2  SSE Integration and Streaming Rendering", level=3)

add_body(doc,
    "Server-Sent Event integration was implemented within the PromptInput and ResultsPanel "
    "components using the browser's native EventSource API extended with custom fetch-based "
    "streaming to support POST requests carrying the query payload. Upon submission, the frontend "
    "opens a streaming connection to the /v1/query/stream endpoint and registers event listeners "
    "for four distinct event types: meta, token, done, and error. The meta event arrives first, "
    "carrying the tier assignment, model name, budget score, cognitive profile dimensions, and "
    "routing reason, which are immediately rendered in the interface without waiting for the "
    "response content. Subsequent token events arrive in approximately 32-character chunks every "
    "10 milliseconds, which the ResultsPanel appends progressively to the displayed response "
    "buffer. The done event triggers final telemetry display including latency in milliseconds, "
    "tokens used, and tokens saved relative to a GPT-4o cost baseline."
)

add_heading(doc, "5.1.3  Visual System and Design Language", level=3)

add_body(doc,
    "The visual design system implemented for CORA was named the Ethereal Engine and was built "
    "exclusively using Vanilla CSS with CSS custom properties as design tokens, deliberately "
    "avoiding utility-class frameworks to achieve maximum stylistic control. The system implements "
    "a dark-mode-first colour palette anchored around deep indigo, electric violet, and teal "
    "accent colours communicating technical precision and cognitive depth. Glassmorphism "
    "abstractions were achieved through backdrop-filter blur composited over layered semi-transparent "
    "gradients, creating depth without relying on heavy shadow systems. Aurora effects were "
    "implemented using radial gradient animations cycling through the primary colour palette at "
    "variable durations to produce an ambient luminescent background. Framer Motion was integrated "
    "to provide fluid entrance transitions, staggered list renders, exit animations, and layout "
    "transition smoothing, with all animations governed by physics-based spring configurations "
    "rather than fixed duration curves to ensure natural motion feel."
)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: BACKEND IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "5.2  Backend Implementation Methodology", level=2)

add_body(doc,
    "The CORA backend was implemented using FastAPI version 0.111, selected because its "
    "ASGI-native architecture is purpose-built for highly concurrent asynchronous workloads. "
    "In a system like CORA that must simultaneously process cognitive analysis, invoke remote "
    "inference APIs, stream response tokens, and persist telemetry, a synchronous WSGI framework "
    "such as Flask or Django would introduce blocking I/O bottlenecks that would collapse "
    "throughput under concurrent load. FastAPI's Starlette foundation enables the backend to "
    "handle all of these operations within a single Python event loop without spawning threads, "
    "achieving scalable concurrency through async/await coroutines."
)

add_body(doc,
    "The backend was designed as the central coordination engine responsible for request "
    "handling, cognitive analysis orchestration, provider communication, streaming response "
    "management, telemetry persistence, and adaptive routing execution. All external "
    "communications, including database queries and LLM API calls, were implemented as "
    "non-blocking coroutines to ensure that a slow external provider could not stall the "
    "processing of other concurrent requests."
)

add_heading(doc, "5.2.1  API Structure and Route Organisation", level=3)

add_body(doc,
    "The API surface was organised under versioned route prefixes beginning with /v1/ to "
    "establish forward compatibility for future schema evolution. Eight distinct API groups "
    "were implemented as documented in Table 5.2. All endpoints except the frontend catch-all "
    "route operate under this versioned namespace, ensuring that the administrative and "
    "inference APIs are clearly separated from static asset serving."
)

# Table 5.2
t52 = add_table_header(doc, ["API Group", "Endpoint Prefix", "Primary Function"])
rows_52 = [
    ("Authentication APIs", "POST /v1/auth/register, /v1/auth/login, /v1/auth/logout",
     "User account creation, credential validation, session token management"),
    ("User Management APIs", "GET/PUT /v1/user/profile, GET/DELETE /v1/user/history",
     "Profile retrieval, profile updates, paginated query history management"),
    ("Query Execution APIs", "POST /v1/query",
     "Synchronous prompt routing with complete response and telemetry"),
    ("Streaming APIs", "POST /v1/query/stream",
     "SSE-based streaming endpoint for real-time token delivery"),
    ("Cognitive Analysis APIs", "POST /v1/cognitive-profile",
     "Prompt complexity analysis without LLM invocation"),
    ("Prompt Optimisation APIs", "POST /v1/optimize",
     "AI-assisted prompt efficiency optimisation with metric comparison"),
    ("Statistics APIs", "GET /v1/stats",
     "Aggregated global telemetry and routing distribution statistics"),
    ("Frontend Serving", "GET /{full_path:path}",
     "Catch-all SPA routing serving index.html for client-side navigation"),
]
for r in rows_52:
    add_row(t52, list(r))
doc.add_paragraph()
add_caption(doc, "Table 5.2: CORA Backend API Groups and Route Organisation")

add_heading(doc, "5.2.2  Middleware Integration", level=3)

add_body(doc,
    "The middleware stack was configured to address cross-origin communication, static asset "
    "serving, and request lifecycle management. CORS middleware was added with permissive "
    "origin settings during development to allow the Vite development server running on port "
    "5173 to communicate with the FastAPI backend on port 8000 without browser security "
    "rejections. In production, this configuration would be tightened to a specific origin "
    "whitelist. Static file middleware was configured to serve the compiled frontend assets "
    "directory, enabling CORA to operate as a self-contained single-server deployment where "
    "both the React application and the API backend are served from the same Python process."
)

add_body(doc,
    "Request lifecycle handling was implemented through the FastAPI lifespan context manager, "
    "which coordinates the startup sequence including database pool initialisation via "
    "init_db(), cognitive scorer construction via create_scorer(), and optional ML pipeline "
    "warm-up if a non-rule scorer is configured. The shutdown sequence ensures that all "
    "persistent HTTP client pools managed by httpx.AsyncClient are gracefully drained before "
    "the process exits, preventing connection leak warnings."
)

add_heading(doc, "5.2.3  Score Caching Layer", level=3)

add_body(doc,
    "To eliminate redundant cognitive analysis overhead for repeated identical prompts, a "
    "lightweight MD5-keyed in-memory score cache was implemented at the application level. "
    "The _cached_profile() function computes an MD5 hash of the prompt text, checks the "
    "cache dictionary, and returns the stored CognitiveProfile immediately if a match is "
    "found. Cache entries are evicted using a first-in-first-out strategy once the cache "
    "exceeds 512 entries, preventing unbounded memory growth during long-running sessions. "
    "This optimisation is particularly valuable for the streaming endpoint where cognitive "
    "analysis precedes LLM invocation, as repeated analytical queries can be served in "
    "microseconds rather than tens of milliseconds."
)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: COGNITIVE ENGINE IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "5.3  Implementation of the Cognitive Engine", level=2)

add_body(doc,
    "The Cognitive Engine was implemented as a dedicated Python package located at "
    "cognitive_module/, comprising eleven independent Python modules each responsible for "
    "a specific concern within the cognitive analysis pipeline. The package was structured "
    "according to the Single Responsibility Principle to ensure that changes to the scoring "
    "strategy, feature extraction logic, or data models could be made independently without "
    "rippling through unrelated components."
)

add_body(doc,
    "The implemented package structure and the responsibility of each module are described "
    "in Table 5.3."
)

# Table 5.3
t53 = add_table_header(doc, ["Module File", "Responsibility"])
rows_53 = [
    ("models.py", "Defines CognitiveProfile dataclass, TaskType enum, and TASK_TYPE_META boost registry"),
    ("config.py", "CognitiveConfig dataclass holding dimension weights, tier boundaries, and scorer mode"),
    ("feature_extractor.py", "FeatureExtractor class producing 45 named numerical features from raw prompt text"),
    ("scorer.py", "BaseScorer abstract interface, create_scorer() factory, and _ScorerAdapter wrapper"),
    ("rule_scorer.py", "RuleBasedScorer: heuristic scoring consuming extracted features without ML inference"),
    ("ml_scorer.py", "DistilBERTScorer: transformer-based scoring using fine-tuned DistilBERT checkpoint"),
    ("nemo_scorer.py", "NeMoScorer: DeBERTa-based scoring using NVIDIA NeMo inference pipeline"),
    ("llm_scorer.py", "LLMBasedScorer: GPT-API-based scoring via structured JSON prompt engineering"),
    ("routing.py", "profile_to_budget_score(), score_to_tier(), generate_routing_reason() functions"),
    ("training_data.py", "Annotated prompt corpus and label generation utilities for ML scorer training"),
    ("__init__.py", "Public API surface exporting create_scorer, generate_routing_reason, TASK_TYPE_META"),
]
for r in rows_53:
    add_row(t53, list(r))
doc.add_paragraph()
add_caption(doc, "Table 5.3: cognitive_module/ Package Structure and Module Responsibilities")

add_heading(doc, "5.3.1  Feature Extraction Implementation", level=3)

add_body(doc,
    "The FeatureExtractor class implemented in feature_extractor.py is the analytical "
    "foundation of the cognitive pipeline. It accepts a raw prompt string and returns a "
    "named dictionary of 45 numerical features organised into seven analytical groups: "
    "text statistics, reasoning signals, domain specificity, code complexity, creative demand, "
    "precision requirement, and structural complexity. These features are used both by the "
    "rule-based scorer for direct heuristic scoring and by the machine learning scorer as "
    "supplementary handcrafted features concatenated with transformer embeddings."
)

add_body(doc,
    "Feature extraction begins with a vocabulary normalisation step that replaces ornamental "
    "synonyms and pretentious academic language with plain equivalents before keyword matching. "
    "This prevents artificially inflating complexity scores for prompts that use verbose "
    "phrasing to express fundamentally simple requests. For example, the phrase 'elucidate the "
    "mechanism' is normalised to 'explain the mechanism' before reasoning marker matching, "
    "preventing the ornamental verb from triggering false high-complexity signals. The "
    "deflation dictionary contains over 60 such normalisations across verbs, nouns, adjectives, "
    "and adverbial phrases."
)

add_body(doc,
    "The reasoning feature group extracts signals from a curated vocabulary of 80 reasoning "
    "markers with associated point weights. Phrases such as 'step by step' contribute 14 "
    "points, 'synthesize' contributes 14 points, 'from first principles' contributes 16 points, "
    "and 'because' contributes 6 points. The total weighted sum is accumulated as the "
    "reasoning_keyword_score, while causal connectors such as 'therefore', 'hence', and "
    "'as a result' are counted separately to distinguish causal reasoning chains from general "
    "analytical requests. Conditional clause frequency is measured through regular expression "
    "matching for if, when, unless, whether, and assuming keywords."
)

add_body(doc,
    "The domain specificity group implements multi-domain vocabulary matching across six "
    "domain taxonomies: computer science (over 120 domain terms), medicine (20 terms), law "
    "(16 terms), science (18 terms), finance (18 terms), and mathematics (30 terms). Each "
    "matched term increments the domain hit counter for its respective domain. Cross-domain "
    "scoring applies a bonus of 10 points for prompts spanning two domains and 20 points "
    "for prompts spanning three or more domains, reflecting that multi-domain queries typically "
    "require broader and deeper reasoning synthesis than single-domain queries."
)

add_body(doc,
    "The code complexity group detects four types of code signals: fenced code blocks using "
    "triple backtick delimiters, inline code using single backtick delimiters, code keyword "
    "presence from a vocabulary of 70 programming-related terms, and error trace patterns. "
    "Code blocks contribute base points scaled by the number of lines they contain, rewarding "
    "longer code submissions with higher complexity scores proportional to the debugging or "
    "analysis surface area. Error trace detection via regular expression matching of patterns "
    "such as Traceback, NullPointerException, and TypeError adds 20 points, reflecting the "
    "additional diagnostic reasoning required for debugging tasks."
)

add_body(doc,
    "The filler detection subsystem represents a novel contribution to the feature extraction "
    "pipeline. It segments the prompt into individual sentences and classifies each sentence "
    "as either actionable (containing a direct instruction or question) or passive (providing "
    "contextual background). A filler ratio is computed as the proportion of passive sentences. "
    "Prompts with a filler ratio exceeding 0.70 across three or more sentences, where the "
    "actionable sentences do not reference the surrounding context, are flagged as "
    "filler-padded. This flag suppresses inflated structural complexity scores that would "
    "otherwise be triggered by the presence of many sentences, even if most of those sentences "
    "are non-substantive padding."
)

add_heading(doc, "5.3.2  Scorer Strategy Implementation", level=3)

add_body(doc,
    "The scorer implementation adopted the Strategy Pattern through a shared BaseScorer "
    "abstract base class. The BaseScorer defines a single abstract method, score(prompt: str) "
    "→ CognitiveProfile, which all concrete scorer implementations must satisfy. The "
    "create_scorer() factory function constructs the appropriate scorer based on either "
    "an explicit mode argument or the CORA_SCORER_MODE environment variable, allowing "
    "deployment-time scorer selection without code changes."
)

add_body(doc,
    "The Rule-Based Scorer implemented in rule_scorer.py consumes the 45-feature vector "
    "from the FeatureExtractor and applies weighted heuristic formulas to produce six "
    "cognitive dimension scores each in the range 0 to 100. The reasoning_depth dimension "
    "is derived from the reasoning keyword score, conditional clause count, and causal "
    "connector count. The domain_specificity dimension is taken from the domain specificity "
    "score computed during feature extraction. The code_complexity dimension combines code "
    "block presence, keyword density, and error trace signals. The creative_demand dimension "
    "reflects creative keyword scores and open-ended phrasing density. The precision_required "
    "dimension captures mathematical expression frequency, precision keyword presence, and "
    "numeric density. The structural_complexity dimension reflects sentence count, word count, "
    "numbered list depth, and constraint marker frequency."
)

add_body(doc,
    "The Machine Learning Scorer implemented in ml_scorer.py uses a DistilBERT-base-uncased "
    "transformer fine-tuned on CORA's annotated prompt corpus. The DistilBERT tokenizer "
    "processes prompt text up to a maximum sequence length of 256 tokens. The fine-tuned "
    "model produces logit outputs that are converted to probability distributions over "
    "cognitive dimension bins through a softmax layer. When the ML scorer's confidence "
    "estimate falls below the configured threshold of 0.40, execution automatically falls "
    "back to the rule-based scorer, ensuring that the system never produces a refused or "
    "undefined cognitive profile even when transformer inference yields a low-confidence result."
)

add_body(doc,
    "The NeMo Scorer implemented in nemo_scorer.py extends the ML scoring pipeline using "
    "a DeBERTa-based architecture accessed through NVIDIA NeMo inference pipelines. DeBERTa "
    "employs disentangled attention mechanisms that separately encode content and positional "
    "information, improving performance on text understanding tasks requiring sensitivity to "
    "word order and clause structure. The LLM-Assisted Scorer implemented in llm_scorer.py "
    "submits a structured system prompt to a remote language model API, requesting a JSON "
    "response containing numerical dimension scores and a confidence estimate, providing "
    "the highest semantic understanding at the cost of additional inference latency."
)

add_body(doc,
    "All four scorer implementations return a standardised CognitiveProfile object containing "
    "the six dimension scores, a confidence estimate between 0 and 1, the detected TaskType "
    "classification, a list of explainability signals describing which features contributed "
    "most to the score, and the scorer_used label identifying which scoring strategy produced "
    "the profile. This standardisation enables the routing layer to operate identically "
    "regardless of which scorer backend is active."
)

add_body(doc,
    "Table 5.4 presents the eight TaskType classifications and their associated complexity "
    "boost values applied during budget score computation."
)

# Table 5.4
t54 = add_table_header(doc, ["Task Type", "Budget Boost", "Description"])
rows_54 = [
    ("FACTUAL", "0", "Direct factual recall questions with no analytical requirement"),
    ("ANALYTICAL", "+10", "Comparative, evaluative, or reasoning-intensive requests"),
    ("CODE", "+15", "Programming tasks, algorithm implementation, or code explanation"),
    ("DEBUGGING", "+20", "Error diagnosis, root cause analysis, and code repair tasks"),
    ("CREATIVE", "+5", "Story generation, brainstorming, and open-ended creative tasks"),
    ("CONVERSATIONAL", "-5", "Casual dialogue, greetings, and simple chitchat"),
    ("MATHEMATICAL", "+12", "Numerical computation, proof derivation, and formula application"),
    ("MULTI_STEP", "+15", "Compound instructions requiring sequential multi-phase execution"),
]
for r in rows_54:
    add_row(t54, list(r))
doc.add_paragraph()
add_caption(doc, "Table 5.4: TaskType Classifications and Cognitive Budget Boost Values")

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: THOUGHT BUDGET ALLOCATOR
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "5.4  Implementation of the Thought Budget Allocator", level=2)

add_body(doc,
    "The Thought Budget Allocator (TBA) was implemented as a deterministic aggregation engine "
    "within the routing.py module, responsible for converting the six-dimensional CognitiveProfile "
    "into a single scalar Cognitive Budget Score in the range 1 to 100. This scalar value then "
    "drives the tier assignment decision through threshold comparisons against pre-configured "
    "tier boundaries."
)

add_heading(doc, "5.4.1  Hybrid Aggregation Formula", level=3)

add_body(doc,
    "A simple unweighted average of the six cognitive dimensions would tend to suppress routing "
    "accuracy for prompts that are exceptionally demanding on two or three dimensions while "
    "scoring zero on the remaining dimensions. For example, a highly complex code debugging "
    "request might score 90 on code_complexity and 80 on reasoning_depth while scoring near "
    "zero on creative_demand and domain_specificity, yielding an unweighted average of "
    "approximately 29, which would incorrectly route the request to a low-capability tier."
)

add_body(doc,
    "The implemented hybrid formula addresses this limitation by blending two complementary "
    "aggregation strategies. The first component is a weighted dimensional average in which "
    "reasoning_depth and code_complexity each contribute 25% weight, domain_specificity and "
    "structural_complexity each contribute 15%, and creative_demand and precision_required "
    "each contribute 10%. The second component is a peak signal average computed as the mean "
    "of the two highest-scoring dimensions regardless of their weight. The final hybrid score "
    "is computed as 50% weighted average plus 50% peak signal average. A task-type boost is "
    "then applied from the TASK_TYPE_META registry according to the detected TaskType. Finally, "
    "scores above 10 are scaled by a factor of 1.25 to expand the effective scoring range and "
    "ensure that high-complexity prompts reliably reach the Tier 4 boundary at 89."
)

add_body(doc,
    "The implemented formula can be expressed as:"
)

# Formula block (as styled paragraph)
formula_para = doc.add_paragraph()
formula_run = formula_para.add_run(
    "    Budget Score = clamp(round((W_avg × 0.5 + Peak_avg × 0.5 + TaskBoost) × 1.25), 1, 100)\n"
    "    Where:\n"
    "    W_avg    = Σ (dimension_i × weight_i)\n"
    "    Peak_avg = (top_dimension_1 + top_dimension_2) / 2\n"
    "    TaskBoost = TASK_TYPE_META[task_type]['boost']"
)
formula_run.font.name = "Courier New"
formula_run.font.size = Pt(10)
formula_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
set_paragraph_spacing(formula_para, before=4, after=8)

add_heading(doc, "5.4.2  Tier Boundary Configuration", level=3)

add_body(doc,
    "The tier assignment boundaries were implemented as a configurable sorted list of "
    "(max_score_inclusive, tier_name) tuples evaluated top-to-bottom with first-match "
    "semantics. This structure allows boundary thresholds to be adjusted through the "
    "CognitiveConfig dataclass without modifying any routing logic. The default tier "
    "boundaries established through iterative calibration are presented in Table 5.5."
)

# Table 5.5
t55 = add_table_header(doc, ["Tier", "Budget Score Range", "Primary Model", "Fallback Model", "Typical Query Category"])
rows_55 = [
    ("Tier 0", "1 – 20",   "Nemotron Mini 4B (NVIDIA NIM)",     "Gemma 3n E4B",        "Greetings, simple facts, basic definitions"),
    ("Tier 1", "21 – 45",  "Nemotron Nano 9B v2 (NVIDIA NIM)",  "—",                   "Moderate factual queries, short explanations"),
    ("Tier 2", "46 – 70",  "Nemotron Nano 30B (NVIDIA NIM)",    "—",                   "Analytical reasoning, creative tasks, domain queries"),
    ("Tier 3", "71 – 88",  "Mistral Medium 3.5 (Mistral API)",  "Nemotron Super 120B", "Complex reasoning, multi-step problems, debugging"),
    ("Tier 4", "89 – 100", "Qwen3 Coder 480B (NVIDIA NIM)",     "Qwen3.5 397B",        "Hardest code generation, proof derivation, multi-domain"),
]
for r in rows_55:
    add_row(t55, list(r))
doc.add_paragraph()
add_caption(doc, "Table 5.5: Tier Boundary Configuration, Model Assignments, and Query Categories")

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: ROUTING INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "5.5  Implementation of the Routing Infrastructure", level=2)

add_body(doc,
    "The routing infrastructure was implemented through the llm_providers/ package, which "
    "abstracts all communication with external LLM providers behind a consistent Python "
    "module interface. The package registers eight model modules organised into five tiers, "
    "with dedicated primary and fallback assignments for Tier 0, Tier 3, and Tier 4. "
    "The selection of NVIDIA NIM APIs as the primary inference substrate was driven by "
    "the availability of a broad model catalogue accessible through a unified OpenAI-compatible "
    "endpoint structure, eliminating the need to implement provider-specific request formatting "
    "for each model family."
)

add_heading(doc, "5.5.1  Provider Abstraction Layer", level=3)

add_body(doc,
    "Each model implementation follows a consistent module-level interface convention: a "
    "MODEL_ID string identifying the provider model path, a DISPLAY_NAME string presented "
    "to the frontend, a TIER string declaring the assigned routing tier, a get_api_key() "
    "function retrieving the API key from environment variables, and an async call() "
    "coroutine accepting the prompt text and optional API key override and returning the "
    "complete response text. This interface uniformity means the routing dispatcher can "
    "invoke any model module through the same code path regardless of the underlying "
    "provider, and new model modules can be registered by simply adding a new file to "
    "the llm_providers/ directory and appending it to the MODEL_REGISTRY list."
)

add_body(doc,
    "The base module base.py implements shared HTTP communication logic used by all "
    "provider modules. It manages a persistent httpx.AsyncClient connection pool with "
    "configured timeout parameters and retry policies. By reusing a single async HTTP "
    "client across all requests rather than instantiating a new client per request, "
    "TCP connection overhead is eliminated from the critical path of each inference "
    "call. This optimisation is particularly impactful for streaming sessions where "
    "multiple sequential requests to the same provider endpoint can reuse established "
    "TLS connections."
)

add_heading(doc, "5.5.2  Fallback Chain Implementation", level=3)

add_body(doc,
    "The call_llm() dispatcher function implements a multi-layer fallback strategy to "
    "maintain service availability when a primary model is unavailable due to API "
    "rate limits, service degradation, or missing API key configuration. The attempt "
    "order is constructed as: primary model first, followed by tier-specific fallback "
    "models as defined in TIER_FALLBACKS, followed by the general fallback chain "
    "comprising all remaining registered models in registry order. Duplicate entries "
    "are eliminated while preserving order through a set-based deduplication pass. "
    "For each attempt, if the model module has no configured API key, it is silently "
    "skipped rather than raising an exception, preventing unnecessary error propagation "
    "when some API keys are simply not configured for optional tiers. If all attempts "
    "fail, a structured error response containing the per-model error messages is "
    "returned to the caller rather than an unhandled exception."
)

add_body(doc,
    "Table 5.6 presents the complete model registry with tier assignments, provider APIs, "
    "approximate parameter counts, and fallback chain relationships."
)

# Table 5.6
t56 = add_table_header(doc, ["Model", "Tier", "Provider API", "Approx. Parameters", "Role"])
rows_56 = [
    ("Nemotron Mini 4B",    "Tier 0", "NVIDIA NIM",  "4B",   "Primary"),
    ("Gemma 3n E4B",        "Tier 0", "NVIDIA NIM",  "4B",   "Fallback"),
    ("Nemotron Nano 9B v2", "Tier 1", "NVIDIA NIM",  "9B",   "Primary"),
    ("Nemotron Nano 30B",   "Tier 2", "NVIDIA NIM",  "30B",  "Primary"),
    ("Mistral Medium 3.5",  "Tier 3", "Mistral API", "119B", "Primary"),
    ("Nemotron Super 120B", "Tier 3", "NVIDIA NIM",  "120B", "Fallback"),
    ("Qwen3 Coder 480B",    "Tier 4", "NVIDIA NIM",  "480B", "Primary"),
    ("Qwen3.5 397B",        "Tier 4", "NVIDIA NIM",  "397B", "Fallback"),
]
for r in rows_56:
    add_row(t56, list(r))
doc.add_paragraph()
add_caption(doc, "Table 5.6: Complete LLM Provider Registry with Tier, API, and Role Assignments")

add_body(doc,
    "Figure 5.1 illustrates the implementation architecture of the routing infrastructure "
    "showing provider abstraction, tier assignment, fallback chaining, and asynchronous "
    "communication flow."
)
add_caption(doc,
    "Figure 5.1: Implementation architecture of the routing infrastructure showing provider "
    "abstraction, tier assignment, fallback chaining, and asynchronous communication flow. "
    "[Insert screenshot: CAM.png — routing infrastructure diagram]"
)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: SSE STREAMING IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "5.6  Implementation of SSE Streaming", level=2)

add_body(doc,
    "Real-time streaming was implemented using Server-Sent Events delivered through FastAPI's "
    "StreamingResponse with media type text/event-stream. Unlike WebSocket-based bidirectional "
    "communication, SSE provides a simpler unidirectional channel from server to client that "
    "is natively supported by modern browsers, reconnects automatically after disconnection, "
    "and does not require a persistent TCP connection upgrade handshake. For the use case of "
    "streaming LLM output, SSE is a more appropriate protocol than WebSockets because all "
    "data flows in one direction from the server to the client."
)

add_heading(doc, "5.6.1  Streaming Event Pipeline", level=3)

add_body(doc,
    "The event_generator() coroutine within the /v1/query/stream endpoint implements a "
    "five-stage streaming pipeline executed sequentially within a single async generator "
    "function. In Stage 1, the cognitive analysis pipeline executes synchronously against "
    "the cached scorer, completing in a few milliseconds for most prompts. The tier "
    "assignment, budget score, cognitive profile dictionary, and routing reason are "
    "assembled into a JSON metadata payload and transmitted immediately as a 'meta' event. "
    "This event reaches the frontend before any LLM inference begins, allowing the interface "
    "to display the routing decision, model assignment, and cognitive dimension breakdown "
    "while the user waits for the response to stream in."
)

add_body(doc,
    "In Stage 2, the call_llm() dispatcher is invoked with the resolved tier and prompt. "
    "The complete response text is returned as a Python string after the inference provider "
    "completes processing. In Stage 3, the response text is segmented into chunks of "
    "approximately 32 characters, yielding each chunk as a 'token' event with a 10-millisecond "
    "artificial delay between chunks introduced through asyncio.sleep(0.01). This chunking "
    "strategy simulates the experience of a genuinely streaming inference pipeline and "
    "significantly improves the perceived responsiveness of the interface by showing progressive "
    "content arrival rather than a single delayed response flash."
)

add_body(doc,
    "Stage 4 computes final telemetry metrics including end-to-end latency in milliseconds, "
    "total token usage estimated from word counts using the standard 0.75 words-per-token "
    "ratio, and tokens saved computed as the delta between the actual token consumption "
    "and the hypothetical cost of processing the same interaction through a GPT-4o class "
    "model at a 1.3x token overhead factor. Stage 5 persists the complete query record to "
    "PostgreSQL and transmits a 'done' event containing the final telemetry and the database "
    "record identifier, which the frontend uses to enable individual query deletion from "
    "the history panel."
)

add_body(doc,
    "Table 5.7 describes each SSE event type, its timing within the pipeline, and its "
    "payload structure."
)

# Table 5.7
t57 = add_table_header(doc, ["Event Type", "Timing", "Payload Fields", "Frontend Effect"])
rows_57 = [
    ("meta",  "Before LLM inference", "tier_assigned, model_used, budget_score, cognitive_profile, routing_reason",
     "Displays tier badge, model name, dimension chart, routing reason"),
    ("token", "During chunked delivery", "text (32-char chunk)",
     "Progressively appends to response display buffer"),
    ("done",  "After full response", "latency_ms, tokens_used, tokens_saved, model_used, record_id",
     "Renders telemetry footer, enables history delete"),
    ("error", "On exception", "error (message string)",
     "Displays error notification, halts streaming"),
]
for r in rows_57:
    add_row(t57, list(r))
doc.add_paragraph()
add_caption(doc, "Table 5.7: SSE Event Types, Timing, Payload Structure, and Frontend Effects")

add_body(doc,
    "The StreamingResponse headers were configured with Cache-Control: no-cache, Connection: "
    "keep-alive, and X-Accel-Buffering: no to prevent proxy servers and load balancers from "
    "buffering the event stream, which would otherwise cause all chunks to arrive "
    "simultaneously at the client after the inference completes, defeating the purpose "
    "of streaming delivery."
)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: DATABASE AND PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "5.7  Database and Persistence Implementation", level=2)

add_body(doc,
    "The persistence layer was implemented using PostgreSQL 14 as the relational database "
    "engine, SQLAlchemy 2.0 operating in async mode as the ORM layer, and the asyncpg "
    "driver providing high-performance native PostgreSQL protocol communication. PostgreSQL "
    "was selected over SQLite for its ACID compliance under concurrent write loads, native "
    "JSON column support required for storing variable-structure cognitive profiles, and "
    "availability of efficient composite index types needed for time-range and user-scoped "
    "query patterns."
)

add_heading(doc, "5.7.1  Relational Schema Design", level=3)

add_body(doc,
    "Three primary ORM models were implemented in db_models.py. The User model stores "
    "account credentials as UUID primary keys with unique indexed constraints on username "
    "and email columns, an optional per-user API key override field allowing users to supply "
    "their own LLM provider keys, an is_active boolean for account deactivation support, "
    "and a last_login timestamp updated on each successful authentication."
)

add_body(doc,
    "The UserSession model implements token-based session management. Each session record "
    "stores a 64-character cryptographically random session token with a unique index, "
    "creation and expiry timestamps, client IP address, and user agent string for audit "
    "purposes. The is_valid property computes session validity at access time by checking "
    "both the is_active flag and comparing the current UTC time against the expires_at "
    "timestamp. Sessions are cascade-deleted when the associated user record is deleted, "
    "preventing orphaned session records."
)

add_body(doc,
    "The QueryRecord model is the telemetry store for every prompt processed by the system. "
    "Each record captures the complete lifecycle of a query including the original prompt "
    "text, the generated response text, the model used for inference, the tier assigned, "
    "the integer budget score, floating-point token consumption and savings estimates, "
    "end-to-end latency in milliseconds, the full cognitive profile dictionary stored as "
    "a native PostgreSQL JSON column, the human-readable routing reason string, and the "
    "classified task type. Three composite indexes on (user_id, created_at), (tier_assigned), "
    "and (created_at) were implemented to accelerate the most frequent query patterns: "
    "per-user history retrieval, tier distribution aggregation, and time-range analytics."
)

add_body(doc,
    "Table 5.8 presents the complete QueryRecord schema with column types and purposes."
)

# Table 5.8
t58 = add_table_header(doc, ["Column", "Type", "Description"])
rows_58 = [
    ("id",               "UUID (PK)",   "Unique identifier, used by frontend for history delete"),
    ("user_id",          "UUID (FK)",   "Foreign key to users.id, nullable for anonymous queries"),
    ("session_id",       "UUID (FK)",   "Foreign key to sessions.id, nullable"),
    ("prompt",           "TEXT",        "Full prompt text submitted by the user"),
    ("response",         "TEXT",        "Complete response text returned by the LLM"),
    ("model_used",       "VARCHAR(100)","Display name of the model that generated the response"),
    ("tier_assigned",    "VARCHAR(20)", "Tier label (Tier 0 through Tier 4)"),
    ("budget_score",     "INTEGER",     "Scalar cognitive budget score 1–100"),
    ("tokens_used",      "FLOAT",       "Estimated total token consumption"),
    ("tokens_saved",     "FLOAT",       "Estimated savings vs. GPT-4o class model"),
    ("latency_ms",       "FLOAT",       "End-to-end request-response latency in milliseconds"),
    ("cognitive_profile","JSON",        "Full CognitiveProfile dict with all 6 dimension scores"),
    ("routing_reason",   "TEXT",        "Human-readable routing decision explanation"),
    ("task_type",        "VARCHAR(30)", "Classified TaskType enum value"),
    ("created_at",       "TIMESTAMPTZ", "UTC timestamp of query creation (auto-set)"),
]
for r in rows_58:
    add_row(t58, list(r))
doc.add_paragraph()
add_caption(doc, "Table 5.8: QueryRecord Schema — Column Types and Descriptions")

add_heading(doc, "5.7.2  Asynchronous Database Operations", level=3)

add_body(doc,
    "All database operations were implemented as async coroutines using SQLAlchemy's "
    "AsyncSession and AsyncEngine. The database.py module initialises a single global "
    "AsyncEngine instance using the asyncpg dialect connection string, with connection "
    "pool parameters configured to maintain a minimum of 5 idle connections and a maximum "
    "of 20 concurrent connections. This pool sizing prevents connection exhaustion under "
    "concurrent request load while avoiding over-allocation of database server resources "
    "during idle periods."
)

add_body(doc,
    "Database sessions are injected into FastAPI route handlers through a "
    "get_db() async generator dependency that creates a fresh AsyncSession for each "
    "request, yields it to the route handler, and commits or rolls back the transaction "
    "upon completion. The use of SQLAlchemy's flush() rather than commit() within route "
    "handlers allows the ORM to resolve primary keys for newly created records (such as "
    "the QueryRecord UUID) before the session is committed, enabling the record identifier "
    "to be included in the SSE 'done' event payload without requiring a separate post-commit "
    "database read."
)

add_body(doc,
    "Figure 5.2 illustrates the database implementation architecture showing asynchronous "
    "persistence, telemetry storage, and relational schema organisation."
)
add_caption(doc,
    "Figure 5.2: Database implementation architecture showing asynchronous persistence, "
    "telemetry storage, and relational schema organisation. "
    "[Insert screenshot: database.png — database architecture diagram]"
)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: INTEGRATED SYSTEM DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "5.8  Integrated System Deployment Methodology", level=2)

add_body(doc,
    "The final implementation integrates all six subsystems — frontend, backend API, "
    "cognitive engine, thought budget allocator, routing infrastructure, and persistence layer — "
    "into a unified orchestration ecosystem operating through asynchronous cognitive routing "
    "pipelines. The complete system is deployable as a single process by running the FastAPI "
    "application with an ASGI server such as Uvicorn, which serves both the React frontend "
    "from the compiled assets directory and the complete API surface from the same port."
)

add_heading(doc, "5.8.1  End-to-End Request Execution Flow", level=3)

add_body(doc,
    "When a user submits a prompt through the frontend interface, the following sequence "
    "of operations executes within the integrated system. The React PromptInput component "
    "initiates a POST request to /v1/query/stream, opening an SSE connection. The FastAPI "
    "backend receives the request and invokes the _cached_profile() function, which checks "
    "the in-memory score cache for the prompt's MD5 hash. On a cache miss, the FeatureExtractor "
    "processes the prompt text through all seven feature groups, generating a 45-element "
    "feature vector. The configured scorer strategy then consumes this vector and returns a "
    "CognitiveProfile with six dimension scores and a task type classification."
)

add_body(doc,
    "The profile_to_budget_score() function in routing.py converts the profile into a "
    "scalar budget score using the hybrid aggregation formula. The score_to_tier() function "
    "maps the budget score to a tier label through boundary threshold matching. The "
    "get_tier_model_info() function resolves the primary model identifier and display name "
    "for the assigned tier. The generate_routing_reason() function constructs a "
    "human-readable explanation of the routing decision incorporating the dominant dimension, "
    "contributing factors, task type, confidence level, and model assignment. The complete "
    "metadata payload is immediately transmitted to the frontend as the initial 'meta' "
    "SSE event, completing this cognitive analysis phase in approximately 2 to 15 milliseconds."
)

add_body(doc,
    "The call_llm() dispatcher then invokes the primary model for the assigned tier through "
    "the persistent HTTP connection pool. If the primary model responds successfully, the "
    "response text is chunked and streamed to the frontend through sequential 'token' events. "
    "If the primary model fails for any reason, the fallback chain is attempted before "
    "an error is reported. After streaming completes, a QueryRecord is constructed with "
    "all telemetry fields and persisted to PostgreSQL through the async session, and "
    "the 'done' event is transmitted to conclude the streaming session."
)

add_heading(doc, "5.8.2  Deployment Configuration", level=3)

add_body(doc,
    "Environment configuration is managed through a .env file loaded at application startup "
    "using python-dotenv before any environment variable reads occur. This file stores API "
    "keys for all configured LLM providers, the PostgreSQL connection string, the selected "
    "scorer mode, and optional JWT secret keys for enhanced session security. A .env.example "
    "file is committed to the repository documenting all required and optional configuration "
    "keys without exposing production credentials. The requirements.txt file pins all direct "
    "dependencies including FastAPI, SQLAlchemy, asyncpg, httpx, python-dotenv, and "
    "transformers to ensure reproducible deployment environments."
)

add_body(doc,
    "Figure 5.3 illustrates the complete integrated request flow from prompt submission "
    "through cognitive analysis, budget scoring, tier routing, LLM inference, streaming "
    "delivery, and telemetry persistence."
)
add_caption(doc,
    "Figure 5.3: Complete integrated system execution flow showing prompt ingestion, "
    "cognitive analysis, budget scoring, dynamic tier routing, real-time SSE streaming "
    "response delivery, and asynchronous telemetry persistence. "
    "[Insert screenshot of the CORA frontend in operation showing real-time routing metadata display]"
)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: PROMPT OPTIMISER IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "5.9  Implementation of the Prompt Optimiser", level=2)

add_body(doc,
    "The Prompt Optimiser was implemented as an auxiliary feature accessible through the "
    "/v1/optimize endpoint and the PromptOptimizer frontend component. Its purpose is to "
    "demonstrate the economic efficiency gains achievable by rewriting verbose or padded "
    "prompts into leaner, more directive formulations before they are submitted for "
    "cognitive routing."
)

add_body(doc,
    "The implementation leverages a remote LLM API accessed through the prompt_optimizer.py "
    "module in llm_providers/. The optimiser constructs a structured system prompt instructing "
    "the model to rewrite the submitted prompt for maximum token efficiency while preserving "
    "complete semantic intent. The rewritten prompt is returned as a plain text string. "
    "The backend then scores both the original and optimised prompts through the cognitive "
    "engine, computes tier assignments and token metrics for each, and returns a "
    "PromptOptimizeResponse containing paired PromptMetrics objects for side-by-side "
    "comparison. Both scoring operations are executed concurrently using asyncio where "
    "possible to minimise total response latency."
)

add_body(doc,
    "The frontend renders the comparison as a two-panel display showing the original and "
    "optimised prompts alongside their respective token counts, budget scores, tier "
    "assignments, and estimated token savings. This feature serves both as a practical "
    "efficiency tool for users and as a transparency mechanism demonstrating how prompt "
    "phrasing affects cognitive routing decisions."
)

doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: IMPLEMENTATION SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "5.10  Implementation Summary", level=2)

add_body(doc,
    "The CORA implementation successfully realised all architectural objectives established "
    "during the design phase. The cognitive analysis pipeline executes in 2 to 15 milliseconds "
    "for the rule-based scorer with MD5-cached repeat queries resolving in microseconds, "
    "ensuring that routing decisions add negligible overhead to the total inference latency. "
    "The five-tier routing infrastructure with per-tier fallback chains provides robust "
    "service continuity across eight model endpoints spanning parameter counts from 4 billion "
    "to 480 billion. The SSE streaming architecture delivers the first cognitive metadata "
    "to the frontend within milliseconds of request receipt, providing a perceived "
    "responsiveness that conceals the actual inference latency from the user."
)

add_body(doc,
    "The persistence layer captures complete telemetry for every query, enabling the "
    "Dashboard and Statistics Panels to provide meaningful analytics on routing distribution, "
    "token savings, and cognitive complexity trends across user sessions. The modular "
    "package structure of the cognitive_module/ and llm_providers/ packages ensures that "
    "new scorer strategies and new LLM providers can be integrated independently without "
    "touching the core routing or streaming infrastructure."
)

add_body(doc,
    "This implementation methodology transforms CORA into a scalable, cognition-aware "
    "middleware infrastructure capable of dynamically balancing computational efficiency, "
    "inference quality, routing transparency, and operational cost across heterogeneous "
    "Large Language Model ecosystems. The resulting system demonstrates how engineered "
    "cognitive orchestration combined with adaptive routing, multi-layer provider fallback, "
    "and asynchronous streaming architectures can significantly improve both the economic "
    "efficiency and the scalability of modern AI inference systems while preserving the "
    "interpretability required for production deployment."
)

# ── Save ──────────────────────────────────────────────────────────────────────
doc.save(OUTPUT_PATH)
print(f"[DONE] Document saved to:\n  {OUTPUT_PATH}")
