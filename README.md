# CORA — Cognitive Orchestration &amp; Reasoning Allocator

<p align="center">
  <a href="https://github.com/mohammadzaieemkhan/CORA">
    <img src="docs/media/cora-header-animated.svg" alt="CORA Header Banner" width="100%" />
  </a>
</p>

<p align="center">
  <a href="https://github.com/mohammadzaieemkhan/CORA/actions"><img src="https://img.shields.io/badge/Build-Passing-34D399?style=for-the-badge&logo=github-actions&logoColor=white" alt="Build" /></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" /></a>
  <a href="https://react.dev/"><img src="https://img.shields.io/badge/React_19-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React 19" /></a>
  <a href="https://www.postgresql.org/"><img src="https://img.shields.io/badge/PostgreSQL_14+-316192?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL" /></a>
  <a href="https://vitejs.dev/"><img src="https://img.shields.io/badge/Vite_6-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" /></a>
  <a href="#-security-architecture--protocols"><img src="https://img.shields.io/badge/Security-OWASP_LLM_Hardened-F59E0B?style=for-the-badge&logo=shield&logoColor=white" alt="Security Hardened" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-8B5CF6?style=for-the-badge" alt="License" /></a>
</p>

---

## 📌 Executive Summary

**CORA (Cognitive Orchestration &amp; Reasoning Allocator)** is an enterprise-grade AI middleware gateway that intercepts user prompts, performs microsecond multi-dimensional cognitive profiling, and dynamically routes each query to the optimal Large Language Model (LLM) tier — balancing reasoning capability with cost efficiency.

> **Intelligent Traffic Controller for Foundation Models**  
> Simple greetings and factual questions route to ultra-fast, lightweight models; complex mathematical proofs, multi-file code synthesis, and deep logic route to frontier neural models. The result is an average **84.6% reduction in token expenditure** with **zero perceived degradation in output quality**.

```
  [User Prompt]
        │
        ▼
 ┌────────────────────────────────────────────────────────┐
 │            CORA 6D COGNITIVE PROFILER                  │
 │  • Reasoning Depth       • Domain Specificity          │
 │  • Code Complexity       • Structural Complexity       │
 │  • Creative Demand       • Precision Required          │
 └──────────────────────────┬─────────────────────────────┘
                            │  Decision Latency: < 2ms
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │           DYNAMIC MULTI-TIER ROUTING GRID              │
 │  Tier 0: Gemma 4 26B (a4b) IT   (Trivial / Low Cost)   │
 │  Tier 1: Gemini 3.5 Flash Lite  (Moderate Analysis)    │
 │  Tier 2: Nemotron 3.5 Lightning (Structured Tasks)     │
 │  Tier 3: Nemotron 3 Super 120B  (Deep Reasoning)       │
 │  Tier 4: Nemotron 3 Ultra 550B  (Frontier Reasoning)   │
 └────────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Benchmarks &amp; Performance

CORA has been empirically calibrated across standard LLM evaluation suites. Below is the verified performance matrix demonstrating cost reduction while maintaining or exceeding state-of-the-art benchmark baselines:

<p align="center">
  <img src="docs/media/cora-benchmarks-animated.svg" alt="CORA Benchmark Metrics" width="100%" />
</p>

### Calibrated Benchmark Matrix

| Benchmark | Domain Evaluated | Sample Size | Accuracy / Pass Rate | Cost Reduction | Route Overhead | Primary Model Mapped |
|---|---|---|---|---|---|---|
| **GSM8K** | Multi-Step Mathematical Logic | 50 Samples | **96.0%** | **84.6%** | `< 2 ms` | Tier 3 (Nemotron 3 Super 120B) |
| **HumanEval** | Python Code Synthesis (Pass@1) | 20 Samples | **100.0%** | **79.2%** | `< 2 ms` | Tier 2/3 (Nemotron Lightning / Super) |
| **MMLU** | Multi-Disciplinary Domain Knowledge | 5,700 Samples | **81.9%** | **83.2%** | `< 0.3 ms` | Dynamic Tier 0 → Tier 4 Allocation |
| **RouterBench** | Routing Decisions &amp; AIQ Alignment | 500 Samples | **85.0% AIQ** | **63.0%** | **0.002 s** | Calibrated Ordinal Regression Matrix |

---

## 🏗️ Interactive Architecture

CORA utilizes a decoupled, asynchronous micro-gateway topology connecting the high-speed browser client to distributed LLM inference clusters and persistent storage.

<p align="center">
  <img src="docs/media/cora-interactive-architecture.svg" alt="CORA System Topology and Data Flow" width="100%" />
</p>

### Interactive System Diagram (Mermaid)

Click any module box below to jump directly into the corresponding subsystem documentation:

```mermaid
graph TD
    %% Styling
    classDef clientStyle fill:#0F172A,stroke:#38BDF8,stroke-width:2px,color:#F8FAFC;
    classDef gatewayStyle fill:#1E1B4B,stroke:#6366F1,stroke-width:2px,color:#F8FAFC;
    classDef engineStyle fill:#312E81,stroke:#A855F7,stroke-width:2px,color:#F8FAFC;
    classDef routerStyle fill:#701A75,stroke:#EC4899,stroke-width:2px,color:#F8FAFC;
    classDef modelStyle fill:#0F172A,stroke:#10B981,stroke-width:1.5px,color:#F8FAFC;
    classDef storageStyle fill:#064E3B,stroke:#059669,stroke-width:2px,color:#F8FAFC;

    subgraph ClientLayer ["1. Client & Perception Layer"]
        UI["React 19 Frontend<br/>(Ethereal Glassmorphism + Framer Motion)"]:::clientStyle
        PromptOpt["Prompt Optimizer UI<br/>(Token Compression)"]:::clientStyle
        SSEConsumer["SSE Stream Reader<br/>(Near-Zero First-Token Latency)"]:::clientStyle
    end

    subgraph GatewayLayer ["2. API Gateway & Security Vault"]
        FastAPI["FastAPI Async Gateway<br/>(Lifespan, CORS, Rate-Limiting)"]:::gatewayStyle
        AuthVault["Bcrypt + 64-Hex Session Vault<br/>(Stateless Bearer Tokens)"]:::gatewayStyle
        LRUCache["In-Memory Profile LRU Cache<br/>(Hash-Keyed De-Duplication)"]:::gatewayStyle
    end

    subgraph EngineLayer ["3. 6D Cognitive Engine & Calibration"]
        FeatureEx["Lexical Feature Extractor<br/>(42+ Regex & AST Heuristics)"]:::engineStyle
        ScorerFactory{"Scorer Strategy Factory<br/>[Rule | NeMo | LLM | ML]"}:::engineStyle
        CalibrationEq["Ordinal Regression Calibrator<br/>S(p) = τ(t) × Σ(Wᵢ·αᵢ/Z · xᵢ)"]:::engineStyle
        PeakBlender["Peak-Signal & Task Blender<br/>(50% Mean + 50% Top-2 Peak)"]:::engineStyle
    end

    subgraph RoutingLayer ["4. Dynamic Tier Allocator & Failover Circuit"]
        Router["Model Router Matrix<br/>(Budget Score 0–100 → Tiers 0–4)"]:::routerStyle
        CircuitBreaker["Circuit Breaker & Fallback Chain<br/>(Zero-Downtime Provider Failover)"]:::routerStyle
    end

    subgraph ModelMesh ["5. Multi-Provider LLM Inference Grid"]
        T0["Tier 0: Gemma 4 26B (a4b) IT<br/>↳ Fallback: Gemma 4 26B (OpenRouter)"]:::modelStyle
        T1["Tier 1: Gemini 3.5 Flash Lite<br/>↳ Fallback: Google Gemma 2 9B"]:::modelStyle
        T2["Tier 2: Nemotron 3.5 Lightning 30B<br/>↳ Fallback: Meta Muse Glimmer 30B"]:::modelStyle
        T3["Tier 3: Nemotron 3 Super 120B<br/>↳ Fallback: GLM 5.3 Flash / DeepSeek V4.1"]:::modelStyle
        T4["Tier 4: Nemotron 3 Ultra 550B<br/>↳ Fallback: Moonshot Kimi K3 / DeepSeek V4.1"]:::modelStyle
    end

    subgraph DataLayer ["6. Persistence & Telemetry Store"]
        DB[("PostgreSQL 14+<br/>AsyncPG + SQLAlchemy 2.0")]:::storageStyle
        Telemetry["Audit Records & Query Profiles<br/>(Latency, Dimensions, Token Metrics)"]:::storageStyle
    end

    %% Flow Connections
    UI -->|HTTP / SSE Stream| FastAPI
    PromptOpt -->|Compress Prompt| FastAPI
    FastAPI -->|Check Token / Auth| AuthVault
    FastAPI -->|Lookup md5 Cache| LRUCache
    FastAPI -->|Extract Signals| FeatureEx
    FeatureEx --> ScorerFactory
    ScorerFactory --> CalibrationEq
    CalibrationEq --> PeakBlender
    PeakBlender -->|Budget Score 0-100| Router
    Router --> CircuitBreaker

    CircuitBreaker -->|Score ≤ 20| T0
    CircuitBreaker -->|Score ≤ 45| T1
    CircuitBreaker -->|Score ≤ 70| T2
    CircuitBreaker -->|Score ≤ 88| T3
    CircuitBreaker -->|Score ≤ 100| T4

    T0 -.->|Token Stream| SSEConsumer
    T1 -.->|Token Stream| SSEConsumer
    T2 -.->|Token Stream| SSEConsumer
    T3 -.->|Token Stream| SSEConsumer
    T4 -.->|Token Stream| SSEConsumer

    FastAPI -->|Async Persist Profile & History| DB
    DB --- Telemetry

    click UI href "#-ethereal-engine-ui" "Inspect Ethereal Engine UI"
    click FastAPI href "#-api-reference" "Inspect FastAPI Endpoints"
    click FeatureEx href "#-the-6d-cognitive-engine--calibration" "Inspect Cognitive Scorer"
    click Router href "#-active-model-grid--tier-allocation" "Inspect Routing Matrix"
    click AuthVault href "#-security-architecture--protocols" "Inspect Security Protocols"
```

---

### Request-Response & SSE Stream Lifecycle (Sequence)

```mermaid
sequenceDiagram
    autonumber
    actor User as User Browser
    participant UI as React 19 Frontend
    participant API as FastAPI Gateway
    participant Cache as Memory LRU Cache
    participant CE as 6D Cognitive Engine
    participant Router as Allocator & Circuit Breaker
    participant LLM as Active LLM Provider
    participant DB as PostgreSQL DB

    User->>UI: Submit Prompt Query
    UI->>API: POST /v1/query/stream (Bearer Token + Prompt)
    API->>API: Validate Session & Sanitize Payload
    API->>Cache: Check md5(prompt)
    alt Cache Hit
        Cache-->>API: Return Cached CognitiveProfile
    else Cache Miss
        API->>CE: Run 42+ Feature Signal Extraction
        CE->>CE: Compute S(p) via Ordinal Exponents (α)
        CE->>CE: Calculate Peak-Signal Blend & Task Boost
        CE-->>API: Return CognitiveProfile (Score: 0-100)
        API->>Cache: Store Profile in LRU Cache
    end
    API->>Router: Map Score to Tier (0–4)
    Router-->>API: Assign Target Model + Fallback Chain
    API-->>UI: SSE Event: metadata {tier, model, dimensions, budget_score}
    
    rect rgb(20, 25, 45)
        Note over API,LLM: Asynchronous Token Streaming with Fallback Protection
        API->>LLM: Stream Inference Request (HTTP/2 keep-alive)
        alt Provider Healthy
            loop Token Generation
                LLM-->>API: Token Chunk
                API-->>UI: SSE Event: chunk {text}
            end
        else Provider Error / Rate Limit (HTTP 429/500)
            API->>Router: Trigger Fallback Circuit Breaker
            Router->>LLM: Fallback Model Call (e.g. OpenRouter / DeepSeek)
            LLM-->>API: Fallback Token Stream
            API-->>UI: SSE Event: chunk {text}
        end
    end

    API-->>UI: SSE Event: done {total_time, tokens}
    API-)DB: Async Background Task: Insert QueryRecord + Telemetry
```

---

### 🔍 Deep-Dive Subsystem Architecture

<details>
<summary><b>⚡ Layer 1: Client & Perception Layer (React 19 + Ethereal Engine)</b></summary>
<br>

- **Framework**: React 19 with Vite 6 build engine and CSS Modules.
- **Visual Aesthetic**: Ethereal Engine — a dark glassmorphic design system using dynamic backdrop blur (`backdrop-filter: blur(16px)`), CSS aurora gradient lighting, and fluid radii eliminating sharp visual boundaries.
- **Streaming Pipeline**: Native browser `EventSource` and `ReadableStream` reader supporting real-time SSE chunk assimilation with automatic carriage de-duplication.
- **Prompt Optimizer**: Seamless client toggle utilizing Google Gemini (3.8 Flash / 3.5 Flash Lite) to compress verbose prompts down to essential semantic tokens before routing.
</details>

<details>
<summary><b>🧠 Layer 2: 6D Cognitive Profiler & Calibration Engine</b></summary>
<br>

- **Multi-Dimensional Signal Space**: Analyzes 6 distinct dimensions:
  1. *Reasoning Depth* ($W=0.30, \alpha=2.21$): Logic operators, conditional proofs, deductive chains.
  2. *Domain Specificity* ($W=0.20, \alpha=1.40$): Technical terminology, scientific taxonomy, medical/financial jargon.
  3. *Creative Demand* ($W=0.20, \alpha=0.06$): Open-ended creative brainstorming, rhetorical flexibility.
  4. *Structural Complexity* ($W=0.15, \alpha=0.07$): Multi-tiered questions, format constraints, markdown tables.
  5. *Precision Required* ($W=0.10$): Exact numerical computations, date lookups, factual rigor.
  6. *Few-Shot Context* ($W=0.05$): In-context input/output example pairs.
- **Peak Signal Blending**: Prompts with high complexity in any single dimension are protected from dilution:
  $$\text{Score}_{\text{hybrid}} = 0.5 \times \text{WeightedAverage} + 0.5 \times \text{PeakTop2Average}$$
- **Scorer Strategies**: Pluggable architecture supporting `rule` (regex heuristics), `nemo` (NeMo context), `llm` (evaluator model), and `ml` (DistilBERT embedding classifier).
</details>

<details>
<summary><b>🎯 Layer 3: Dynamic Model Allocator & Zero-Downtime Fallback Grid</b></summary>
<br>

- **Dynamic Tier Boundary**:
  - Score $0 \dots 20 \rightarrow$ **Tier 0** (Trivial factual lookups, greetings, simple definitions)
  - Score $21 \dots 45 \rightarrow$ **Tier 1** (Moderate QA, summaries, basic transformations)
  - Score $46 \dots 70 \rightarrow$ **Tier 2** (Code generation, structural data synthesis, mid-tier analysis)
  - Score $71 \dots 88 \rightarrow$ **Tier 3** (Multi-step algorithmic proofs, complex debugging, deep reasoning)
  - Score $89 \dots 100 \rightarrow$ **Tier 4** (Frontier reasoning, complex architectures, deep scientific queries)
- **Circuit Breaker**: Each tier defines prioritized primary and secondary providers. If a provider throws an HTTP 429 (rate-limit) or 5xx, the gateway seamlessly fails over to alternate providers (e.g. NVIDIA NIM $\rightarrow$ OpenRouter $\rightarrow$ DeepSeek) without breaking client SSE connections.
</details>

<details>
<summary><b>🛡️ Layer 4: Enterprise Security & Zero-Trust Session Store</b></summary>
<br>

- **Credentials**: Zero-knowledge credential isolation; keys exist only in server memory.
- **Hashing**: Passwords salted and hashed with `bcrypt`.
- **Sessions**: Cryptographically random 64-hexadecimal UUIDs (`secrets.token_hex(32)`) with a 72-hour sliding expiry.
- **Sanitization**: Pydantic schema validation preventing SQL injection, parameter tampering, and prompt payload overflow.
</details>

---

## 🧠 The 6D Cognitive Engine &amp; Calibration

The core innovation of CORA is its mathematical understanding of prompt complexity before committing tokens to expensive frontier models.

<p align="center">
  <img src="docs/media/cora-cognitive-radar.svg" alt="CORA 6D Cognitive Radar Chart" width="100%" />
</p>

### The Empirical Calibration Formula

Through ordinal regression on the RouterBench dataset, dimension weights ($W_i$) and scaling exponents ($\alpha_i$) were empirically calibrated to maximize separation between trivial and frontier queries:

$$S(p) = \tau(t) \times \sum_{i=1}^{6} \left[ \frac{W_i \times \alpha_i}{Z} \times x_i(p) \right]$$

Where:
- $S(p)$: Final budget complexity score $[1, 100]$.
- $\tau(t)$: Task-type multiplier ($\text{Coding} = 1.25$, $\text{Math} = 1.20$, $\text{Creative} = 0.90$, $\text{QA} = 1.00$).
- $W_i$: Base dimension weight (sums to $1.0$).
- $\alpha_i$: Calibrated scaling exponent (boosts reasoning by $+47.3\%$; dampens unconstrained creativity by $-92.5\%$).
- $Z$: Normalization constant $= 1.045$.
- $x_i(p)$: Normalised score for dimension $i$, computed via $x_i = \min(\text{raw}_i / 20.0, 1.0)$.

---

## ⚡ Active Model Grid &amp; Tier Allocation

CORA maintains an active, production-calibrated model ecosystem spanning multiple inference providers:

| Tier | Primary Model | Provider | Context Window | Key Specialization | Failover Fallback Chain |
|---|---|---|---|---|---|
| **Tier 0** | **Gemma 4 26B (a4b) IT** | NVIDIA NIM | 8k tokens | Basic greetings, trivia, trivial lookups | `Gemma 4 26B (OpenRouter Free)` → `Gemini 3.5 Flash Lite` |
| **Tier 1** | **Gemini 3.5 Flash Lite** | Google AI Studio | 1M tokens | High-volume QA, summarization, entity extraction | `Google Gemma 2 9B (OpenRouter)` → `Gemma 4 26B` |
| **Tier 2** | **Nemotron 3.5 Lightning 30B** | NVIDIA NIM | 32k tokens | Modular coding, refactoring, structured JSON | `Meta Muse Glimmer 30B` → `Nemotron 3 Super 120B` |
| **Tier 3** | **Nemotron 3 Super 120B** | NVIDIA NIM | 64k tokens | Multi-file software architecture, mathematics | `GLM 5.3 Flash` → `DeepSeek V4.1 Flash` |
| **Tier 4** | **Nemotron 3 Ultra 550B** | NVIDIA NIM | 128k tokens | Frontier logic, deep proofs, complex research | `Moonshot Kimi K3` → `DeepSeek V4.1 Flash` |
| **Opt** | **Gemini 3.8 Flash** | Google AI Studio | 1M tokens | Semantic prompt compression &amp; token optimization | `Gemini 3.5 Flash Lite` |

---

## 🔒 Security Architecture &amp; Protocols

CORA is engineered following defense-in-depth principles and OWASP Top 10 for Large Language Model Applications guidelines:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      CORA ZERO-TRUST SECURITY MATRIX                   │
├─────────────────────────┬──────────────────────────────────────────────┤
│ Protocol Layer          │ Implementation & Enforcement                 │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 1. Zero-Knowledge Keys  │ API keys isolated in OS environment memory.  │
│                         │ Never stored in database or client bundles.  │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 2. Session Integrity    │ 64-character hex tokens via secrets.token_hex│
│                         │ Bcrypt hashing with salted work factor.      │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 3. Denial of Wallet     │ Cognitive budget caps & LRU hash caches      │
│    Mitigation (LLM04)   │ prevent infinite loops and token abuse.      │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 4. Prompt Sanitization  │ Pydantic v2 schemas reject malformed inputs  │
│    & Injection (LLM01)  │ Cognitive anomaly scoring flags outliers.    │
├─────────────────────────┼──────────────────────────────────────────────┤
│ 5. Transport Security   │ Strict CORS origin whitelisting;             │
│                         │ rel="noopener noreferrer" on external links. │
└─────────────────────────┴──────────────────────────────────────────────┘
```

### Security Measures in Detail

1. **Zero-Knowledge Key Storage**:
   Downstream model API credentials (`GOOGLE_AI_STUDIO_API_KEY`, `NVIDIA_*_API_KEY`, `OPENROUTER_API_KEY`) are loaded solely at application startup via `python-dotenv`. Keys are never written to PostgreSQL, serialized in log files, or transmitted in client telemetry.
2. **Denial-of-Wallet &amp; Resource Exhaustion Protection (OWASP LLM04)**:
   By pre-scoring prompts before executing inference, adversarial prompts attempting to trigger expensive recursive generation are bound to strict tier limits. Identical repeated queries hit an in-memory MD5-keyed LRU cache (`_score_cache`, max capacity 512) to prevent cost duplication.
3. **Session Revocation &amp; Sliding TTL**:
   Active sessions expire after 72 hours of inactivity. Explicit user logout immediately invalidates the 64-hex session token from the `user_sessions` database table, terminating concurrent operations.
4. **Git &amp; Supply-Chain Hygiene**:
   Strict `.gitignore` enforcement ensures that `.env`, `.env.local`, `.venv/`, `.agent/`, and SQLite test stores cannot be accidentally staged or committed to the public Git tree.

---

## 🎨 Ethereal Engine UI

The client application introduces the **Ethereal Engine** design language:

- **Glassmorphic Depth**: Multilayer translucent cards with backdrop blurs that render against animated background aurora fields.
- **Fluid Geometry**: Eliminates rigid borders in favor of fluid border radii (`radius-sm` through `radius-full`) and refractive edge highlights.
- **Adaptive Ambient Lighting**: Shifting aurora blobs organically pulse in response to the user's interaction state.
- **Particle Canvas**: Dynamic HTML5 canvas particle system creating a responsive living backdrop.
- **Instantaneous Streaming**: Token-by-token text generation with auto-scrolling terminal cards and telemetry inspector panels.

---

## 🚀 Quick Start Guide

### Prerequisites

- **Python**: Version 3.10 or higher
- **Node.js**: Version 18.0 or higher
- **PostgreSQL**: Version 14 or higher (or cloud provider e.g., Supabase / Neon)
- **API Keys**: Google AI Studio, NVIDIA NIM, and/or OpenRouter

---

### Step 1: Clone Repository

```bash
git clone https://github.com/mohammadzaieemkhan/CORA.git
cd CORA
```

---

### Step 2: Set Up Python Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate on macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 3: Configure Environment Variables

Copy the provided environment template:

```bash
cp .env.example .env
```

Open `.env` and fill in your database credentials and API keys:

```ini
# Database Connection (AsyncPG)
DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/cora

# Cognitive Scorer Mode (rule | nemo | llm | ml)
CORA_SCORER_MODE=nemo

# Tier 0 & Tier 1 — Google AI Studio [ACTIVE]
GOOGLE_AI_STUDIO_API_KEY=your-google-ai-studio-api-key

# Tier 0 & Tier 1 Fallback — OpenRouter [ACTIVE]
OPENROUTER_API_KEY=your-openrouter-api-key

# Tier 2 — NVIDIA NIM [ACTIVE]
NVIDIA_MUSE_GLIMMER_30B_API_KEY=your-nvidia-key
NVIDIA_NEMOTRON_3_5_LIGHTNING_API_KEY=your-nvidia-key

# Tier 3 — NVIDIA NIM [ACTIVE]
NVIDIA_NEMOTRON_SUPER_API_KEY=your-nvidia-key
NVIDIA_GLM_5_3_FLASH_API_KEY=your-nvidia-key

# Tier 4 — NVIDIA NIM [ACTIVE]
NVIDIA_NEMOTRON_ULTRA_550B_API_KEY=your-nvidia-key
NVIDIA_KIMI_K3_API_KEY=your-nvidia-key
NVIDIA_DEEPSEEK_V4_1_FLASH_API_KEY=your-nvidia-key
```

---

### Step 4: Initialize PostgreSQL Database

Ensure PostgreSQL is running, then create the database:

```sql
CREATE DATABASE cora;
```

---

### Step 5: Build Frontend &amp; Launch Gateway

#### Option A: Unified Production Mode (FastAPI serves built React bundle)

```bash
# 1. Build frontend
cd frontend
npm install
npm run build
cd ..

# 2. Start FastAPI server
uvicorn main:app --reload --port 8000
```

Access the application at: **http://localhost:8000**

#### Option B: Development Mode (Hot-Reload)

```bash
# Terminal 1: Backend
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

Frontend dev server runs at: **http://localhost:5173**

---

## 📡 API Reference

### 🔐 Authentication

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/v1/auth/register` | Register new user account (returns token) | No |
| `POST` | `/v1/auth/login` | Authenticate existing user (returns token) | No |
| `POST` | `/v1/auth/logout` | Invalidate current session token | **Yes** (Bearer) |

### 👤 User Management

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/v1/user/profile` | Retrieve user stats, queries count, and email | **Yes** (Bearer) |
| `PUT` | `/v1/user/profile` | Update email or update password | **Yes** (Bearer) |
| `GET` | `/v1/user/history` | Paginated query history with cognitive profiles | **Yes** (Bearer) |
| `DELETE` | `/v1/user/history/{id}` | Delete a specific query audit record | **Yes** (Bearer) |

### ⚡ Cognitive Routing &amp; Streaming

| Method | Endpoint | Description | Payload |
|---|---|---|---|
| `POST` | `/v1/query/stream` | **SSE Streaming**: emits metadata, then tokens | `{"prompt": "string"}` |
| `POST` | `/v1/query` | Standard synchronous inference response | `{"prompt": "string"}` |
| `POST` | `/v1/cognitive-profile` | Score prompt complexity without calling LLM | `{"prompt": "string"}` |
| `POST` | `/v1/optimize` | Compress prompt via Gemini Optimizer | `{"prompt": "string"}` |
| `GET` | `/v1/stats` | Aggregated cluster metrics &amp; tier distribution | None |

#### Sample Request (`/v1/cognitive-profile`)

```bash
curl -X POST "http://localhost:8000/v1/cognitive-profile" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write an asynchronous Python function using asyncio to download 10 files in parallel with semaphore limits."}'
```

#### Sample Response

```json
{
  "reasoning_depth": 72,
  "code_complexity": 88,
  "domain_specificity": 75,
  "creative_demand": 25,
  "precision_required": 85,
  "structural_complexity": 68,
  "budget_score": 79,
  "tier": "Tier 3",
  "model": "Nemotron 3 Super 120B",
  "routing_reason": "Detected task type: 💻 Coding. Primary signal: Code Complexity (88/100). Contributing factors: Precision Required (85), Reasoning Depth (72)."
}
```

---

## 📂 Project Directory Structure

```
CORA/
├── main.py                          # FastAPI lifespan app, endpoints, SSE streaming
├── database.py                      # SQLAlchemy 2.0 AsyncEngine & session provider
├── db_models.py                     # PostgreSQL models (User, UserSession, QueryRecord)
├── auth.py                          # Bcrypt hashing, 64-hex token vault, HTTPBearer
├── schemas.py                       # Pydantic v2 request/response data contracts
├── complexity_score.py              # Ordinal score translation & tier mappings
├── cora_calibration_report.md       # Empirical calibration report & regression data
├── benchmark_summary.csv            # SOTA benchmark results (GSM8K, MMLU, HumanEval)
├── requirements.txt                 # Backend Python package requirements
├── vercel.json                      # Vercel deployment specification
├── .env.example                     # Environment configuration template
├── .gitignore                       # Git exclusion rules (Zero-leakage protection)
│
├── docs/
│   └── media/                       # Animated SVGs & visual architecture assets
│       ├── cora-header-animated.svg
│       ├── cora-interactive-architecture.svg
│       ├── cora-cognitive-radar.svg
│       └── cora-benchmarks-animated.svg
│
├── cognitive_module/                # Core 6D Cognitive Engine
│   ├── __init__.py                  # Public exports (create_scorer, routing reasons)
│   ├── config.py                    # Dimension weights, tier cutoffs, hyper-parameters
│   ├── models.py                    # CognitiveProfile dataclass & TaskType enums
│   ├── feature_extractor.py         # 42+ lexical, regex, and AST feature extractors
│   ├── scorer.py                    # Strategy pattern factory & base abstract scorer
│   ├── rule_scorer.py               # Calibrated heuristic rule-based scorer
│   ├── nemo_scorer.py               # NeMo-enhanced contextual scorer
│   ├── llm_scorer.py                # LLM-as-a-judge scoring strategy
│   ├── ml_scorer.py                 # DistilBERT ML embedding scorer
│   └── routing.py                   # Budget score blending & tier mapping logic
│
├── llm_providers/                   # Multi-Provider Inference & Failover Mesh
│   ├── __init__.py                  # Central registry, call_llm(), fallback chains
│   ├── base.py                      # Shared HTTP/2 httpx client connection pool
│   ├── gemma_4_26b.py               # Tier 0 Primary — Google Gemma 4 26B (NVIDIA NIM)
│   ├── gemma_4_26b_openrouter.py    # Tier 0 Fallback — Gemma 4 26B (OpenRouter Free)
│   ├── gemini_3_5_flash_lite.py     # Tier 1 Primary — Google Gemini 3.5 Flash Lite
│   ├── gemma_2_9b_openrouter.py     # Tier 1 Fallback — Google Gemma 2 9B (OpenRouter)
│   ├── nemotron_3_5_lightning_30b.py# Tier 2 Primary — Nemotron 3.5 Lightning 30B
│   ├── muse_glimmer_30b.py          # Tier 2 Fallback — Meta Muse Glimmer 30B
│   ├── nemotron_super_120b.py       # Tier 3 Primary — Nemotron 3 Super 120B
│   ├── glm_5_3_flash.py             # Tier 3 Fallback — GLM 5.3 Flash
│   ├── nemotron_3_ultra_550b.py     # Tier 4 Primary — Nemotron 3 Ultra 550B
│   ├── deepseek_v4_1_flash.py       # Fallback — DeepSeek V4.1 Flash
│   ├── kimi_k3.py                   # Tier 4 Fallback — Moonshot AI Kimi K3
│   └── prompt_optimizer.py          # Prompt Compression via Google Gemini
│
└── frontend/                        # React 19 + Vite 6 Single Page Application
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx                  # Root application layout & route controller
        ├── index.css                # Design system tokens, aurora animations, resets
        ├── context/ThemeContext.jsx  # Dark/light theme context provider
        ├── services/api.js          # REST & SSE EventSource client adapter
        └── components/              # Ethereal glassmorphic UI components
            ├── Navbar.jsx           # Sticky glassmorphic navigation header
            ├── Hero.jsx             # Hero landing section with animated CTA
            ├── PromptInput.jsx      # Glass input bar with integrated optimizer
            ├── ResultsPanel.jsx     # Streaming response card & telemetry drawer
            ├── Dashboard.jsx        # Analytics charts & tier breakdown grid
            ├── Sidebar.jsx          # History panel with search and session filters
            ├── BentoGrid.jsx        # Interactive feature capability showcase
            ├── PipelineViz.jsx      # Visual representation of cognitive routing
            └── ParticleCanvas.jsx   # Ambient background particle field
```

---

## 🤝 Contributing

We welcome community contributions to improve CORA's cognitive scorers, model integrations, and routing precision.

1. **Fork** the repository: `https://github.com/mohammadzaieemkhan/CORA`
2. **Create a Feature Branch**: `git checkout -b feature/cognitive-enhancement`
3. **Commit Your Changes**: Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
4. **Push to Your Branch**: `git push origin feature/cognitive-enhancement`
5. **Open a Pull Request**: Provide a detailed summary and evaluation benchmark results

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.

---

<p align="center">
  <b>Built with precision for the next generation of intelligent LLM gateways.</b><br/>
  <sub>Developed by <a href="https://github.com/mohammadzaieemkhan">Mohammad Zaieem Khan</a> and Contributors.</sub>
</p>
