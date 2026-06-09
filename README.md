# CORA — Cognitive Orchestration & Reasoning Allocator

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" />
  <img src="https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
</p>

**CORA** is an advanced AI Middleware application that intercepts user prompts, performs high-speed multi-dimensional cognitive analysis, and dynamically routes each request to the most cost-efficient and capable Large Language Model — all through a stunning, immersive glassmorphic interface.

> **Think of it as an intelligent traffic controller for LLMs** — simple questions go to lightweight models, complex reasoning gets routed to powerful ones, and you save tokens (and money) on every single query.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **6D Cognitive Profiling** | Every prompt is analysed across Reasoning, Domain, Code, Creativity, Precision, and Structure dimensions |
| **5-Tier LLM Routing** | Automatic model selection from Gemma 3n (Tier 0) to Mistral Large 3 (Tier 4) based on complexity |
| **Real-Time SSE Streaming** | Token-by-token response streaming with near-zero perceived latency |
| **Prompt Optimizer** | AI-powered prompt compression via GPT-OSS-120B for maximum token efficiency |
| **Ethereal Engine UI** | Glassmorphic, aurora-lit React interface with fluid animations and adaptive theming |
| **Full Persistence** | PostgreSQL-backed user accounts, session management, and complete query history |
| **Analytics Dashboard** | Live metrics, tier distribution charts, and cognitive profile breakdowns |
| **Provider Fallback Chain** | Automatic failover across models if the primary provider is unavailable |

---

## 🏗️ Architecture

```mermaid
graph TD
    UI["React Frontend<br/>(Vite + Framer Motion)"] -->|SSE Stream| API["FastAPI Backend"]
    API -->|Prompt| CE["Cognitive Engine"]
    CE -->|Rule-Based / NeMo / LLM Scorer| Profile["Cognitive Profile<br/>(6 dimensions)"]
    Profile --> Router["Tier Assignment<br/>(0-4)"]
    Router -->|Tier 0| Gemma3n["Gemma 3n E4B<br/>(NVIDIA)"]
    Router -->|Tier 1| Nemotron["Mistral Nemotron<br/>(NVIDIA)"]
    Router -->|Tier 2| Magistral["Magistral Small<br/>(NVIDIA)"]
    Router -->|Tier 3| MistralMedium["Mistral Medium 3<br/>(NVIDIA)"]
    Router -->|Tier 4| MistralLarge["Mistral Large 3<br/>(NVIDIA)"]
    API -->|Persist| DB[(PostgreSQL)]
    API -->|Optimize| Optimizer["Optimizer MOdel<br/>(Prompt Optimizer)"]
```

### Technology Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI (async), SQLAlchemy 2.0, asyncpg |
| **Frontend** | React 19, Vite, Framer Motion, CSS Modules |
| **Database** | PostgreSQL 14+ |
| **Auth** | bcrypt password hashing, stateless 64-hex UUID session tokens |
| **LLM Providers** | NVIDIA NIM (Gemma 3n, Mistral Nemotron, Magistral, Mistral Large), OpenAI-compatible endpoints |
| **Design System** | Ethereal Engine — glassmorphism, aurora backgrounds, fluid radii |

---

## 🧠 The Cognitive Engine

The core of CORA is its ability to *understand* the complexity of a prompt before sending it to an LLM. It computes a **Cognitive Profile** across 6 distinct dimensions:

| Dimension | What It Measures |
|---|---|
| **Reasoning Depth** | Multi-step logic, causal chain indicators, deep analytical requirements |
| **Domain Specificity** | Technical jargon density, acronym usage, specialized field knowledge |
| **Code Complexity** | Programming language detection, code blocks, debugging syntax |
| **Creative Demand** | Open-ended generation, brainstorming, stylistic instructions |
| **Precision Required** | Mathematical constraints, factual lookups, exact instructional rigor |
| **Structural Complexity** | Sentence nestedness, multi-part questions, overall prompt length |

### Scorer Modes

CORA supports multiple scoring strategies, selectable via the `CORA_SCORER_MODE` environment variable:

| Mode | Description |
|---|---|
| `rule` | **RuleBasedScorer** — 42+ regex and lexical heuristic features. Fast, deterministic. |
| `nemo` | **NeMoScorer** — Enhanced scorer with deeper contextual analysis. |
| `llm` | **LLMScorer** — Uses an LLM to evaluate prompt complexity. Most accurate, slower. |
| `ml` | **MLScorer** — Custom DistilBERT regression model combining text embeddings with handcrafted features. |

### Tier Assignment

| Tier | Model | Use Case |
|---|---|---|
| Tier 0 | Gemma 3n E4B | Greetings, simple factual lookups, trivial queries |
| Tier 1 | Mistral Nemotron | Moderate factual questions, light analytical tasks |
| Tier 2 | Magistral Small 2506 | Mid-range analysis, creative writing, structured outputs |
| Tier 3 | Mistral Medium 3 | Complex reasoning, multi-step problems |
| Tier 4 | Mistral Large 3 675B | Deep analytical tasks, advanced code generation |

---

## 🎨 Ethereal Engine UI

The frontend implements the **Ethereal Engine** design system — a premium, immersive interface built for AI-native experiences:

- **Glassmorphism** — Translucent surfaces with backdrop blur allow the living aurora background to permeate through every component
- **Aurora Background** — Continuously shifting colour blobs create an organic, breathing atmosphere
- **Fluid Geometry** — All corners use soft radii (`radius-sm` → `radius-full`), eliminating harsh edges
- **No-Line Philosophy** — Borders are replaced with tonal background transitions and refractive edge effects
- **Adaptive Theming** — Seamless light/dark mode transitions that morph the entire visual language
- **Particle Canvas** — Interactive particle system that responds to the ambient environment
- **Framer Motion** — Layout animations, staggered reveals, and smooth state transitions

### Frontend Components

| Component | Purpose |
|---|---|
| `Hero` | Landing page with animated intro and CTA |
| `PromptInput` | Glassmorphic input with glow-on-focus and integrated optimizer |
| `ResultsPanel` | Streaming response display with telemetry metadata |
| `Dashboard` | Analytics grid with tier distribution, savings metrics, and insights |
| `Sidebar` | Collapsible history panel with search and delete |
| `BentoGrid` | Architecture showcase with feature cards |
| `PipelineViz` | Visual flow diagram of the cognitive routing pipeline |
| `Navbar` | Sticky navigation with theme toggle and auth controls |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for frontend)
- **PostgreSQL 14+** (running locally or remotely)

### 1. Clone & Setup

```bash
git clone https://github.com/AakashSharma08/CORA.git
cd CORA

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows
source .venv/bin/activate       # Mac/Linux

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Setup Database

```sql
CREATE DATABASE cora;
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```ini
# Database
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/cora

# Cognitive Scorer Mode (rule | nemo | llm | ml)
CORA_SCORER_MODE=rule

# NVIDIA NIM API Keys (one per model tier)
NVIDIA_GEMMA_API_KEY=your-key
NVIDIA_LLAMA_API_KEY=your-key
NVIDIA_MISTRAL_API_KEY=your-key
NVIDIA_MIXTRAL_API_KEY=your-key

# Google Gemini (optional fallback)
GOOGLE_GEMINI_API_KEY=your-key

# Prompt Optimizer
NVIDIA_GPT_OSS_API_KEY=your-key
```

### 4. Install & Build Frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### 5. Run the Server

```bash
uvicorn main:app --reload --port 8000
```

Visit **http://localhost:8000** to experience CORA.

For frontend development with hot reload:

```bash
cd frontend
npm run dev
```

---

## 📡 API Reference

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/auth/register` | Create a new user account |
| `POST` | `/v1/auth/login` | Authenticate and receive a Bearer token |
| `POST` | `/v1/auth/logout` | Invalidate current session *(Auth required)* |

### User Management

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/user/profile` | Get profile stats and aggregate metrics *(Auth required)* |
| `PUT` | `/v1/user/profile` | Update email or password *(Auth required)* |
| `GET` | `/v1/user/history` | Paginated query history with cognitive profiles *(Auth required)* |
| `DELETE` | `/v1/user/history/{id}` | Delete a specific query record *(Auth required)* |

### Core Routing

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/query` | Analyse, route, and return LLM response (standard) |
| `POST` | `/v1/query/stream` | **SSE streaming** — metadata first, then token-by-token response |
| `POST` | `/v1/cognitive-profile` | Analyse prompt complexity without calling any LLM |
| `POST` | `/v1/optimize` | Compress and optimize a prompt via GPT-OSS-120B |

### Analytics

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/stats` | Global aggregated statistics from PostgreSQL |

---

## 📂 Project Structure

```
CORA/
├── main.py                          # FastAPI app (lifespan, routes, streaming)
├── database.py                      # SQLAlchemy AsyncEngine & session factory
├── db_models.py                     # ORM models (User, UserSession, QueryRecord)
├── auth.py                          # bcrypt hashing, HTTPBearer dependencies
├── schemas.py                       # Pydantic request/response models
├── cognitive.py                     # Legacy cognitive analysis utilities
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
│
├── cognitive_module/                # Core Cognitive Engine
│   ├── __init__.py                  # Public API exports
│   ├── config.py                    # Weights, tier boundaries, task multipliers
│   ├── models.py                    # CognitiveProfile dataclass
│   ├── feature_extractor.py         # 42+ heuristic feature extraction
│   ├── scorer.py                    # Abstract base + Strategy Factory
│   ├── rule_scorer.py               # Rule-based scoring implementation
│   ├── nemo_scorer.py               # NeMo-enhanced scorer
│   ├── llm_scorer.py                # LLM-powered complexity scoring
│   ├── ml_scorer.py                 # DistilBERT ML scorer
│   ├── routing.py                   # Tier → model assignment logic
│   └── training_data.py             # Training data utilities
│
├── llm_providers/                   # LLM Provider Modules
│   ├── __init__.py                  # Central registry, call_llm(), fallback chain
│   ├── base.py                      # Shared httpx client pool
│   ├── gemma_2b.py                  # Tier 0 — Gemma 2B (NVIDIA NIM)
│   ├── llama_8b.py                  # Tier 1 — Llama 3.1 8B (NVIDIA NIM)
│   ├── mistral_7b.py                # Tier 2 — Mistral 7B v3 (NVIDIA NIM)
│   ├── mixtral_8x7b.py              # Tier 3/4 — Mixtral 8x7B (NVIDIA NIM)
│   ├── devstral_123b.py             # Devstral 123B (currently degraded)
│   └── prompt_optimizer.py          # GPT-OSS-120B prompt compression
│
└── frontend/                        # React + Vite Frontend
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx                  # Root layout with aurora background
        ├── App.module.css           # Global layout constraints (85% width)
        ├── index.css                # Design system tokens & aurora animation
        ├── context/ThemeContext.jsx  # Light/dark theme provider
        ├── services/api.js          # HTTP + SSE streaming client
        └── components/
            ├── Navbar.jsx           # Sticky nav with auth & theme toggle
            ├── Hero.jsx             # Landing page hero section
            ├── PromptInput.jsx      # Glassmorphic input with optimizer
            ├── ResultsPanel.jsx     # Streaming response + telemetry
            ├── Dashboard.jsx        # Analytics metrics grid
            ├── Sidebar.jsx          # Collapsible history panel
            ├── BentoGrid.jsx        # Feature showcase grid
            ├── PipelineViz.jsx      # Cognitive pipeline flow diagram
            ├── ParticleCanvas.jsx   # Ambient particle system
            ├── PromptOptimizer.jsx  # Prompt comparison UI
            ├── AuthModal.jsx        # Login/register modal
            ├── ProfileModal.jsx     # Profile editor modal
            ├── Footer.jsx           # Footer with GitHub link
            └── Stats.jsx            # Quick statistics display
```

---

## 🔒 Security

- **Passwords** are hashed with `bcrypt` — never stored in plaintext
- **Session tokens** are 64-character hex UUIDs with 72-hour auto-expiry
- **API keys** are loaded exclusively from environment variables — never hardcoded
- **`.env`** is gitignored and never committed to the repository
- **CORS** is configured at the middleware level
- All external links use `rel="noopener noreferrer"` to prevent tabnabbing

---
