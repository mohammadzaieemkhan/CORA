"""
CORA — Cognitive Orchestration and Reasoning Allocator
FastAPI backend that routes prompts to LLMs based on cognitive complexity.

Features:
  - Multi-dimensional cognitive analysis (cognitive_module)
  - PostgreSQL persistence (users, sessions, query history)
  - User authentication with session tokens
  - Tiered LLM routing with provider fallback
"""

import os
import time
import json
import asyncio
import logging
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
from functools import lru_cache


from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

# ── Load .env FIRST (before anything reads env vars) ─────────────────────────
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ── Local imports ────────────────────────────────────────────────────────────
from cognitive_module import (
    create_scorer,
    generate_routing_reason,
    TASK_TYPE_META,
)
from complexity_score import score_to_tier, get_score_breakdown
from database import init_db, close_db, get_db
from llm_providers import call_llm, get_tier_model_info, close_clients
from db_models import User, UserSession, QueryRecord
from auth import (
    hash_password,
    verify_password,
    create_user_session,
    invalidate_session,
    get_current_user,
    get_optional_user,
    get_current_session,
)
from schemas import (
    RegisterRequest,
    LoginRequest,
    AuthResponse,
    UserProfileResponse,
    QueryRequest,
    QueryResponse,
    CognitiveProfileResponse,
    StatsResponse,
    QueryHistoryItem,
    QueryHistoryResponse,
    UpdateProfileRequest,
    PromptOptimizeRequest,
    PromptMetrics,
    PromptOptimizeResponse,
)

from llm_providers.prompt_optimizer import optimize_prompt as gpt_optimize

logger = logging.getLogger("cora")
logging.basicConfig(level=logging.INFO)

# ── Cognitive Scorer ─────────────────────────────────────────────────────────
cognitive_scorer = create_scorer()

print(f"[CORA] .env path: {env_path} (exists: {env_path.exists()})")

# ── Score Cache (avoids re-scoring identical prompts) ────────────────────────
_score_cache: dict[str, tuple] = {}  # hash → (profile, budget_score, tier)
_CACHE_MAX = 512


def _cached_profile(prompt: str):
    """Score a prompt with LRU-style caching for profile."""
    key = hashlib.md5(prompt.encode()).hexdigest()
    if key in _score_cache:
        return _score_cache[key]
    
    profile = cognitive_scorer.score(prompt)
    
    # Evict oldest if full
    if len(_score_cache) >= _CACHE_MAX:
        oldest = next(iter(_score_cache))
        del _score_cache[oldest]
    
    _score_cache[key] = profile
    return profile


# ── App Lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB. Shutdown: close DB pool."""
    await init_db()
    print("[CORA] Database initialized.")
    
    # Warm up cognitive scorer
    if hasattr(cognitive_scorer, "_ready") and getattr(cognitive_scorer, "_ready", False):
        print("[CORA] Warming up PyTorch ML pipeline...")
        cognitive_scorer.score("Warming up the neural network with a tiny prompt.")
        print("[CORA] ML pipeline warmed up and ready.")
        
    yield
    await close_clients()  # Shutdown persistent HTTP client pools
    await close_db()
    print("[CORA] Database connection closed.")


app = FastAPI(title="CORA", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static Files (Frontend Assets) ───────────────────────────────────────────
assets_path = Path(__file__).parent / "assets"
if assets_path.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")





# ════════════════════════════════════════════════════════════════════════════════
#  AUTH ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/v1/auth/register", response_model=AuthResponse)
async def register(
    req: RegisterRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create a new user account and return a session token."""
    # Check username uniqueness
    existing = await db.execute(select(User).where(User.username == req.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already taken.")

    # Check email uniqueness
    existing_email = await db.execute(select(User).where(User.email == req.email))
    if existing_email.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered.")

    # Create user
    user = User(
        username=req.username,
        email=req.email,
        password_hash=hash_password(req.password),
    )
    db.add(user)
    await db.flush()

    # Create session
    session = await create_user_session(
        db, user,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    return AuthResponse(
        token=session.token,
        username=user.username,
        user_id=str(user.id),
        expires_at=session.expires_at.isoformat(),
        message="Account created successfully.",
    )


@app.post("/v1/auth/login", response_model=AuthResponse)
async def login(
    req: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Authenticate and return a session token."""
    result = await db.execute(select(User).where(User.username == req.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated.")

    session = await create_user_session(
        db, user,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    return AuthResponse(
        token=session.token,
        username=user.username,
        user_id=str(user.id),
        expires_at=session.expires_at.isoformat(),
        message="Login successful.",
    )


@app.post("/v1/auth/logout")
async def logout(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    request: Request = None,
):
    """Invalidate the current session."""
    auth_header = request.headers.get("authorization", "")
    token = auth_header.replace("Bearer ", "").strip() if auth_header else ""

    if token:
        await invalidate_session(db, token)

    return {"message": "Logged out successfully."}


# ════════════════════════════════════════════════════════════════════════════════
#  USER ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/v1/user/profile", response_model=UserProfileResponse)
async def get_user_profile(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the authenticated user's profile and query stats."""
    # Aggregate stats
    stats_result = await db.execute(
        select(
            func.count(QueryRecord.id).label("total"),
            func.coalesce(func.sum(QueryRecord.tokens_saved), 0).label("tokens_saved"),
            func.coalesce(func.avg(QueryRecord.budget_score), 0).label("avg_score"),
        ).where(QueryRecord.user_id == user.id)
    )
    row = stats_result.one()

    # Most common task type
    task_result = await db.execute(
        select(QueryRecord.task_type, func.count().label("cnt"))
        .where(QueryRecord.user_id == user.id, QueryRecord.task_type.isnot(None))
        .group_by(QueryRecord.task_type)
        .order_by(func.count().desc())
        .limit(1)
    )
    top_task_row = task_result.first()

    return UserProfileResponse(
        user_id=str(user.id),
        username=user.username,
        email=user.email,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None,
        total_queries=row.total,
        total_tokens_saved=round(float(row.tokens_saved), 2),
        average_budget_score=round(float(row.avg_score), 1),
        top_task_type=top_task_row[0] if top_task_row else None,
    )


@app.put("/v1/user/profile", response_model=UserProfileResponse)
async def update_user_profile(
    req: UpdateProfileRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the authenticated user's profile (email and/or password)."""
    if req.email:
        existing = await db.execute(select(User).where(User.email == req.email))
        if existing.scalar_one_or_none() and existing.scalar_one_or_none().id != user.id:
            raise HTTPException(status_code=409, detail="Email already safely linked to another account.")
        user.email = req.email

    if req.password:
        user.password_hash = hash_password(req.password)

    db.add(user)
    await db.flush()

    return await get_user_profile(user=user, db=db)


@app.get("/v1/user/history", response_model=QueryHistoryResponse)
async def get_query_history(
    page: int = 1,
    page_size: int = 20,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the authenticated user's query history (paginated)."""
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 20

    offset = (page - 1) * page_size

    # Total count
    count_result = await db.execute(
        select(func.count(QueryRecord.id)).where(QueryRecord.user_id == user.id)
    )
    total = count_result.scalar() or 0

    # Fetch page
    result = await db.execute(
        select(QueryRecord)
        .where(QueryRecord.user_id == user.id)
        .order_by(QueryRecord.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    records = result.scalars().all()

    items = [
        QueryHistoryItem(
            id=str(q.id),
            prompt=q.prompt[:200] + ("..." if len(q.prompt) > 200 else ""),
            response=q.response,
            model_used=q.model_used,
            tier_assigned=q.tier_assigned,
            budget_score=q.budget_score,
            tokens_used=q.tokens_used,
            tokens_saved=q.tokens_saved,
            latency_ms=q.latency_ms,
            task_type=q.task_type,
            routing_reason=q.routing_reason,
            cognitive_profile=q.cognitive_profile,
            created_at=q.created_at.isoformat(),
        )
        for q in records
    ]

    return QueryHistoryResponse(
        queries=items,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(offset + page_size) < total,
    )


@app.delete("/v1/user/history/{query_id}")
async def delete_query_history(
    query_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a specific query from the authenticated user's history."""
    import uuid as _uuid
    try:
        qid = _uuid.UUID(query_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid query ID format.")

    result = await db.execute(
        select(QueryRecord).where(
            QueryRecord.id == qid,
            QueryRecord.user_id == user.id,
        )
    )
    record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(status_code=404, detail="Query not found.")

    await db.delete(record)
    await db.flush()

    return {"message": "Query deleted successfully.", "deleted_id": query_id}


# ════════════════════════════════════════════════════════════════════════════════
#  QUERY ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

@app.post("/v1/optimize", response_model=PromptOptimizeResponse)
async def optimize_prompt_endpoint(
    req: PromptOptimizeRequest,
    user: Optional[User] = Depends(get_optional_user)
):
    """Use DeepSeek to safely optimize a prompt for maximum token efficiency."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
        
    try:
        # Score original WHILE the optimizer runs (concurrent)
        orig_profile = _cached_profile(req.prompt)
        orig_tier, _, orig_budget = score_to_tier(orig_profile, req.prompt)
        
        # Call GPT optimizer
        optimized_text = await gpt_optimize(req.prompt, req.user_api_key)
        
        # Score the optimized text
        opt_profile = _cached_profile(optimized_text)
        opt_tier, _, opt_budget = score_to_tier(opt_profile, optimized_text)
        
        # Helper to compute metrics from pre-computed scores
        def build_metrics(text: str, budget_score: int, tier: str) -> PromptMetrics:
            _, model_display = get_tier_model_info(tier)
            words = len(text.split())
            tokens_used = round(words / 0.75, 2)
            gpt4o_equiv = round(tokens_used * 1.3, 2)
            tokens_saved = round(gpt4o_equiv - tokens_used, 2)
            
            return PromptMetrics(
                prompt=text,
                tokens_used=tokens_used,
                tokens_saved=tokens_saved,
                budget_score=budget_score,
                tier_assigned=tier,
                model_used=model_display
            )
            
        orig_metrics = build_metrics(req.prompt, orig_budget, orig_tier)
        opt_metrics = build_metrics(optimized_text, opt_budget, opt_tier)
        
        return PromptOptimizeResponse(
            original=orig_metrics,
            suggested=opt_metrics
        )
        
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/query", response_model=QueryResponse)
async def handle_query(
    req: QueryRequest,
    user: Optional[User] = Depends(get_optional_user),
    session: Optional[UserSession] = Depends(get_current_session),
    db: AsyncSession = Depends(get_db),
):
    """Send a prompt to CORA — analysed, routed, and persisted."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    # ── Cognitive Analysis (cached — near-instant on repeat prompts) ──
    profile = _cached_profile(req.prompt)
    tier, cora_score, budget_score = score_to_tier(profile, req.prompt)
    breakdown = get_score_breakdown(profile, req.prompt)
    
    profile_dict = profile.to_dict()
    profile_dict["cora_score"] = cora_score
    profile_dict["complexity_breakdown"] = breakdown
    
    model_id, model_display = get_tier_model_info(tier)
    routing_reason = generate_routing_reason(profile, tier, model_display)

    # Use user's API key override if available
    user_key = req.user_api_key
    if not user_key and user and user.api_key_override:
        user_key = user.api_key_override

    start = time.time()
    response_text, display_model = await call_llm(tier, req.prompt, user_key)
    latency_ms = round((time.time() - start) * 1000, 2)

    prompt_words = len(req.prompt.split())
    response_words = len(response_text.split())
    tokens_used = round((prompt_words + response_words) / 0.75, 2)
    gpt4o_equiv = round(tokens_used * 1.3, 2)
    tokens_saved = round(gpt4o_equiv - tokens_used, 2)

    # ── Persist to database ──
    query_record = QueryRecord(
        user_id=user.id if user else None,
        session_id=session.id if session else None,
        prompt=req.prompt,
        response=response_text,
        model_used=display_model,
        tier_assigned=tier,
        budget_score=budget_score,
        tokens_used=tokens_used,
        tokens_saved=tokens_saved,
        latency_ms=latency_ms,
        cognitive_profile=profile_dict,
        routing_reason=routing_reason,
        task_type=profile.task_type.value,
    )
    db.add(query_record)
    await db.flush()

    return QueryResponse(
        response=response_text,
        model_used=display_model,
        tier_assigned=tier,
        budget_score=budget_score,
        tokens_used=tokens_used,
        tokens_saved=tokens_saved,
        latency_ms=latency_ms,
        cognitive_profile=profile_dict,
        routing_reason=routing_reason,
    )


@app.post("/v1/query/stream")
async def handle_query_stream(
    req: QueryRequest,
    user: Optional[User] = Depends(get_optional_user),
    session: Optional[UserSession] = Depends(get_current_session),
    db: AsyncSession = Depends(get_db),
):
    """
    SSE streaming endpoint — sends cognitive metadata first, then streams
    the LLM response token-by-token for near-zero perceived latency.
    
    Event types:
      - "meta"   : JSON with tier, model, cognitive profile, routing reason
      - "token"  : individual text chunk from the LLM
      - "done"   : final summary with latency_ms, tokens_used, tokens_saved, record_id
      - "error"  : error message if something goes wrong
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    # ── Cognitive Analysis (instant with rule scorer + cache) ──
    profile = _cached_profile(req.prompt)
    tier, cora_score, budget_score = score_to_tier(profile, req.prompt)
    breakdown = get_score_breakdown(profile, req.prompt)

    profile_dict = profile.to_dict()
    profile_dict["cora_score"] = cora_score
    profile_dict["complexity_breakdown"] = breakdown

    model_id, model_display = get_tier_model_info(tier)
    routing_reason = generate_routing_reason(profile, tier, model_display)

    user_key = req.user_api_key
    if not user_key and user and hasattr(user, 'api_key_override') and user.api_key_override:
        user_key = user.api_key_override

    async def event_generator():
        # 1. Send metadata immediately (user sees tier/model before LLM even starts)
        meta = {
            "tier_assigned": tier,
            "model_used": model_display,
            "budget_score": budget_score,
            "cognitive_profile": profile_dict,
            "routing_reason": routing_reason,
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        # 2. Stream LLM response
        start = time.time()
        full_response = ""
        try:
            # We use call_llm even for streaming meta-data, but since we don't have
            # a real stream=True implementation for every provider yet, we simulate
            # the streaming experience by chunking the final response.
            response_text, display_model = await call_llm(tier, req.prompt, user_key)
            
            # Send response in chunks for streaming feel
            chunk_size = 32  # slightly larger chunks for better throughput
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                full_response += chunk
                yield f"event: token\ndata: {json.dumps({'text': chunk})}\n\n"
                await asyncio.sleep(0.01) # tiny sleep to ensure smooth UI updates

            latency_ms = round((time.time() - start) * 1000, 2)
            prompt_words = len(req.prompt.split())
            response_words = len(full_response.split())
            tokens_used = round((prompt_words + response_words) / 0.75, 2)
            gpt4o_equiv = round(tokens_used * 1.3, 2)
            tokens_saved = round(gpt4o_equiv - tokens_used, 2)

            # ── Persist to database ──
            record_id = None
            try:
                query_record = QueryRecord(
                    user_id=user.id if user else None,
                    session_id=session.id if session else None,
                    prompt=req.prompt,
                    response=full_response,
                    model_used=display_model,
                    tier_assigned=tier,
                    budget_score=budget_score,
                    tokens_used=tokens_used,
                    tokens_saved=tokens_saved,
                    latency_ms=latency_ms,
                    cognitive_profile=profile_dict,
                    routing_reason=routing_reason,
                    task_type=profile.task_type.value,
                )
                db.add(query_record)
                await db.flush()
                record_id = str(query_record.id)
            except Exception as db_err:
                logger.error(f"Failed to persist streaming query: {db_err}")

            # 3. Send final summary (includes real DB record_id for delete operations)
            done_data = {
                "latency_ms": latency_ms,
                "tokens_used": tokens_used,
                "tokens_saved": tokens_saved,
                "model_used": display_model,
                "record_id": record_id,
            }
            yield f"event: done\ndata: {json.dumps(done_data)}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/v1/cognitive-profile", response_model=CognitiveProfileResponse)
async def get_cognitive_profile(req: QueryRequest):
    """Analyse a prompt's cognitive complexity without calling any LLM."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    profile = _cached_profile(req.prompt)
    tier, cora_score, budget_score = score_to_tier(profile, req.prompt)
    breakdown = get_score_breakdown(profile, req.prompt)

    profile_dict = profile.to_dict()
    profile_dict["cora_score"] = cora_score
    profile_dict["complexity_breakdown"] = breakdown

    model_id, model_display = get_tier_model_info(tier)
    routing_reason = generate_routing_reason(profile, tier, model_display)
    meta = TASK_TYPE_META[profile.task_type]

    return CognitiveProfileResponse(
        budget_score=budget_score,
        tier=tier,
        cognitive_profile=profile_dict,
        routing_reason=routing_reason,
        task_type=meta["label"],
        task_type_icon=meta["icon"],
        confidence=profile.confidence,
    )


# ════════════════════════════════════════════════════════════════════════════════
#  STATS ENDPOINT (from PostgreSQL)
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/v1/stats", response_model=StatsResponse)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Global stats aggregated from the database."""
    # Aggregate counts
    result = await db.execute(
        select(
            func.count(QueryRecord.id).label("total"),
            func.coalesce(func.sum(QueryRecord.tokens_saved), 0).label("tokens_saved"),
            func.coalesce(func.avg(QueryRecord.budget_score), 0).label("avg_score"),
        )
    )
    row = result.one()

    # Routing distribution
    tier_result = await db.execute(
        select(QueryRecord.tier_assigned, func.count().label("cnt"))
        .where(QueryRecord.tier_assigned.isnot(None))
        .group_by(QueryRecord.tier_assigned)
    )
    dist = {"Tier 0": 0, "Tier 1": 0, "Tier 2": 0, "Tier 3": 0, "Tier 4": 0}
    for tier_name, count in tier_result.all():
        dist[tier_name] = count

    return StatsResponse(
        total_queries=row.total,
        total_tokens_saved=round(float(row.tokens_saved), 2),
        average_budget_score=round(float(row.avg_score), 1),
        routing_distribution=dist,
    )


# ════════════════════════════════════════════════════════════════════════════════
#  FRONTEND
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve index.html for any non-API route to support client-side routing."""
    # If it's an API route that reached here, it's a 404
    if full_path.startswith("v1/"):
         raise HTTPException(status_code=404, detail="API endpoint not found.")
         
    index_file = Path(__file__).parent / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {"message": "CORA Backend is running. Frontend not found."}
