from __future__ import annotations

import json
import os
import uuid
from datetime import date
from typing import Any, Dict, List, Optional

import redis
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from blog_writer import app as langgraph_app, State

# ============================================================
# Redis connection
# host="redis" matches the service name in docker-compose.yml
# Falls back to "localhost" when running without Docker
# ============================================================
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True,
)

# Jobs stay in Redis for 7 days then auto-expire
JOB_TTL_SECONDS = 60 * 60 * 24 * 7


# ============================================================
# Redis helpers
# Redis hashes only store strings, so complex fields
# (dicts, lists) are JSON-serialised before saving
# ============================================================
def job_key(job_id: str) -> str:
    """Namespaced Redis key for a job."""
    return f"job:{job_id}"


def save_job(job_id: str, data: dict) -> None:
    """Save or overwrite a full job dict in Redis."""
    flat: dict = {}
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            flat[k] = json.dumps(v, default=str)
        else:
            flat[k] = str(v) if v is not None else ""
    redis_client.hset(job_key(job_id), mapping=flat)
    redis_client.expire(job_key(job_id), JOB_TTL_SECONDS)


def load_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Load a job from Redis and deserialise complex fields."""
    raw = redis_client.hgetall(job_key(job_id))
    if not raw:
        return None
    result: Dict[str, Any] = {}
    for k, v in raw.items():
        try:
            result[k] = json.loads(v)   # handles dicts/lists
        except (json.JSONDecodeError, TypeError):
            result[k] = v or None       # plain string fields
    return result


def delete_job(job_id: str) -> bool:
    """Delete a job from Redis. Returns True if it existed."""
    return bool(redis_client.delete(job_key(job_id)))


def list_jobs() -> List[Dict[str, Any]]:
    """Return a lightweight summary of all jobs (no final_md)."""
    keys = redis_client.keys("job:*")
    jobs = []
    for k in keys:
        raw = redis_client.hgetall(k)
        if raw:
            jobs.append({
                "job_id":     raw.get("job_id", ""),
                "status":     raw.get("status", ""),
                "topic":      raw.get("topic") or None,
                "blog_title": raw.get("blog_title") or None,
            })
    return jobs


# ============================================================
# Pydantic schemas
# ============================================================
class GenerateBlogRequest(BaseModel):
    topic: str
    as_of: Optional[str] = None          # ISO date e.g. "2025-07-01"


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str                           # queued | running | done | failed
    topic: Optional[str] = None
    as_of: Optional[str] = None
    blog_title: Optional[str] = None
    mode: Optional[str] = None           # closed_book | hybrid | open_book
    sections_count: Optional[int] = None
    plan: Optional[Dict[str, Any]] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    image_specs: Optional[List[Dict[str, Any]]] = None
    final_md: Optional[str] = None       # full markdown — only when done
    error: Optional[str] = None


class BlogListItem(BaseModel):
    job_id: str
    status: str
    topic: Optional[str] = None
    blog_title: Optional[str] = None


# ============================================================
# Background worker
# Runs in a thread so FastAPI stays responsive
# ============================================================
def _run_blog_job(job_id: str, topic: str, as_of: str) -> None:
    # 1. Mark job as running in Redis
    redis_client.hset(job_key(job_id), "status", "running")
    redis_client.expire(job_key(job_id), JOB_TTL_SECONDS)

    # 2. Build the initial LangGraph state
    initial_state: State = {
        "topic":               topic,
        "as_of":               as_of,
        "mode":                "closed_book",
        "needs_research":      False,
        "queries":             [],
        "evidence":            [],
        "plan":                None,
        "recency_days":        3650,
        "sections":            [],
        "merged_md":           "",
        "md_with_placeholders": "",
        "image_specs":         [],
        "final":               "",
    }

    try:
        # 3. Run the full LangGraph pipeline (blocking call)
        result: Dict[str, Any] = langgraph_app.invoke(initial_state)

        # 4. Serialise Pydantic objects returned by LangGraph
        plan = result.get("plan")
        plan_dict = (
            plan.model_dump() if hasattr(plan, "model_dump")
            else dict(plan) if plan else None
        )
        evidence_list = [
            (e.model_dump() if hasattr(e, "model_dump") else dict(e))
            for e in (result.get("evidence") or [])
        ]

        # 5. Save the completed job to Redis
        save_job(job_id, {
            "job_id":         job_id,
            "status":         "done",
            "topic":          topic,
            "as_of":          as_of,
            "blog_title":     plan.blog_title if plan else None,
            "mode":           result.get("mode"),
            "sections_count": len(result.get("sections", [])),
            "plan":           plan_dict,
            "evidence":       evidence_list,
            "image_specs":    result.get("image_specs") or [],
            "final_md":       result.get("final", ""),
            "error":          None,
        })

    except Exception as exc:
        # 6. Save failure state to Redis
        redis_client.hset(job_key(job_id), mapping={
            "status": "failed",
            "error":  str(exc),
        })
        redis_client.expire(job_key(job_id), JOB_TTL_SECONDS)


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(
    title="Blog Writer API",
    version="2.0.0",
    description="FastAPI + LangGraph pipeline backed by Redis job store.",
)

# Allow Streamlit container to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── POST /blogs/generate ─────────────────────────────────────
# Creates a job, saves it to Redis, fires background thread
@app.post(
    "/blogs/generate",
    response_model=JobCreatedResponse,
    status_code=202,
    summary="Start a new blog generation job",
)
async def generate_blog(
    body: GenerateBlogRequest,
    background_tasks: BackgroundTasks,
):
    job_id = str(uuid.uuid4())
    as_of  = body.as_of or date.today().isoformat()

    # Save initial queued state
    save_job(job_id, {
        "job_id":         job_id,
        "status":         "queued",
        "topic":          body.topic,
        "as_of":          as_of,
        "blog_title":     None,
        "mode":           None,
        "sections_count": None,
        "plan":           None,
        "evidence":       [],
        "image_specs":    [],
        "final_md":       None,
        "error":          None,
    })

    # Fire the LangGraph pipeline in the background
    background_tasks.add_task(_run_blog_job, job_id, body.topic, as_of)

    return JobCreatedResponse(
        job_id=job_id,
        status="queued",
        message=f"Job {job_id} queued. Poll GET /blogs/{job_id} for updates.",
    )


# ── GET /blogs/{job_id} ──────────────────────────────────────
# Streamlit polls this endpoint until status == "done"
@app.get(
    "/blogs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status and result",
)
async def get_blog_job(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatusResponse(**job)


# ── GET /blogs/{job_id}/markdown ─────────────────────────────
# Download just the raw markdown text
@app.get(
    "/blogs/{job_id}/markdown",
    summary="Download finished blog as plain Markdown",
)
async def get_blog_markdown(job_id: str):
    job = load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.get("status") != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job is '{job.get('status')}', not done yet.",
        )
    return JSONResponse({"job_id": job_id, "markdown": job.get("final_md", "")})


# ── GET /blogs ───────────────────────────────────────────────
# Sidebar job history list in Streamlit
@app.get(
    "/blogs",
    response_model=List[BlogListItem],
    summary="List all jobs",
)
async def list_blogs():
    return [BlogListItem(**j) for j in list_jobs()]


# ── DELETE /blogs/{job_id} ───────────────────────────────────
# Sidebar delete button in Streamlit calls this
@app.delete(
    "/blogs/{job_id}",
    status_code=204,
    summary="Delete a job from Redis",
)
async def delete_blog_job(job_id: str):
    if not delete_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")


# ── GET /health ──────────────────────────────────────────────
# Docker healthcheck + API status badge in Streamlit
@app.get("/health", tags=["meta"], summary="Health check")
async def health():
    try:
        redis_client.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {
        "status":     "ok",
        "redis":      redis_ok,
        "total_jobs": len(list_jobs()),
    }