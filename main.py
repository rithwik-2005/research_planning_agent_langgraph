from __future__ import annotations

import uuid
from datetime import date
from typing import Optional, List, Any, Dict

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend import app as langgraph_app, State

# ============================================================
# In-memory job store  (swap for Redis/DB in production)
# ============================================================
_jobs: dict[str, dict] = {}


# ============================================================
# Request / Response schemas
# ============================================================
class GenerateBlogRequest(BaseModel):
    topic: str
    as_of: Optional[str] = None


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str                          # queued | running | done | failed
    topic: Optional[str] = None
    as_of: Optional[str] = None
    blog_title: Optional[str] = None
    mode: Optional[str] = None
    sections_count: Optional[int] = None
    # rich fields consumed by Streamlit tabs
    plan: Optional[Dict[str, Any]] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    image_specs: Optional[List[Dict[str, Any]]] = None
    final_md: Optional[str] = None
    error: Optional[str] = None


class BlogListItem(BaseModel):
    job_id: str
    status: str
    topic: Optional[str] = None
    blog_title: Optional[str] = None


# ============================================================
# Background runner
# ============================================================
def _run_blog_job(job_id: str, topic: str, as_of: str) -> None:
    _jobs[job_id]["status"] = "running"

    initial_state: State = {
        "topic": topic,
        "as_of": as_of,
        "mode": "closed_book",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "recency_days": 3650,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    try:
        result: dict[str, Any] = langgraph_app.invoke(initial_state)

        plan = result.get("plan")

        # Serialise plan (Pydantic model -> dict)
        plan_dict: Optional[Dict[str, Any]] = None
        if plan is not None:
            plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else dict(plan)

        # Serialise evidence list
        raw_evidence = result.get("evidence") or []
        evidence_list = [
            (e.model_dump() if hasattr(e, "model_dump") else dict(e))
            for e in raw_evidence
        ]

        _jobs[job_id].update(
            {
                "status": "done",
                "blog_title": plan.blog_title if plan else None,
                "mode": result.get("mode"),
                "sections_count": len(result.get("sections", [])),
                "plan": plan_dict,
                "evidence": evidence_list,
                "image_specs": result.get("image_specs") or [],
                "final_md": result.get("final", ""),
            }
        )
    except Exception as exc:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(exc)


# ============================================================
# App
# ============================================================
app = FastAPI(
    title="Blog Writer API",
    description="Async FastAPI wrapper around the LangGraph blog-writer pipeline.",
    version="1.1.0",
)

# Allow Streamlit (different port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- POST /blogs/generate ------------------------------------
@app.post(
    "/blogs/generate",
    response_model=JobCreatedResponse,
    status_code=202,
    summary="Kick off a blog-generation job (async)",
)
async def generate_blog(body: GenerateBlogRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    as_of = body.as_of or date.today().isoformat()

    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "topic": body.topic,
        "as_of": as_of,
        "blog_title": None,
        "mode": None,
        "sections_count": None,
        "plan": None,
        "evidence": [],
        "image_specs": [],
        "final_md": None,
        "error": None,
    }

    background_tasks.add_task(_run_blog_job, job_id, body.topic, as_of)

    return JobCreatedResponse(
        job_id=job_id,
        status="queued",
        message=f"Job {job_id} queued. Poll GET /blogs/{job_id} for status.",
    )


# -- GET /blogs/{job_id} -------------------------------------
@app.get(
    "/blogs/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll job status / fetch finished blog",
)
async def get_blog_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatusResponse(**job)


# -- GET /blogs/{job_id}/markdown ----------------------------
@app.get("/blogs/{job_id}/markdown", summary="Download finished blog as Markdown")
async def get_blog_markdown(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is '{job['status']}', not done yet.")
    return JSONResponse(content={"job_id": job_id, "markdown": job.get("final_md", "")})


# -- GET /blogs ----------------------------------------------
@app.get("/blogs", response_model=List[BlogListItem], summary="List all jobs")
async def list_blogs():
    return [
        BlogListItem(
            job_id=j["job_id"],
            status=j["status"],
            topic=j.get("topic"),
            blog_title=j.get("blog_title"),
        )
        for j in _jobs.values()
    ]


# -- DELETE /blogs/{job_id} ----------------------------------
@app.delete("/blogs/{job_id}", status_code=204, summary="Delete a job record")
async def delete_blog_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    del _jobs[job_id]


# -- GET /health ---------------------------------------------
@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "total_jobs": len(_jobs)}