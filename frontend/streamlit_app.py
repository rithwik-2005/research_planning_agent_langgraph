from __future__ import annotations

import json
import os
import re
import time
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# ============================================================
# Config
# API_BASE is injected by docker-compose as an env var.
# Falls back to localhost when running without Docker.
# ============================================================
API_BASE = os.getenv("API_BASE", "http://localhost:8000")


# ============================================================
# API helpers — all communication with FastAPI goes here
# ============================================================
def api_health() -> bool:
    """Check if FastAPI + Redis are reachable."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        data = r.json()
        # healthy only if both FastAPI and Redis are up
        return r.status_code == 200 and data.get("redis", False)
    except Exception:
        return False


def api_generate(topic: str, as_of: str) -> Optional[str]:
    """POST /blogs/generate → returns job_id."""
    try:
        r = requests.post(
            f"{API_BASE}/blogs/generate",
            json={"topic": topic, "as_of": as_of},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["job_id"]
    except Exception as exc:
        st.error(f"Failed to start job: {exc}")
        return None


def api_poll(job_id: str) -> Dict[str, Any]:
    """GET /blogs/{job_id} → full job dict from Redis."""
    r = requests.get(f"{API_BASE}/blogs/{job_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def api_list() -> List[Dict[str, Any]]:
    """GET /blogs → list of all jobs stored in Redis."""
    try:
        r = requests.get(f"{API_BASE}/blogs", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def api_delete(job_id: str) -> bool:
    """DELETE /blogs/{job_id} → removes job from Redis."""
    try:
        r = requests.delete(f"{API_BASE}/blogs/{job_id}", timeout=10)
        return r.status_code == 204
    except Exception:
        return False


# ============================================================
# Polling loop
# Streamlit polls FastAPI every 3 seconds until the job
# stored in Redis reaches status "done" or "failed"
# ============================================================
def poll_until_done(
    job_id: str,
    interval: float = 3.0,
    timeout: float = 900.0,
) -> Dict[str, Any]:
    deadline       = time.time() + timeout
    status_widget  = st.status("⏳ Waiting for job to start…", expanded=True)
    progress_widget = st.empty()

    status_labels = {
        "queued":  "⏳ Queued — waiting to start…",
        "running": "⚙️  Running — LangGraph pipeline active…",
        "done":    "✅ Done",
        "failed":  "❌ Failed",
    }

    while time.time() < deadline:
        job = api_poll(job_id)
        s   = job.get("status", "queued")

        status_widget.update(
            label=status_labels.get(s, s),
            expanded=(s not in ("done", "failed")),
        )

        # Live progress card shown while waiting
        progress_widget.json({
            "status":      s,
            "job_id":      job_id,
            "topic":       job.get("topic"),
            "blog_title":  job.get("blog_title"),
            "mode":        job.get("mode"),
            "sections":    job.get("sections_count"),
        })

        if s == "done":
            status_widget.update(label="✅ Blog ready!", state="complete", expanded=False)
            progress_widget.empty()
            return job

        if s == "failed":
            status_widget.update(label="❌ Failed", state="error", expanded=True)
            st.error(f"Job failed: {job.get('error')}")
            return job

        time.sleep(interval)

    st.warning("⚠️ Timed out. Check the API logs.")
    return api_poll(job_id)


# ============================================================
# Utilities
# ============================================================
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def list_past_blogs() -> List[Path]:
    """Scans current directory for saved .md files."""
    files = [p for p in Path(".").glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


# ============================================================
# Markdown renderer with local image support
# ============================================================
_MD_IMG_RE  = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        if md[last:m.start()]:
            parts.append(("md", md[last:m.start()]))
        parts.append(("img", f"{m.group('alt')}|||{m.group('src')}"))
        last = m.end()
    if md[last:]:
        parts.append(("md", md[last:]))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]
        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)
        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt   = parts[i + 1][1].lstrip()
            first = nxt.splitlines()[0].strip() if nxt.strip() else ""
            mc    = _CAPTION_RE.match(first)
            if mc:
                caption = mc.group("cap").strip()
                parts[i + 1] = ("md", "\n".join(nxt.splitlines()[1:]))

        if src.startswith("http"):
            st.image(src, caption=caption or alt or None, use_container_width=True)
        else:
            p = Path(src.strip().lstrip("./")).resolve()
            if p.exists():
                st.image(str(p), caption=caption or alt or None, use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}`")
        i += 1


# ============================================================
# Session state bootstrap
# ============================================================
if "last_job" not in st.session_state:
    st.session_state["last_job"] = None

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")
st.title("Blog Writing Agent")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:

    # ── API + Redis status badge ─────────────────────────────
    healthy = api_health()
    st.caption(
        f"API + Redis: {'🟢 Online' if healthy else '🔴 Offline'} — `{API_BASE}`"
    )
    if not healthy:
        st.warning(
            "Services not reachable. Run:\n"
            "```\ndocker-compose up --build\n```"
        )

    # ── Generate new blog ────────────────────────────────────
    st.header("Generate New Blog")
    topic   = st.text_area("Topic", height=120)
    as_of   = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary", disabled=not healthy)

    # ── Past blogs (local .md files) ────────────────────────
    # These are blogs saved to disk by blog_writer.py
    st.divider()
    st.subheader("📁 Past blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved .md blogs found.")
    else:
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title   = extract_title_from_md(md_text, p.stem)
            except Exception:
                title   = p.stem
                md_text = ""

            st.markdown(f"**{title}**  \n`{p.name}`")

            col_load, col_del = st.columns(2)

            # Load blog into main area
            if col_load.button("📂 Open", key=f"load_md_{p.name}", use_container_width=True):
                st.session_state["last_job"] = {
                    "status":      "done",
                    "blog_title":  title,
                    "final_md":    md_text,
                    "plan":        None,
                    "evidence":    [],
                    "image_specs": [],
                }
                st.rerun()

            # Delete .md file from disk
            if col_del.button("🗑️ Delete", key=f"del_md_{p.name}", use_container_width=True):
                try:
                    p.unlink()
                    if (st.session_state.get("last_job") or {}).get("blog_title") == title:
                        st.session_state["last_job"] = None
                    st.success(f"Deleted `{p.name}`")
                except Exception as e:
                    st.error(f"Could not delete: {e}")
                st.rerun()

            st.markdown("---")

    # ── API Job History (from Redis via FastAPI) ─────────────
    # These are jobs stored in Redis — persist across restarts
    st.divider()
    st.subheader("🕓 API Job History")
    st.caption("Jobs stored in Redis — survive server restarts")

    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    job_list = api_list()
    if not job_list:
        st.caption("No jobs in Redis yet.")
    else:
        for j in job_list:
            label = j.get("blog_title") or j.get("topic") or j["job_id"][:8]
            status_icon = {
                "done":    "✅",
                "running": "⚙️",
                "queued":  "⏳",
                "failed":  "❌",
            }.get(j["status"], "•")

            st.markdown(f"{status_icon} **{label}**")

            c_load, c_del = st.columns(2)

            # Load job from Redis into main area
            if c_load.button("📂 Open", key=f"load_api_{j['job_id']}", use_container_width=True):
                job_detail = api_poll(j["job_id"])
                st.session_state["last_job"] = job_detail
                st.rerun()

            # Delete job from Redis
            if c_del.button("🗑️ Delete", key=f"del_api_{j['job_id']}", use_container_width=True):
                api_delete(j["job_id"])
                current = st.session_state.get("last_job") or {}
                if current.get("job_id") == j["job_id"]:
                    st.session_state["last_job"] = None
                st.rerun()

            st.markdown("---")


# ============================================================
# Generate — kicks off job and polls until done
# ============================================================
if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    job_id = api_generate(topic.strip(), as_of.isoformat())
    if job_id:
        st.info(f"Job started: `{job_id}`")
        final_job = poll_until_done(job_id)
        st.session_state["last_job"] = final_job
        st.rerun()


# ============================================================
# Main area — render the active job
# ============================================================
job = st.session_state.get("last_job")

tab_plan, tab_evidence, tab_preview, tab_images, tab_debug = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Debug"]
)

if not job or job.get("status") != "done":
    for tab in (tab_plan, tab_evidence, tab_preview, tab_images, tab_debug):
        with tab:
            st.info("Generate a new blog or open one from the sidebar.")
    st.stop()


# ── Plan tab ────────────────────────────────────────────────
with tab_plan:
    st.subheader("Plan")
    plan_obj = job.get("plan")
    if not plan_obj:
        st.info("Plan metadata not available (only present on a fresh API run, not when loading a .md file).")
    else:
        pd_ = (
            plan_obj if isinstance(plan_obj, dict)
            else json.loads(json.dumps(plan_obj, default=str))
        )
        st.write("**Title:**", pd_.get("blog_title"))
        c = st.columns(3)
        c[0].write("**Audience:** " + str(pd_.get("audience", "")))
        c[1].write("**Tone:** "     + str(pd_.get("tone", "")))
        c[2].write("**Kind:** "     + str(pd_.get("blog_kind", "")))

        tasks = pd_.get("tasks", [])
        if tasks:
            df = pd.DataFrame([{
                "id":           t.get("id"),
                "title":        t.get("title"),
                "target_words": t.get("target_words"),
                "research":     t.get("requires_research"),
                "citations":    t.get("requires_citations"),
                "code":         t.get("requires_code"),
                "tags":         ", ".join(t.get("tags") or []),
            } for t in tasks]).sort_values("id")
            st.dataframe(df, use_container_width=True, hide_index=True)
            with st.expander("Full task JSON"):
                st.json(tasks)


# ── Evidence tab ─────────────────────────────────────────────
with tab_evidence:
    st.subheader("Evidence")
    evidence = job.get("evidence") or []
    if not evidence:
        st.info("No evidence (closed_book mode, or loaded from a .md file).")
    else:
        st.dataframe(
            pd.DataFrame([{
                "title":        e.get("title"),
                "published_at": e.get("published_at"),
                "source":       e.get("source"),
                "url":          e.get("url"),
            } for e in evidence]),
            use_container_width=True,
            hide_index=True,
        )


# ── Preview tab ───────────────────────────────────────────────
with tab_preview:
    st.subheader("Markdown Preview")
    final_md = job.get("final_md") or ""
    if not final_md:
        st.warning("No markdown content available.")
    else:
        render_markdown_with_local_images(final_md)

        blog_title  = job.get("blog_title") or extract_title_from_md(final_md, "blog")
        md_filename = f"{safe_slug(blog_title)}.md"

        st.download_button(
            "⬇️ Download Markdown",
            data=final_md.encode(),
            file_name=md_filename,
            mime="text/markdown",
        )
        bundle = bundle_zip(final_md, md_filename, Path("images"))
        st.download_button(
            "📦 Download Bundle (MD + images)",
            data=bundle,
            file_name=f"{safe_slug(blog_title)}_bundle.zip",
            mime="application/zip",
        )


# ── Images tab ────────────────────────────────────────────────
with tab_images:
    st.subheader("Images")
    specs      = job.get("image_specs") or []
    images_dir = Path("images")

    if not specs and not images_dir.exists():
        st.info("No images generated for this blog.")
    else:
        if specs:
            st.json(specs)
        if images_dir.exists():
            files = sorted(p for p in images_dir.iterdir() if p.is_file())
            if not files:
                st.warning("images/ folder exists but is empty.")
            for p in files:
                st.image(str(p), caption=p.name, use_container_width=True)
            z = images_zip(images_dir)
            if z:
                st.download_button(
                    "⬇️ Download Images (zip)",
                    data=z,
                    file_name="images.zip",
                    mime="application/zip",
                )


# ── Debug tab ─────────────────────────────────────────────────
with tab_debug:
    st.subheader("Raw Job Payload (from Redis)")
    st.caption("Everything stored in Redis for this job, except the full markdown.")
    st.json({k: v for k, v in job.items() if k != "final_md"})
    if job.get("final_md"):
        with st.expander("Raw Markdown"):
            st.code(job["final_md"], language="markdown")