from __future__ import annotations

import json
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
# FastAPI base URL  —  change to your deployed URL in prod
# ============================================================
API_BASE = "http://localhost:8000"


# ============================================================
# API helpers
# ============================================================
def api_generate(topic: str, as_of: str) -> Optional[str]:
    """POST /blogs/generate → returns job_id or None on error."""
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
    """GET /blogs/{job_id} → full status dict."""
    r = requests.get(f"{API_BASE}/blogs/{job_id}", timeout=10)
    r.raise_for_status()
    return r.json()


def api_list() -> List[Dict[str, Any]]:
    """GET /blogs → list of job summaries."""
    try:
        r = requests.get(f"{API_BASE}/blogs", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def api_delete(job_id: str) -> bool:
    try:
        r = requests.delete(f"{API_BASE}/blogs/{job_id}", timeout=10)
        return r.status_code == 204
    except Exception:
        return False


def api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


# ============================================================
# Poll loop — blocks until done/failed, yields status dicts
# ============================================================
def poll_until_done(
    job_id: str,
    interval: float = 3.0,
    timeout: float = 900.0,
) -> Dict[str, Any]:
    """
    Repeatedly polls the API and updates a Streamlit status widget.
    Returns the final job dict.
    """
    deadline = time.time() + timeout
    status_map = {
        "queued": "⏳ Queued…",
        "running": "⚙️ Running…",
        "done": "✅ Done",
        "failed": "❌ Failed",
    }

    status_widget = st.status("Waiting for API…", expanded=True)
    progress_placeholder = st.empty()

    while time.time() < deadline:
        job = api_poll(job_id)
        s = job.get("status", "queued")
        label = status_map.get(s, s)
        status_widget.update(label=label, expanded=(s not in ("done", "failed")))

        progress_placeholder.json(
            {
                "status": s,
                "job_id": job_id,
                "topic": job.get("topic"),
                "blog_title": job.get("blog_title"),
                "mode": job.get("mode"),
                "sections": job.get("sections_count"),
            }
        )

        if s == "done":
            status_widget.update(label="✅ Done", state="complete", expanded=False)
            return job

        if s == "failed":
            status_widget.update(label="❌ Failed", state="error", expanded=True)
            st.error(f"Job failed: {job.get('error')}")
            return job

        time.sleep(interval)

    st.warning("⚠️ Timed out waiting for the job.")
    return api_poll(job_id)


# ============================================================
# Utilities (unchanged from original)
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
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def list_past_blogs() -> List[Path]:
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# ============================================================
# Markdown renderer with local images (unchanged)
# ============================================================
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))
        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()
    tail = md[last:]
    if tail:
        parts.append(("md", tail))

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
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = Path(src.strip().lstrip("./")).resolve()
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")
        i += 1


# ============================================================
# Page setup
# ============================================================
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")
st.title("Blog Writing Agent")

# API health badge
with st.sidebar:
    healthy = api_health()
    st.caption(
        f"API: {'🟢 Online' if healthy else '🔴 Offline'} — `{API_BASE}`"
    )
    if not healthy:
        st.warning("FastAPI server is not reachable. Start it with:\n```\nuvicorn main:app --reload\n```")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Generate New Blog")
    topic = st.text_area("Topic", height=120)
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary", disabled=not healthy)

    # ── Past blogs ──────────────────────────────────────────
    st.divider()
    st.subheader("Past blogs")
    past_files = list_past_blogs()

    if not past_files:
        st.caption("No saved blogs found (*.md in current folder).")
        selected_md_file = None
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load selected blog"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                st.session_state["last_job"] = {
                    "status": "done",
                    "blog_title": extract_title_from_md(md_text, selected_md_file.stem),
                    "final_md": md_text,
                    "mode": None,
                    "evidence": [],
                    "image_specs": [],
                    "plan": None,
                }

    # ── API job history ──────────────────────────────────────
    st.divider()
    st.subheader("API Job History")
    if st.button("🔄 Refresh"):
        st.rerun()

    job_list = api_list()
    if not job_list:
        st.caption("No jobs yet.")
    else:
        for j in job_list:
            col1, col2 = st.columns([3, 1])
            label = j.get("blog_title") or j.get("topic") or j["job_id"][:8]
            col1.write(f"**{label}** — `{j['status']}`")
            if col2.button("Load", key=f"load_{j['job_id']}"):
                job_detail = api_poll(j["job_id"])
                st.session_state["last_job"] = job_detail
                st.rerun()


# ============================================================
# Session state init
# ============================================================
if "last_job" not in st.session_state:
    st.session_state["last_job"] = None

# ============================================================
# Trigger generation
# ============================================================
if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    job_id = api_generate(topic.strip(), as_of.isoformat())
    if job_id:
        st.info(f"Job created: `{job_id}`")
        job = poll_until_done(job_id)
        st.session_state["last_job"] = job
        st.rerun()

# ============================================================
# Render result
# ============================================================
job = st.session_state.get("last_job")

tab_plan, tab_evidence, tab_preview, tab_images, tab_debug = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Debug"]
)

if not job or job.get("status") != "done":
    for tab in (tab_plan, tab_evidence, tab_preview, tab_images, tab_debug):
        with tab:
            st.info("Enter a topic and click **Generate Blog**, or load a past blog from the sidebar.")
    st.stop()

# ── Plan tab ────────────────────────────────────────────────
with tab_plan:
    st.subheader("Plan")
    plan_obj = job.get("plan")
    if not plan_obj:
        st.info("Plan metadata not returned by the API (available during a live run).")
    else:
        plan_dict = plan_obj if isinstance(plan_obj, dict) else json.loads(json.dumps(plan_obj, default=str))
        st.write("**Title:**", plan_dict.get("blog_title"))
        cols = st.columns(3)
        cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
        cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
        cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))

        tasks = plan_dict.get("tasks", [])
        if tasks:
            df = pd.DataFrame(
                [
                    {
                        "id": t.get("id"),
                        "title": t.get("title"),
                        "target_words": t.get("target_words"),
                        "requires_research": t.get("requires_research"),
                        "requires_citations": t.get("requires_citations"),
                        "requires_code": t.get("requires_code"),
                        "tags": ", ".join(t.get("tags") or []),
                    }
                    for t in tasks
                ]
            ).sort_values("id")
            st.dataframe(df, use_container_width=True, hide_index=True)
            with st.expander("Task details"):
                st.json(tasks)

# ── Evidence tab ─────────────────────────────────────────────
with tab_evidence:
    st.subheader("Evidence")
    evidence = job.get("evidence") or []
    if not evidence:
        st.info("No evidence (closed_book mode, or not returned by API).")
    else:
        rows = [
            {
                "title": e.get("title"),
                "published_at": e.get("published_at"),
                "source": e.get("source"),
                "url": e.get("url"),
            }
            for e in evidence
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Preview tab ───────────────────────────────────────────────
with tab_preview:
    st.subheader("Markdown Preview")
    final_md = job.get("final_md") or ""
    if not final_md:
        st.warning("No markdown content available.")
    else:
        render_markdown_with_local_images(final_md)

        blog_title = job.get("blog_title") or extract_title_from_md(final_md, "blog")
        md_filename = f"{safe_slug(blog_title)}.md"

        st.download_button(
            "⬇️ Download Markdown",
            data=final_md.encode("utf-8"),
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
    specs = job.get("image_specs") or []
    images_dir = Path("images")

    if not specs and not images_dir.exists():
        st.info("No images generated for this blog.")
    else:
        if specs:
            st.write("**Image plan:**")
            st.json(specs)
        if images_dir.exists():
            files = [p for p in images_dir.iterdir() if p.is_file()]
            if not files:
                st.warning("images/ directory exists but is empty.")
            else:
                for p in sorted(files):
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
    st.subheader("Raw Job Payload")
    st.json(
        {
            k: v
            for k, v in job.items()
            if k != "final_md"  # skip large field
        }
    )
    if job.get("final_md"):
        with st.expander("Raw Markdown"):
            st.code(job["final_md"], language="markdown")