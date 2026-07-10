"""
api.py — Full Pipeline REST API  (Local File Storage edition)
====================================================
Endpoints:

  STATUS
    GET  /status                      — check pipeline state (local files)

  TRANSCRIPTION
    POST /transcribe/video            — upload a video file → starts background job → returns run_id
    POST /transcribe/youtube          — provide YouTube URL → starts background job → returns run_id
    GET  /transcript                  — get latest transcript from local file

  SEGMENTATION
    POST /segment                     — segment latest transcript → save to local file
    POST /segment/titles              — segment → return titles only
    POST /segment/summaries           — segment → return summaries only
    GET  /segments                    — get latest segments from local file

  DESCRIPTION
    POST /describe                    — describe latest segments → save to local file
    GET  /describe                    — get latest description from local file

  FULL PIPELINE
    POST /pipeline/video              — upload video → starts background job → returns run_id
    POST /pipeline/youtube            — YouTube URL → starts background job → returns run_id

  HISTORY
    GET  /runs                        — list recent pipeline runs (metadata only)
    GET  /runs/{run_id}               — get a specific run by ID (poll this for job status)

NOTE: Long-running endpoints (transcribe, pipeline) return immediately with a run_id
      and status="processing". Poll GET /runs/{run_id} until status is "done" or "error".

Run:
  uvicorn api:app --reload
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

from transcriber import transcribe_from_video, transcribe_from_youtube
from segmenter   import segment_transcript
from describer   import build_segments_summary, generate_description

load_dotenv(Path(__file__).parent / ".env")


# ─────────────────────────────────────────────
# Local file storage (replaces MongoDB)
# ─────────────────────────────────────────────
# The pipeline outputs are saved as plain files in this same folder,
# exactly like the original standalone scripts did:
#   transcript.txt            (from transcriber.py)
#   segments.json             (from segmenter.py)
#   content_description.json  (from describer.py)
#
# Nothing is written to any database. Run/job status (used only for
# polling background jobs) is kept in memory for the life of the process.

import json
import time
import uuid
import threading

_SCRIPT_DIR       = Path(__file__).parent
TRANSCRIPT_PATH   = _SCRIPT_DIR / "transcript.txt"
SEGMENTS_PATH     = _SCRIPT_DIR / "segments.json"
DESCRIPTION_PATH  = _SCRIPT_DIR / "content_description.json"


class _LocalStore:
    """Drop-in replacement for the old `database` module.

    Same function names/signatures as before. Transcript/segments/
    description are persisted to local files in this folder; run
    metadata (for job polling) lives only in memory.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._runs: dict[str, dict] = {}

    # ---- runs (in-memory only, used for polling job status) ----
    def new_run(self, kind: str, source: str) -> str:
        run_id = str(uuid.uuid4())
        with self._lock:
            self._runs[run_id] = {
                "run_id": run_id,
                "type": kind,
                "source": source,
                "status": "processing",
                "created_at": time.time(),
                "error": None,
            }
        return run_id

    def update_run_status(self, run_id: str, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None:
                run["status"] = status
                if error is not None:
                    run["error"] = error

    def list_runs(self, limit: int = 20) -> list:
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda r: r["created_at"], reverse=True)
        return runs[:limit]

    def get_run(self, run_id: str) -> Optional[dict]:
        with self._lock:
            return self._runs.get(run_id)

    # ---- transcript (saved to transcript.txt) ----
    def save_transcript(self, transcript: str, run_id: Optional[str] = None) -> None:
        TRANSCRIPT_PATH.write_text(transcript, encoding="utf-8")
        if run_id:
            with self._lock:
                if run_id in self._runs:
                    self._runs[run_id]["transcript"] = transcript

    def get_transcript(self) -> Optional[str]:
        if not TRANSCRIPT_PATH.exists():
            return None
        return TRANSCRIPT_PATH.read_text(encoding="utf-8")

    # ---- segments (saved to segments.json) ----
    def save_segments(self, segments: list, run_id: Optional[str] = None) -> None:
        with open(SEGMENTS_PATH, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        if run_id:
            with self._lock:
                if run_id in self._runs:
                    self._runs[run_id]["segments"] = segments

    def get_segments(self) -> Optional[list]:
        if not SEGMENTS_PATH.exists():
            return None
        with open(SEGMENTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---- description (saved to content_description.json) ----
    def save_description(self, description: dict, run_id: Optional[str] = None) -> None:
        with open(DESCRIPTION_PATH, "w", encoding="utf-8") as f:
            json.dump(description, f, indent=2, ensure_ascii=False)
        if run_id:
            with self._lock:
                if run_id in self._runs:
                    self._runs[run_id]["description"] = description

    def get_description(self) -> Optional[dict]:
        if not DESCRIPTION_PATH.exists():
            return None
        with open(DESCRIPTION_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---- status ----
    def status(self) -> dict:
        return {
            "transcript":  TRANSCRIPT_PATH.exists(),
            "segments":    SEGMENTS_PATH.exists(),
            "description": DESCRIPTION_PATH.exists(),
        }


db = _LocalStore()


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

class YoutubeRequest(BaseModel):
    url: str

class SegmentOut(BaseModel):
    index: int
    title: str
    summary: str
    text: str
    start_time: str
    end_time: str

class StatusResponse(BaseModel):
    transcript: bool
    segments: bool
    description: bool

class TranscribeResponse(BaseModel):
    transcript: str

class AsyncJobResponse(BaseModel):
    run_id: str
    status: str          # "processing" | "done" | "error"
    message: str

class SegmentResponse(BaseModel):
    segments: list[SegmentOut]

class TitlesResponse(BaseModel):
    titles: list[str]

class SummariesResponse(BaseModel):
    summaries: list[dict]

class DescribeResponse(BaseModel):
    summary: str
    tone_and_style: str
    seo_tags: list[str]

class PipelineResponse(BaseModel):
    run_id: str
    transcript: str
    segments: list[SegmentOut]
    description: DescribeResponse


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Video Pipeline API",
    description=(
        "Full pipeline: video/YouTube → transcribe → segment → describe. "
        "Outputs stored as local files in this folder. Long-running jobs return immediately with a "
        "run_id — poll GET /runs/{run_id} until status is 'done'."
    ),
    version="6.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in .env file.")
    return Groq(api_key=key)


def _to_segment_out(segments: list[dict]) -> list[SegmentOut]:
    return [SegmentOut(**seg) for seg in segments]


async def _save_upload(file: UploadFile) -> str:
    """Save an uploaded video file to a temp location and return the path."""
    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)
    content  = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)
    return tmp_path


def _require_transcript() -> str:
    transcript = db.get_transcript()
    if not transcript:
        raise HTTPException(
            status_code=404,
            detail="No transcript found. Run POST /transcribe/video or /transcribe/youtube first.",
        )
    return transcript


def _require_segments() -> list:
    segments = db.get_segments()
    if not segments:
        raise HTTPException(
            status_code=404,
            detail="No segments found. Run POST /segment first.",
        )
    return segments


# ─────────────────────────────────────────────
# Background task workers
# ─────────────────────────────────────────────

def _bg_transcribe_video(run_id: str, tmp_path: str, tmp_dir: str) -> None:
    """Background worker: transcribe a local video file and save to local files."""
    try:
        client     = Groq(api_key=os.getenv("GROQ_API_KEY"))
        transcript = transcribe_from_video(tmp_path, client)
        db.save_transcript(transcript, run_id)
        db.update_run_status(run_id, "done")
    except Exception as e:
        db.update_run_status(run_id, "error", str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _bg_transcribe_youtube(run_id: str, url: str) -> None:
    """Background worker: download + transcribe a YouTube URL and save to local files."""
    try:
        client     = Groq(api_key=os.getenv("GROQ_API_KEY"))
        transcript = transcribe_from_youtube(url, client)
        db.save_transcript(transcript, run_id)
        db.update_run_status(run_id, "done")
    except Exception as e:
        db.update_run_status(run_id, "error", str(e))


def _bg_pipeline_video(run_id: str, tmp_path: str, tmp_dir: str) -> None:
    """Background worker: full pipeline (transcribe → segment → describe) for a local video."""
    try:
        client     = Groq(api_key=os.getenv("GROQ_API_KEY"))
        transcript = transcribe_from_video(tmp_path, client)
        db.save_transcript(transcript, run_id)

        segments = segment_transcript(transcript, client)
        db.save_segments(segments, run_id)

        description = generate_description(build_segments_summary(segments), client)
        db.save_description(description, run_id)

        db.update_run_status(run_id, "done")
    except Exception as e:
        db.update_run_status(run_id, "error", str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _bg_pipeline_youtube(run_id: str, url: str) -> None:
    """Background worker: full pipeline (transcribe → segment → describe) for a YouTube URL."""
    try:
        client     = Groq(api_key=os.getenv("GROQ_API_KEY"))
        transcript = transcribe_from_youtube(url, client)
        db.save_transcript(transcript, run_id)

        segments = segment_transcript(transcript, client)
        db.save_segments(segments, run_id)

        description = generate_description(build_segments_summary(segments), client)
        db.save_description(description, run_id)

        db.update_run_status(run_id, "done")
    except Exception as e:
        db.update_run_status(run_id, "error", str(e))


# ─────────────────────────────────────────────
# STATUS
# ─────────────────────────────────────────────

@app.get("/status", response_model=StatusResponse)
def get_status():
    """Check which pipeline stages have been completed (based on local files)."""
    return StatusResponse(**db.status())


# ─────────────────────────────────────────────
# TRANSCRIPTION
# ─────────────────────────────────────────────

@app.post("/transcribe/video", response_model=AsyncJobResponse)
async def transcribe_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a video file and start transcription in the background.
    Returns immediately with a run_id. Poll GET /runs/{run_id} for progress.
    Status will be "processing" → "done" (or "error").
    """
    run_id   = db.new_run("video", file.filename)
    tmp_path = await _save_upload(file)
    tmp_dir  = os.path.dirname(tmp_path)

    background_tasks.add_task(_bg_transcribe_video, run_id, tmp_path, tmp_dir)

    return AsyncJobResponse(
        run_id=run_id,
        status="processing",
        message=f"Transcription started. Poll GET /runs/{run_id} for status.",
    )


@app.post("/transcribe/youtube", response_model=AsyncJobResponse)
async def transcribe_youtube(background_tasks: BackgroundTasks, body: YoutubeRequest):
    """
    Provide a YouTube URL and start transcription in the background.
    Returns immediately with a run_id. Poll GET /runs/{run_id} for progress.
    Status will be "processing" → "done" (or "error").
    """
    run_id = db.new_run("youtube", body.url)

    background_tasks.add_task(_bg_transcribe_youtube, run_id, body.url)

    return AsyncJobResponse(
        run_id=run_id,
        status="processing",
        message=f"Transcription started. Poll GET /runs/{run_id} for status.",
    )


@app.get("/transcript", response_model=TranscribeResponse)
def get_transcript():
    """Get the latest transcript from the local transcript.txt file."""
    return TranscribeResponse(transcript=_require_transcript())


# ─────────────────────────────────────────────
# SEGMENTATION
# ─────────────────────────────────────────────

@app.post("/segment", response_model=SegmentResponse)
def segment():
    """
    Segment the latest transcript from the local file, save segments back to segments.json.
    """
    transcript = _require_transcript()
    segments   = segment_transcript(transcript, _client())
    db.save_segments(segments)
    return SegmentResponse(segments=_to_segment_out(segments))


@app.post("/segment/titles", response_model=TitlesResponse)
def segment_titles():
    """Segment the latest transcript and return only the topic titles."""
    transcript = _require_transcript()
    segments   = segment_transcript(transcript, _client())
    db.save_segments(segments)
    return TitlesResponse(titles=[seg["title"] for seg in segments])


@app.post("/segment/summaries", response_model=SummariesResponse)
def segment_summaries():
    """Segment the latest transcript and return only the summaries."""
    transcript = _require_transcript()
    segments   = segment_transcript(transcript, _client())
    db.save_segments(segments)
    return SummariesResponse(
        summaries=[
            {"index": seg["index"], "title": seg["title"], "summary": seg["summary"]}
            for seg in segments
        ]
    )


@app.get("/segments", response_model=SegmentResponse)
def get_segments():
    """Get the latest segments from the local segments.json file."""
    segments = _require_segments()
    return SegmentResponse(segments=_to_segment_out(segments))


# ─────────────────────────────────────────────
# DESCRIPTION
# ─────────────────────────────────────────────

@app.post("/describe", response_model=DescribeResponse)
def describe():
    """
    Generate content description from the latest segments in the local file,
    and save the description back to content_description.json.
    """
    segments    = _require_segments()
    client      = _client()
    description = generate_description(build_segments_summary(segments), client)
    db.save_description(description)
    return DescribeResponse(**description)


@app.get("/describe", response_model=DescribeResponse)
def get_description():
    """Get the latest content description from content_description.json."""
    description = db.get_description()
    if not description:
        raise HTTPException(
            status_code=404,
            detail="No description found. Run POST /describe first.",
        )
    return DescribeResponse(**description)


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────

@app.post("/pipeline/video", response_model=AsyncJobResponse)
async def pipeline_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a video file and run the full pipeline in the background:
      1. Extract audio + transcribe
      2. Segment
      3. Describe
    Returns immediately with a run_id. Poll GET /runs/{run_id} for progress.
    Status will be "processing" → "done" (or "error").
    When done, the full result is available at GET /runs/{run_id}.
    """
    run_id   = db.new_run("video", file.filename)
    tmp_path = await _save_upload(file)
    tmp_dir  = os.path.dirname(tmp_path)

    background_tasks.add_task(_bg_pipeline_video, run_id, tmp_path, tmp_dir)

    return AsyncJobResponse(
        run_id=run_id,
        status="processing",
        message=f"Pipeline started. Poll GET /runs/{run_id} for status.",
    )


@app.post("/pipeline/youtube", response_model=AsyncJobResponse)
async def pipeline_youtube(background_tasks: BackgroundTasks, body: YoutubeRequest):
    """
    Provide a YouTube URL and run the full pipeline in the background:
      1. Download audio + transcribe
      2. Segment
      3. Describe
    Returns immediately with a run_id. Poll GET /runs/{run_id} for progress.
    Status will be "processing" → "done" (or "error").
    When done, the full result is available at GET /runs/{run_id}.
    """
    run_id = db.new_run("youtube", body.url)

    background_tasks.add_task(_bg_pipeline_youtube, run_id, body.url)

    return AsyncJobResponse(
        run_id=run_id,
        status="processing",
        message=f"Pipeline started. Poll GET /runs/{run_id} for status.",
    )


# ─────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────

@app.get("/runs")
def list_runs(limit: int = 20):
    """List the most recent pipeline runs (metadata only, no transcript text)."""
    return db.list_runs(limit=limit)


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    """
    Fetch a specific pipeline run by its run_id.
    Use this to poll for job completion after POST /transcribe/* or /pipeline/*.
    Possible status values: "processing", "done", "error".
    """
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    run.pop("_id", None)    # remove non-serialisable ObjectId
    return run