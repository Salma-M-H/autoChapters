"""
api.py — Full Pipeline REST API  (MongoDB edition)
====================================================
Endpoints:

  STATUS
    GET  /status                      — check pipeline state in MongoDB

  TRANSCRIPTION
    POST /transcribe/video            — upload a video file → starts background job → returns run_id
    POST /transcribe/youtube          — provide YouTube URL → starts background job → returns run_id
    GET  /transcript                  — get latest transcript from MongoDB

  SEGMENTATION
    POST /segment                     — segment latest transcript → save to MongoDB
    POST /segment/titles              — segment → return titles only
    POST /segment/summaries           — segment → return summaries only
    GET  /segments                    — get latest segments from MongoDB

  DESCRIPTION
    POST /describe                    — describe latest segments → save to MongoDB
    GET  /describe                    — get latest description from MongoDB

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
import database as db

load_dotenv(Path(__file__).parent / ".env")


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
        "Outputs stored in MongoDB. Long-running jobs return immediately with a "
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
            detail="No transcript found in MongoDB. Run POST /transcribe/video or /transcribe/youtube first.",
        )
    return transcript


def _require_segments() -> list:
    segments = db.get_segments()
    if not segments:
        raise HTTPException(
            status_code=404,
            detail="No segments found in MongoDB. Run POST /segment first.",
        )
    return segments


# ─────────────────────────────────────────────
# Background task workers
# ─────────────────────────────────────────────

def _bg_transcribe_video(run_id: str, tmp_path: str, tmp_dir: str) -> None:
    """Background worker: transcribe a local video file and save to MongoDB."""
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
    """Background worker: download + transcribe a YouTube URL and save to MongoDB."""
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
    """Check which pipeline stages have been completed (based on latest run in MongoDB)."""
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
    """Get the latest transcript from MongoDB."""
    return TranscribeResponse(transcript=_require_transcript())


# ─────────────────────────────────────────────
# SEGMENTATION
# ─────────────────────────────────────────────

@app.post("/segment", response_model=SegmentResponse)
def segment():
    """
    Segment the latest transcript from MongoDB, save segments back to MongoDB.
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
    """Get the latest segments from MongoDB."""
    segments = _require_segments()
    return SegmentResponse(segments=_to_segment_out(segments))


# ─────────────────────────────────────────────
# DESCRIPTION
# ─────────────────────────────────────────────

@app.post("/describe", response_model=DescribeResponse)
def describe():
    """
    Generate content description from the latest segments in MongoDB,
    and save the description back to MongoDB.
    """
    segments    = _require_segments()
    client      = _client()
    description = generate_description(build_segments_summary(segments), client)
    db.save_description(description)
    return DescribeResponse(**description)


@app.get("/describe", response_model=DescribeResponse)
def get_description():
    """Get the latest content description from MongoDB."""
    description = db.get_description()
    if not description:
        raise HTTPException(
            status_code=404,
            detail="No description found in MongoDB. Run POST /describe first.",
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