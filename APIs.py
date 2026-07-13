"""
api.py — Full Pipeline REST API
=========================================================
No database, no output files (job results live in memory only).

VIDEO endpoints (/transcribe/video, /pipeline/video) are long-running
(download/decode + Groq calls), which is what was hitting Railway's
request timeout. These now run as BACKGROUND JOBS:
  1. The endpoint saves the upload, queues the work, and immediately
     returns 202 with a job_id.
  2. Poll GET /jobs/{job_id} for status ("pending" | "processing" |
     "completed" | "failed") and the result once it's done.

YOUTUBE endpoints (/transcribe/youtube, /pipeline/youtube) are left
synchronous since they were reported to work fine as-is.

NOTE on scaling: job state is kept in an in-process dict. This is fine
for a single Railway instance/worker. If you scale to multiple workers
or replicas, swap `_jobs` for Redis (or a DB table) so all workers can
see the same job state — otherwise a poll can land on a worker that
never ran the job.

Endpoints:

  TRANSCRIPTION  (no timestamps)
    POST /transcribe/audio             — upload audio  → 202 + job_id (poll /jobs/{job_id}) — PREFERRED, smaller upload
    POST /transcribe/video             — upload video  → 202 + job_id (poll /jobs/{job_id})
    POST /transcribe/youtube           — YouTube URL   → returns transcript (plain text, no timestamps)

  SEGMENTATION
    POST /segment                     — transcript in body → returns segments
    POST /segment/titles              — transcript in body → returns titles only
    POST /segment/summaries           — transcript in body → returns summaries only

  DESCRIPTION
    POST /describe                    — segments in body  → returns description

  FULL PIPELINE
    POST /pipeline/audio               — upload audio  → 202 + job_id (poll /jobs/{job_id}) — PREFERRED, smaller upload
    POST /pipeline/video               — upload video  → 202 + job_id (poll /jobs/{job_id})
    POST /pipeline/video/sync         — BENCHMARK ONLY: same pipeline, blocks until done
    POST /pipeline/youtube            — YouTube URL   → returns transcript + segments + description

  JOBS
    GET  /jobs/{job_id}                — poll status/result for a queued video job

Run:
  uvicorn APIs:app --reload
"""

import os
import tempfile
import shutil
import uuid
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq

from transcriber import transcribe_from_video, transcribe_from_youtube, transcribe_from_audio_file
from segmenter   import segment_transcript
from describer   import build_segments_summary, generate_description

load_dotenv(Path(__file__).parent / ".env")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

import re as _re
_TIMESTAMP_RE = _re.compile(r"^\[\d{2}:\d{2}:\d{2}\]\s*", _re.MULTILINE)

def strip_timestamps(transcript: str) -> str:
    """Remove [HH:MM:SS] prefixes from every line of the transcript."""
    return _TIMESTAMP_RE.sub("", transcript)


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

class YoutubeRequest(BaseModel):
    url: str

class TranscriptRequest(BaseModel):
    transcript: str

class SegmentsRequest(BaseModel):
    segments: list[dict]

class SegmentOut(BaseModel):
    index: int
    title: str
    summary: str
    text: str
    start_time: str
    end_time: str

class TranscribeResponse(BaseModel):
    transcript: str

class SegmentResponse(BaseModel):
    segments: list[SegmentOut]

class TitlesResponse(BaseModel):
    titles: list[str]

class SummariesResponse(BaseModel):
    summaries: list[dict]

class DescribeResponse(BaseModel):
    summary: str
    target_audience: str
    tone_and_style: str
    seo_tags: list[str]

class PipelineResponse(BaseModel):
    transcript: str
    segments: list[SegmentOut]
    description: DescribeResponse

class JobQueuedResponse(BaseModel):
    job_id: str
    status: str = "pending"

class JobStatusResponse(BaseModel):
    job_id: str
    status: str                     # "pending" | "processing" | "completed" | "failed"
    result: Optional[dict] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Video Pipeline API",
    description=(
        "Full pipeline: video/YouTube → transcribe → segment → describe. "
        "All results are returned directly in the response — nothing is saved to disk or a database."
    ),
    version="8.0.0",
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


async def _save_upload(file: UploadFile) -> tuple[str, str]:
    """Save uploaded file to a temp dir. Returns (file_path, tmp_dir)."""
    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)
    content  = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(content)
    return tmp_path, tmp_dir


# ─────────────────────────────────────────────
# Background job store (in-memory — see note in module docstring)
# ─────────────────────────────────────────────

_jobs: dict[str, dict] = {}
_jobs_lock = Lock()

# Caps how many video jobs run at once. Keep this low — each job hits
# ffmpeg + Groq, so unbounded concurrency here just recreates the
# resource-contention problem, not fixes it.
_job_executor = ThreadPoolExecutor(max_workers=2)


def _set_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs[job_id].update(fields)


def _get_job(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _run_transcribe_video_job(job_id: str, tmp_path: str, tmp_dir: str) -> None:
    _set_job(job_id, status="processing")
    try:
        transcript = transcribe_from_video(tmp_path, _client())
        result = TranscribeResponse(transcript=strip_timestamps(transcript))
        _set_job(job_id, status="completed", result=result.model_dump())
    except Exception as e:
        _set_job(job_id, status="failed", error=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_transcribe_audio_job(job_id: str, audio_path: str, tmp_dir: str) -> None:
    _set_job(job_id, status="processing")
    try:
        transcript = transcribe_from_audio_file(audio_path, _client())
        result = TranscribeResponse(transcript=strip_timestamps(transcript))
        _set_job(job_id, status="completed", result=result.model_dump())
    except Exception as e:
        _set_job(job_id, status="failed", error=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_pipeline_audio_job(job_id: str, audio_path: str, tmp_dir: str) -> None:
    _set_job(job_id, status="processing")
    try:
        client      = _client()
        transcript  = transcribe_from_audio_file(audio_path, client)
        segments    = segment_transcript(transcript, client)
        description = generate_description(build_segments_summary(segments), client)
        result = PipelineResponse(
            transcript=transcript,
            segments=_to_segment_out(segments),
            description=DescribeResponse(**description),
        )
        _set_job(job_id, status="completed", result=result.model_dump())
    except Exception as e:
        _set_job(job_id, status="failed", error=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_pipeline_video_job(job_id: str, tmp_path: str, tmp_dir: str) -> None:
    _set_job(job_id, status="processing")
    try:
        client      = _client()
        transcript  = transcribe_from_video(tmp_path, client)
        segments    = segment_transcript(transcript, client)
        description = generate_description(build_segments_summary(segments), client)
        result = PipelineResponse(
            transcript=transcript,
            segments=_to_segment_out(segments),
            description=DescribeResponse(**description),
        )
        _set_job(job_id, status="completed", result=result.model_dump())
    except Exception as e:
        _set_job(job_id, status="failed", error=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────
# JOBS
# ─────────────────────────────────────────────

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    """
    Poll the status of a queued video job.
    status: "pending" (queued, not started) | "processing" (running) |
            "completed" (result populated) | "failed" (error populated)
    """
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(**job)


# ─────────────────────────────────────────────
# TRANSCRIPTION
# ─────────────────────────────────────────────

@app.post("/transcribe/video", response_model=JobQueuedResponse, status_code=202)
async def transcribe_video(file: UploadFile = File(...)):
    """
    Upload a video file. Queues transcription as a background job and
    returns immediately with a job_id. Poll GET /jobs/{job_id} — once
    status is "completed", result.transcript holds the plain-text
    transcript (no [HH:MM:SS] prefixes).
    """
    tmp_path, tmp_dir = await _save_upload(file)
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"job_id": job_id, "status": "pending", "result": None, "error": None}
    _job_executor.submit(_run_transcribe_video_job, job_id, tmp_path, tmp_dir)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "pending"})


@app.post("/transcribe/audio", response_model=JobQueuedResponse, status_code=202)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Upload an AUDIO file directly (mp3, m4a, wav, aac, ogg, etc.) — no
    video, no extraction step. Use this instead of /transcribe/video
    whenever the client can send audio only: it's a much smaller upload,
    which avoids the slow-connection timeout issue large video uploads
    can hit on Railway.

    Queues transcription as a background job and returns immediately
    with a job_id. Poll GET /jobs/{job_id} — once status is "completed",
    result.transcript holds the plain-text transcript (no timestamps).
    """
    tmp_path, tmp_dir = await _save_upload(file)
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"job_id": job_id, "status": "pending", "result": None, "error": None}
    _job_executor.submit(_run_transcribe_audio_job, job_id, tmp_path, tmp_dir)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "pending"})


@app.post("/transcribe/youtube", response_model=TranscribeResponse)
async def transcribe_youtube(body: YoutubeRequest):
    """
    Provide a YouTube URL. Downloads the video via the SocialKit API,
    extracts the audio, and returns the full transcript as plain text.
    Each line is formatted as: [HH:MM:SS] spoken text
    """
    try:
        transcript = transcribe_from_youtube(body.url, _client())
        return TranscribeResponse(transcript=strip_timestamps(transcript))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# # ─────────────────────────────────────────────
# # SEGMENTATION
# # ─────────────────────────────────────────────

# @app.post("/segment", response_model=SegmentResponse)
# async def segment(body: TranscriptRequest):
#     """
#     Accepts a transcript string and returns the segmented topics.
#     Pass the transcript text you got from POST /transcribe/*.
#     """
#     try:
#         segments = segment_transcript(body.transcript, _client())
#         return SegmentResponse(segments=_to_segment_out(segments))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/segment/titles", response_model=TitlesResponse)
# async def segment_titles(body: TranscriptRequest):
#     """
#     Accepts a transcript string and returns segment titles only.
#     """
#     try:
#         segments = segment_transcript(body.transcript, _client())
#         return TitlesResponse(titles=[seg["title"] for seg in segments])
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/segment/summaries", response_model=SummariesResponse)
# async def segment_summaries(body: TranscriptRequest):
#     """
#     Accepts a transcript string and returns segment summaries only.
#     """
#     try:
#         segments = segment_transcript(body.transcript, _client())
#         return SummariesResponse(
#             summaries=[
#                 {"index": seg["index"], "title": seg["title"], "summary": seg["summary"]}
#                 for seg in segments
#             ]
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # ─────────────────────────────────────────────
# # DESCRIPTION
# # ─────────────────────────────────────────────

# @app.post("/describe", response_model=DescribeResponse)
# async def describe(body: SegmentsRequest):
#     """
#     Accepts a list of segments (as returned by POST /segment) and returns
#     a structured content description: summary, target audience, tone, SEO tags.
#     """
#     try:
#         client      = _client()
#         description = generate_description(build_segments_summary(body.segments), client)
#         return DescribeResponse(**description)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────

@app.post("/pipeline/video", response_model=JobQueuedResponse, status_code=202)
async def pipeline_video(file: UploadFile = File(...)):
    """
    Upload a video and queue the full pipeline (transcribe → segment →
    describe) as a background job. Returns immediately with a job_id.
    Poll GET /jobs/{job_id} — once status is "completed", result holds
    transcript + segments + description, matching PipelineResponse.
    """
    tmp_path, tmp_dir = await _save_upload(file)
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"job_id": job_id, "status": "pending", "result": None, "error": None}
    _job_executor.submit(_run_pipeline_video_job, job_id, tmp_path, tmp_dir)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "pending"})


@app.post("/pipeline/audio", response_model=JobQueuedResponse, status_code=202)
async def pipeline_audio(file: UploadFile = File(...)):
    """
    Upload an AUDIO file directly (mp3, m4a, wav, aac, ogg, etc.) and
    queue the full pipeline (transcribe → segment → describe) as a
    background job. Prefer this over /pipeline/video when the client can
    send audio only — the smaller upload avoids the slow-connection
    timeout issue large video uploads can hit on Railway.

    Returns immediately with a job_id. Poll GET /jobs/{job_id} — once
    status is "completed", result holds transcript + segments +
    description, matching PipelineResponse.
    """
    tmp_path, tmp_dir = await _save_upload(file)
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"job_id": job_id, "status": "pending", "result": None, "error": None}
    _job_executor.submit(_run_pipeline_audio_job, job_id, tmp_path, tmp_dir)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "pending"})


@app.post("/pipeline/video/sync", response_model=PipelineResponse)
async def pipeline_video_sync(file: UploadFile = File(...)):
    """
    BENCHMARK-ONLY. Runs the exact same pipeline as /pipeline/video, but
    synchronously — blocks until the full result is ready, like the
    original (pre-job-queue) endpoint did.

    This exists so benchmark.py can measure sync vs. job-based timing
    side by side. Do NOT use this in production on Railway — it's the
    endpoint that was hitting the request timeout in the first place.
    """
    tmp_path, tmp_dir = await _save_upload(file)
    try:
        client      = _client()
        transcript  = transcribe_from_video(tmp_path, client)
        segments    = segment_transcript(transcript, client)
        description = generate_description(build_segments_summary(segments), client)
        return PipelineResponse(
            transcript=transcript,
            segments=_to_segment_out(segments),
            description=DescribeResponse(**description),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/pipeline/youtube", response_model=PipelineResponse)
async def pipeline_youtube(body: YoutubeRequest):
    """
    Provide a YouTube URL and run the full pipeline end-to-end:
      1. Download video via SocialKit, extract audio, and transcribe
      2. Segment into topics
      3. Generate content description
    Returns all three results in one response.
    """
    try:
        client      = _client()
        transcript  = transcribe_from_youtube(body.url, client)
        segments    = segment_transcript(transcript, client)
        description = generate_description(build_segments_summary(segments), client)
        return PipelineResponse(
            transcript=transcript,
            segments=_to_segment_out(segments),
            description=DescribeResponse(**description),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))