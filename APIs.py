"""
api.py — Full Pipeline REST API
=================================
Endpoints:

  STATUS
    GET  /status                      — check which files exist on server

  TRANSCRIPTION
    POST /transcribe/video            — upload a video file → transcript.txt → return transcript
    POST /transcribe/youtube          — provide YouTube URL → transcript.txt → return transcript
    GET  /transcript                  — get saved transcript.txt

  SEGMENTATION
    POST /segment                     — segment transcript.txt → segments.json → return segments
    POST /segment/titles              — segment → return titles only
    POST /segment/summaries           — segment → return summaries only
    GET  /segments                    — get saved segments.json

  DESCRIPTION
    POST /describe                    — describe segments.json → content_description.json → return description
    GET  /describe                    — get saved content_description.json

  FULL PIPELINE
    POST /pipeline/video              — upload video → all 3 steps → return full result
    POST /pipeline/youtube            — YouTube URL → all 3 steps → return full result

Run:
  uvicorn api:app --reload
"""

import os
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

from transcriber import transcribe_from_video, transcribe_from_youtube
from segmenter   import segment_transcript
from describer  import build_segments_summary, generate_description

load_dotenv(Path(__file__).parent / ".env")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GROQ_MODEL       = "llama-3.3-70b-versatile"
DIR              = Path(__file__).parent
TRANSCRIPT_FILE  = DIR / "transcript.txt"
SEGMENTS_FILE    = DIR / "segments.json"
DESCRIPTION_FILE = DIR / "content_description.json"


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
    transcript: str
    segments: list[SegmentOut]
    description: DescribeResponse


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Video Pipeline API",
    description="Full pipeline: video/YouTube → transcribe → segment → describe.",
    version="4.0.0",
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


def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found. {hint}")


def _save_transcript(transcript: str) -> None:
    TRANSCRIPT_FILE.write_text(transcript, encoding="utf-8")


def _save_segments(segments: list[dict]) -> None:
    SEGMENTS_FILE.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_description(description: dict) -> None:
    DESCRIPTION_FILE.write_text(json.dumps(description, ensure_ascii=False, indent=2), encoding="utf-8")


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


# ─────────────────────────────────────────────
# STATUS
# ─────────────────────────────────────────────

@app.get("/status", response_model=StatusResponse)
def get_status():
    """Check which pipeline output files exist on the server."""
    return StatusResponse(
        transcript=TRANSCRIPT_FILE.exists(),
        segments=SEGMENTS_FILE.exists(),
        description=DESCRIPTION_FILE.exists(),
    )


# ─────────────────────────────────────────────
# TRANSCRIPTION
# ─────────────────────────────────────────────

@app.post("/transcribe/video", response_model=TranscribeResponse)
async def transcribe_video(file: UploadFile = File(...)):
    """
    Upload a video file, extract its audio, and transcribe it.
    Saves transcript.txt and returns the transcript.
    """
    tmp_path = await _save_upload(file)
    tmp_dir  = os.path.dirname(tmp_path)
    try:
        transcript = transcribe_from_video(tmp_path, _client())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _save_transcript(transcript)
    return TranscribeResponse(transcript=transcript)


@app.post("/transcribe/youtube", response_model=TranscribeResponse)
def transcribe_youtube(body: YoutubeRequest):
    """
    Provide a YouTube URL, download its audio, and transcribe it.
    Saves transcript.txt and returns the transcript.
    """
    transcript = transcribe_from_youtube(body.url, _client())
    _save_transcript(transcript)
    return TranscribeResponse(transcript=transcript)


@app.get("/transcript", response_model=TranscribeResponse)
def get_transcript():
    """Get the saved transcript.txt content."""
    _require_file(TRANSCRIPT_FILE, "Run POST /transcribe/video or /transcribe/youtube first.")
    return TranscribeResponse(transcript=TRANSCRIPT_FILE.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────
# SEGMENTATION
# ─────────────────────────────────────────────

@app.post("/segment", response_model=SegmentResponse)
def segment():
    """
    Segment transcript.txt by topic.
    Saves segments.json and returns all segments.
    """
    _require_file(TRANSCRIPT_FILE, "Run POST /transcribe/video or /transcribe/youtube first.")
    transcript = TRANSCRIPT_FILE.read_text(encoding="utf-8")
    segments   = segment_transcript(transcript, _client())
    _save_segments(segments)
    return SegmentResponse(segments=_to_segment_out(segments))


@app.post("/segment/titles", response_model=TitlesResponse)
def segment_titles():
    """Segment transcript.txt and return only the topic titles."""
    _require_file(TRANSCRIPT_FILE, "Run POST /transcribe/video or /transcribe/youtube first.")
    transcript = TRANSCRIPT_FILE.read_text(encoding="utf-8")
    segments   = segment_transcript(transcript, _client())
    _save_segments(segments)
    return TitlesResponse(titles=[seg["title"] for seg in segments])


@app.post("/segment/summaries", response_model=SummariesResponse)
def segment_summaries():
    """Segment transcript.txt and return only the summaries."""
    _require_file(TRANSCRIPT_FILE, "Run POST /transcribe/video or /transcribe/youtube first.")
    transcript = TRANSCRIPT_FILE.read_text(encoding="utf-8")
    segments   = segment_transcript(transcript, _client())
    _save_segments(segments)
    return SummariesResponse(
        summaries=[
            {"index": seg["index"], "title": seg["title"], "summary": seg["summary"]}
            for seg in segments
        ]
    )


@app.get("/segments", response_model=SegmentResponse)
def get_segments():
    """Get the saved segments.json content."""
    _require_file(SEGMENTS_FILE, "Run POST /segment first.")
    data = json.loads(SEGMENTS_FILE.read_text(encoding="utf-8"))
    return SegmentResponse(segments=[SegmentOut(**s) for s in data])


# ─────────────────────────────────────────────
# DESCRIPTION
# ─────────────────────────────────────────────

@app.post("/describe", response_model=DescribeResponse)
def describe():
    """
    Generate content description from segments.json.
    Saves content_description.json and returns the description.
    """
    _require_file(SEGMENTS_FILE, "Run POST /segment first.")
    segments    = json.loads(SEGMENTS_FILE.read_text(encoding="utf-8"))
    client      = _client()
    description = generate_description(build_segments_summary(segments), client)
    _save_description(description)
    return DescribeResponse(**description)


@app.get("/describe", response_model=DescribeResponse)
def get_description():
    """Get the saved content_description.json content."""
    _require_file(DESCRIPTION_FILE, "Run POST /describe first.")
    return DescribeResponse(**json.loads(DESCRIPTION_FILE.read_text(encoding="utf-8")))


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────

@app.post("/pipeline/video", response_model=PipelineResponse)
async def pipeline_video(file: UploadFile = File(...)):
    """
    Upload a video file and run the full pipeline:
      1. Extract audio + transcribe → transcript.txt
      2. Segment → segments.json
      3. Describe → content_description.json
    Returns everything.
    """
    client   = _client()
    tmp_path = await _save_upload(file)
    tmp_dir  = os.path.dirname(tmp_path)

    try:
        transcript = transcribe_from_video(tmp_path, client)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _save_transcript(transcript)

    segments = segment_transcript(transcript, client)
    _save_segments(segments)

    description = generate_description(build_segments_summary(segments), client)
    _save_description(description)

    return PipelineResponse(
        transcript=transcript,
        segments=_to_segment_out(segments),
        description=DescribeResponse(**description),
    )


@app.post("/pipeline/youtube", response_model=PipelineResponse)
def pipeline_youtube(body: YoutubeRequest):
    """
    Provide a YouTube URL and run the full pipeline:
      1. Download audio + transcribe → transcript.txt
      2. Segment → segments.json
      3. Describe → content_description.json
    Returns everything.
    """
    client     = _client()
    transcript = transcribe_from_youtube(body.url, client)
    _save_transcript(transcript)

    segments = segment_transcript(transcript, client)
    _save_segments(segments)

    description = generate_description(build_segments_summary(segments), client)
    _save_description(description)

    return PipelineResponse(
        transcript=transcript,
        segments=_to_segment_out(segments),
        description=DescribeResponse(**description),
    )