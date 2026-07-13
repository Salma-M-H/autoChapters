"""
api.py — Full Pipeline REST API  (synchronous edition)
=========================================================
No database, no output files, no job queue. Every endpoint runs the
pipeline and returns the result directly in the same HTTP response.

NOTE: because these are synchronous, a very long-running request (e.g.
a huge file, or a slow model call) can still hit Railway's request
timeout. If that becomes a problem again, the background-job pattern
(job_id + polling) can be reintroduced.

Endpoints:

  TRANSCRIPTION  (no timestamps)
    POST /transcribe/audio             — upload audio  → returns transcript (plain text, no timestamps)
    POST /transcribe/video             — upload video  → returns transcript (plain text, no timestamps)
    POST /transcribe/youtube           — YouTube URL   → returns transcript (plain text, no timestamps)

  SEGMENTATION
    POST /segment                     — transcript in body → returns segments
    POST /segment/titles              — transcript in body → returns titles only
    POST /segment/summaries           — transcript in body → returns summaries only

  DESCRIPTION
    POST /describe                    — segments in body  → returns description

  FULL PIPELINE
    POST /pipeline/audio               — upload audio  → returns transcript + segments + description
    POST /pipeline/video               — upload video  → returns transcript + segments + description
    POST /pipeline/youtube            — YouTube URL   → returns transcript + segments + description

Run:
  uvicorn APIs:app --reload
"""

import os
import tempfile
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Video Pipeline API",
    description=(
        "Full pipeline: audio/video/YouTube → transcribe → segment → describe. "
        "All results are returned directly in the response — nothing is saved to disk or a database."
    ),
    version="9.0.0",
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
# TRANSCRIPTION
# ─────────────────────────────────────────────

# @app.post("/transcribe/audio", response_model=TranscribeResponse)
# async def transcribe_audio(file: UploadFile = File(...)):
#     """
#     Upload an audio file directly (mp3, m4a, wav, aac, ogg, etc.) — no
#     video, no extraction step. Returns the full transcript as plain text.
#     """
#     tmp_path, tmp_dir = await _save_upload(file)
#     try:
#         transcript = transcribe_from_audio_file(tmp_path, _client())
#         return TranscribeResponse(transcript=strip_timestamps(transcript))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/transcribe/video", response_model=TranscribeResponse)
async def transcribe_video(file: UploadFile = File(...)):
    """
    Upload a video file. Returns the full transcript as plain text.
    Each line is formatted as: [HH:MM:SS] spoken text
    """
    tmp_path, tmp_dir = await _save_upload(file)
    try:
        transcript = transcribe_from_video(tmp_path, _client())
        return TranscribeResponse(transcript=strip_timestamps(transcript))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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

# @app.post("/pipeline/audio", response_model=PipelineResponse)
# async def pipeline_audio(file: UploadFile = File(...)):
#     """
#     Upload an audio file directly (mp3, m4a, wav, aac, ogg, etc.) and run
#     the full pipeline end-to-end:
#       1. Transcribe
#       2. Segment into topics
#       3. Generate content description
#     Returns all three results in one response.
#     """
#     tmp_path, tmp_dir = await _save_upload(file)
#     try:
#         client      = _client()
#         transcript  = transcribe_from_audio_file(tmp_path, client)
#         segments    = segment_transcript(transcript, client)
#         description = generate_description(build_segments_summary(segments), client)
#         return PipelineResponse(
#             transcript=transcript,
#             segments=_to_segment_out(segments),
#             description=DescribeResponse(**description),
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/pipeline/video", response_model=PipelineResponse)
async def pipeline_video(file: UploadFile = File(...)):
    """
    Upload a video and run the full pipeline end-to-end:
      1. Extract audio + transcribe
      2. Segment into topics
      3. Generate content description
    Returns all three results in one response.
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