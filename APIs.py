from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

from autoChapters import (
    segment_by_topic,
    cosine_segment,
    call_openrouter,
    build_prompt,
    parse_response,
    apply_grouping,
    is_arabic,
    estimate_tokens,
    get_labels,
)
import autoChapters as pipeline

app = FastAPI(
    title="Topic Segmentation API",
    description="API لتقسيم النصوص العربية والإنجليزية حسب الموضوع",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load API key from environment ──────────────────────────────────────────────
pipeline.OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


# ── Request models ─────────────────────────────────────────────────────────────

class SegmentRequest(BaseModel):
    text: str
    model: Optional[str] = "openai/gpt-oss-20b"
    min_segment_sentences: Optional[int] = 2
    arabic_chunk_words: Optional[int] = 30

class CosineRequest(BaseModel):
    text: str
    min_segment_sentences: Optional[int] = 2
    arabic_chunk_words: Optional[int] = 30

class GroupRequest(BaseModel):
    segments: List[str]
    model: Optional[str] = "openai/gpt-oss-20b"


# ── Helper ─────────────────────────────────────────────────────────────────────

def apply_config(req):
    if req.min_segment_sentences:
        pipeline.MIN_SEGMENT_SENTENCES = req.min_segment_sentences
    if hasattr(req, "arabic_chunk_words") and req.arabic_chunk_words:
        pipeline.ARABIC_SPEECH_CHUNK_WORDS = req.arabic_chunk_words
    if req.model:
        pipeline.MODEL = req.model


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Topic Segmentation API is running"}


@app.post("/segment")
def segment(req: SegmentRequest):
    if not pipeline.OPENROUTER_API_KEY:
        raise HTTPException(status_code=401, detail="OPENROUTER_API_KEY غير موجود في البيئة")
    apply_config(req)
    try:
        results = segment_by_topic(req.text, verbose=False)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "language": "Arabic" if is_arabic(req.text) else "English",
        "segments_count": len(results),
        "segments": results,
        "labels": get_labels(results),
    }


@app.post("/segment/cosine")
def segment_cosine(req: CosineRequest):
    apply_config(req)
    segments = cosine_segment(req.text)
    return {
        "language": "Arabic" if is_arabic(req.text) else "English",
        "segments_count": len(segments),
        "segments": segments,
    }


@app.post("/segment/group")
def segment_group(req: GroupRequest):
    if not pipeline.OPENROUTER_API_KEY:
        raise HTTPException(status_code=401, detail="OPENROUTER_API_KEY غير موجود في البيئة")
    if req.model:
        pipeline.MODEL = req.model

    arabic = is_arabic(" ".join(req.segments))
    prompt = build_prompt(req.segments, arabic=arabic)

    if estimate_tokens(prompt) > 100_000:
        raise HTTPException(status_code=422, detail="المقاطع كبيرة جداً، قلل حجم النص")

    try:
        raw       = call_openrouter(prompt)
        decisions = parse_response(raw)
        results   = apply_grouping(req.segments, decisions)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "segments_count": len(results),
        "segments": results,
        "labels": get_labels(results),
    }