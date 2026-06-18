"""
database.py — MongoDB Storage Layer
=====================================
Replaces local file storage (transcript.txt, segments.json,
content_description.json) with a MongoDB collection.

Each pipeline run is stored as a single document:
{
  "_id":         ObjectId (auto),
  "run_id":      str  — unique per run (timestamp-based),
  "source":      "video" | "youtube",
  "source_name": str  — filename or YouTube URL,
  "created_at":  datetime,
  "transcript":  str  | None,
  "segments":    list | None,
  "description": dict | None,
}

The module also keeps a "latest" pointer so GET endpoints
(/transcript, /segments, /describe) always return the most
recent run without requiring a run_id.

Requirements:
    pip install pymongo python-dotenv
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING
from pymongo.collection import Collection

load_dotenv(Path(__file__).parent / ".env")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MONGO_URI   = os.getenv("MONGO_URI")
DB_NAME         = os.getenv("MONGO_DB_NAME")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME")


# ─────────────────────────────────────────────
# Connection (lazy singleton)
# ─────────────────────────────────────────────

_client: Optional[MongoClient] = None


def _get_collection() -> Collection:
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client[DB_NAME][COLLECTION_NAME]


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def new_run(source: str, source_name: str) -> str:
    """
    Create a new pipeline run document and return its run_id.
    Call this at the start of a pipeline run.
    """
    col    = _get_collection()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    col.insert_one({
        "run_id":      run_id,
        "source":      source,
        "source_name": source_name,
        "created_at":  datetime.now(timezone.utc),
        "transcript":  None,
        "segments":    None,
        "description": None,
    })
    return run_id


def save_transcript(transcript: str, run_id: Optional[str] = None) -> str:
    """
    Save (or update) the transcript for a run.
    If run_id is None a new run document is created with source="standalone".
    Returns the run_id.
    """
    col = _get_collection()
    if run_id is None:
        run_id = new_run("standalone", "")
    col.update_one(
        {"run_id": run_id},
        {"$set": {"transcript": transcript, "updated_at": datetime.now(timezone.utc)}},
    )
    return run_id


def save_segments(segments: list, run_id: Optional[str] = None) -> str:
    """Save (or update) segments for a run. Returns run_id."""
    col = _get_collection()
    if run_id is None:
        run_id = new_run("standalone", "")
    col.update_one(
        {"run_id": run_id},
        {"$set": {"segments": segments, "updated_at": datetime.now(timezone.utc)}},
    )
    return run_id


def save_description(description: dict, run_id: Optional[str] = None) -> str:
    """Save (or update) the description for a run. Returns run_id."""
    col = _get_collection()
    if run_id is None:
        run_id = new_run("standalone", "")
    col.update_one(
        {"run_id": run_id},
        {"$set": {"description": description, "updated_at": datetime.now(timezone.utc)}},
    )
    return run_id


# ── Getters (always return the most recent run) ──────────────

def _latest() -> Optional[dict]:
    col = _get_collection()
    return col.find_one({}, sort=[("created_at", DESCENDING)])


def get_transcript() -> Optional[str]:
    doc = _latest()
    return doc["transcript"] if doc else None


def get_segments() -> Optional[list]:
    doc = _latest()
    return doc["segments"] if doc else None


def get_description() -> Optional[dict]:
    doc = _latest()
    return doc["description"] if doc else None


def get_run(run_id: str) -> Optional[dict]:
    """Fetch a specific run by its run_id."""
    col = _get_collection()
    return col.find_one({"run_id": run_id})


def list_runs(limit: int = 20) -> list[dict]:
    """Return the most recent `limit` runs (metadata only, no transcript text)."""
    col  = _get_collection()
    docs = col.find(
        {},
        {"transcript": 0},          # exclude large transcript field
        sort=[("created_at", DESCENDING)],
        limit=limit,
    )
    result = []
    for d in docs:
        d.pop("_id", None)          # ObjectId is not JSON-serialisable
        result.append(d)
    return result


def status() -> dict:
    """Return True/False for each pipeline stage based on the latest run."""
    doc = _latest()
    if not doc:
        return {"transcript": False, "segments": False, "description": False}
    return {
        "transcript":  doc.get("transcript")  is not None,
        "segments":    doc.get("segments")    is not None,
        "description": doc.get("description") is not None,
    }