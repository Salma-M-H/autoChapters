"""
Transcription Segmentation Pipeline
=====================================
Stage 1: Sentence-transformers embeddings
Stage 2: Cosine similarity boundary detection (time-ordered)
Stage 3: Groq LLM — merge, label, and summarize segments

HOW TO USE:
  1. Copy your transcript (.txt) into the same folder as this script
  2. Fill in your GROQ_API_KEY in the .env file
  3. Edit the CONFIG section below if needed
  4. Run: python pipeline.py
"""

import os
import re
import json
import textwrap
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load .env file (looks for .env in the same folder as this script)
load_dotenv()


# ─────────────────────────────────────────────
# CONFIG — edit these values directly
# ─────────────────────────────────────────────

# Name of your transcript file (must be in the same folder as this script)

TRANSCRIPT_FILE = "transcription.txt"

# Groq model to use for merging, labelling, and summarising
# Options: "llama-3.3-70b-versatile" | "llama-3.1-8b-instant" | "mixtral-8x7b-32768"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Sentence-transformers embedding model (runs locally, no API needed)
# Options: "all-MiniLM-L6-v2" | "all-mpnet-base-v2" | "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Number of sentences grouped into one chunk before embedding
# Higher = smoother, fewer segments. Lower = more granular.
SENTENCE_WINDOW = 3

# Boundary detection sensitivity (leave as None for automatic)
# Lower value = more topic boundaries detected. Try 0.1 to 0.3 to tune manually.
BOUNDARY_THRESHOLD = None

# Where to save the results JSON (set to None to skip saving)
OUTPUT_FILE = "results.json"

# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class RawSegment:
    """A segment produced by cosine-similarity boundary detection."""
    index: int
    sentences: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    def __repr__(self):
        preview = self.text[:80].replace("\n", " ")
        return f"RawSegment(index={self.index}, sentences={len(self.sentences)}, preview='{preview}...')"


@dataclass
class TopicSegment:
    """A final segment after Groq merging, labelling, and summarising."""
    index: int
    title: str
    summary: str
    text: str
    source_segment_indices: list[int] = field(default_factory=list)


# ─────────────────────────────────────────────
# Stage 1 — Text splitting
# ─────────────────────────────────────────────

def split_into_sentences(transcript: str) -> list[str]:
    """
    Split transcript into individual sentences or speaker turns.
    If the text has multiple lines (e.g. speaker-turn format), splits by line.
    Otherwise splits on sentence-ending punctuation.
    """
    lines = [l.strip() for l in transcript.splitlines() if l.strip()]
    if len(lines) > 5:
        return lines

    sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentences(sentences: list[str], window: int = 3) -> list[str]:
    """
    Group consecutive sentences into chunks of size `window`.
    Larger window = richer context per embedding.
    """
    if window <= 1:
        return sentences
    chunks = []
    for i in range(0, len(sentences), window):
        chunk = " ".join(sentences[i : i + window])
        chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────
# Stage 2 — Cosine similarity segmentation
# ─────────────────────────────────────────────

def embed_chunks(chunks: list[str], model_name: str) -> np.ndarray:
    """Embed chunks using a local sentence-transformers model."""
    print(f"  Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print(f"  Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def compute_similarity_scores(embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity between each consecutive pair of chunk embeddings."""
    scores = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]
        scores.append(float(sim))
    return np.array(scores)


def detect_boundaries(
    similarity_scores: np.ndarray,
    threshold: Optional[float] = None,
    depth_score_k: int = 1,
) -> list[int]:
    """
    Find topic boundary positions using depth scoring.

    A "depth score" measures how much of a valley each position is:
        depth(i) = (peak to the left - value at i) + (peak to the right - value at i)

    Positions whose depth exceeds the cutoff are marked as boundaries.
    """
    n = len(similarity_scores)
    depths = np.zeros(n)

    for i in range(depth_score_k, n - depth_score_k):
        left  = max(similarity_scores[i - depth_score_k : i])
        right = max(similarity_scores[i + 1 : i + depth_score_k + 1])
        depths[i] = (left - similarity_scores[i]) + (right - similarity_scores[i])

    cutoff = threshold if threshold is not None else (depths.mean() + depths.std())
    boundaries = [i for i in range(depth_score_k, n - depth_score_k) if depths[i] > cutoff]
    return boundaries


def build_raw_segments(chunks: list[str], boundaries: list[int]) -> list[RawSegment]:
    """Split chunks into RawSegments at each detected boundary."""
    segments = []
    prev = 0
    for i, boundary in enumerate(sorted(boundaries)):
        seg_chunks = chunks[prev : boundary + 1]
        if seg_chunks:
            segments.append(RawSegment(index=i, sentences=seg_chunks))
        prev = boundary + 1

    remaining = chunks[prev:]
    if remaining:
        segments.append(RawSegment(index=len(segments), sentences=remaining))

    return segments


# ─────────────────────────────────────────────
# Stage 3 — Groq LLM merging / labelling
# ─────────────────────────────────────────────

GROQ_SYSTEM_PROMPT = """\
You are an expert transcript analyst. You receive a list of raw text segments \
from a transcript (already in chronological order). Your task is to:

1. MERGE any consecutive segments that belong to the same topic.
2. LABEL each final segment with a concise topic title (8 words or less).
3. SUMMARIZE each final segment in 1-3 sentences.

Return ONLY a valid JSON array (no markdown fences, no commentary) like:
[
  {
    "title": "...",
    "summary": "...",
    "source_indices": [0, 1],
    "text": "..."
  },
  ...
]

Rules:
- Preserve chronological order strictly.
- Keep `text` as the concatenation of merged segment texts.
- `source_indices` lists the original segment indices you merged.
- Do NOT hallucinate content. Only use what is in the segments.
"""


def call_groq(
    raw_segments: list[RawSegment],
    groq_api_key: str,
    model: str,
) -> list[TopicSegment]:
    """Call Groq to merge, label, and summarise raw segments."""
    client = Groq(api_key=groq_api_key)

    user_message = "\n\n---\n\n".join(
        f"[Segment {seg.index}]\n{seg.text}" for seg in raw_segments
    )

    print(f"  Calling Groq model '{model}' with {len(raw_segments)} raw segments...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GROQ_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
        max_tokens=4096,
    )

    raw_json = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
        raw_json = raw_json.strip()

    data = json.loads(raw_json)

    return [
        TopicSegment(
            index=i,
            title=item.get("title", f"Topic {i+1}"),
            summary=item.get("summary", ""),
            text=item.get("text", ""),
            source_segment_indices=item.get("source_indices", []),
        )
        for i, item in enumerate(data)
    ]


# ─────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────

def print_results(topic_segments: list[TopicSegment]) -> None:
    width = 80
    print("\n" + "=" * width)
    print("  SEGMENTATION RESULTS")
    print("=" * width)
    for seg in topic_segments:
        print(f"\n{'-' * width}")
        print(f"  [{seg.index + 1}]  {seg.title.upper()}")
        print(f"{'-' * width}")
        print(f"  SUMMARY: {seg.summary}")
        print()
        wrapped = textwrap.fill(
            seg.text, width=width - 4,
            initial_indent="  ", subsequent_indent="  "
        )
        print(wrapped)
    print("\n" + "=" * width)


def save_results(topic_segments: list[TopicSegment], output_path: str) -> None:
    data = [
        {
            "index": seg.index,
            "title": seg.title,
            "summary": seg.summary,
            "text": seg.text,
            "source_segment_indices": seg.source_segment_indices,
        }
        for seg in topic_segments
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved -> {output_path}")


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_pipeline() -> list[TopicSegment]:
    # Load API key from .env
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Make sure it is set in your .env file.\n"
            "Example .env content:  GROQ_API_KEY=your_key_here"
        )

    # Resolve transcript path (same folder as this script)
    script_dir = Path(__file__).parent
    transcript_path = script_dir / TRANSCRIPT_FILE
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"Transcript file not found: {transcript_path}\n"
            f"Place your .txt file in the same folder as pipeline.py "
            f"and update TRANSCRIPT_FILE at the top of the script."
        )

    print(f"\nReading transcript: {transcript_path}")
    transcript = transcript_path.read_text(encoding="utf-8")

    # Stage 1: Split & chunk
    print("\n[Stage 1] Splitting transcript into chunks...")
    sentences = split_into_sentences(transcript)
    print(f"  {len(sentences)} sentences/lines found.")
    chunks = chunk_sentences(sentences, window=SENTENCE_WINDOW)
    print(f"  Grouped into {len(chunks)} chunks (window={SENTENCE_WINDOW}).")

    # Stage 2: Embed + boundary detection
    print("\n[Stage 2] Embedding & boundary detection...")
    embeddings   = embed_chunks(chunks, model_name=EMBEDDING_MODEL)
    sim_scores   = compute_similarity_scores(embeddings)
    boundaries   = detect_boundaries(sim_scores, threshold=BOUNDARY_THRESHOLD)
    print(f"  Detected {len(boundaries)} boundaries -> {len(boundaries)+1} raw segments.")
    raw_segments = build_raw_segments(chunks, boundaries)

    # Stage 3: Groq merge + label + summarise
    print("\n[Stage 3] Groq LLM — merge, label & summarise...")
    topic_segments = call_groq(raw_segments, groq_api_key, model=GROQ_MODEL)
    print(f"  Produced {len(topic_segments)} final topic segments.")

    # Output
    print_results(topic_segments)
    if OUTPUT_FILE:
        output_path = script_dir / OUTPUT_FILE
        save_results(topic_segments, str(output_path))

    return topic_segments


if __name__ == "__main__":
    run_pipeline()