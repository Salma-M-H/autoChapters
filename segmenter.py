#!/usr/bin/env python3
"""
segment.py  —  Step 2 of 3
============================
Reads transcript.txt (output of transcribe.py), segments it into
coherent topics using Groq LLM, and attaches real timestamps to each segment.

Handles long transcripts that exceed Groq's token limit by splitting into
overlapping chunks, segmenting each independently, then merging the results.

Output:
    segments.json  — array of topic segments, each with:
        {
          "index":       0,
          "title":       "...",
          "summary":     "...",
          "start_time":  "00:01:23",
          "end_time":    "00:04:11",
          "text":        "..."
        }

Requirements:
    pip install groq python-dotenv
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).parent / ".env")

# ============================================================
#  CONFIG
# ============================================================
TRANSCRIPT_FILE  = "transcript.txt"
OUTPUT_FILE      = "segments.json"
GROQ_MODEL       = "llama-3.3-70b-versatile"

# Each chunk sent to Groq will have at most this many lines.
# At ~15 tokens/line, 150 lines ≈ 2250 tokens + ~300 prompt = safely under the 12k TPM limit.
MAX_LINES_PER_CHUNK = 150

# Overlap between consecutive chunks so topics that span a boundary
# are not split in half. Must be less than MAX_LINES_PER_CHUNK.
OVERLAP_LINES = 20
# ============================================================


# ── Prompts ──────────────────────────────────────────────────

SEGMENTATION_PROMPT = """\
You are an expert transcript analyst. You will receive a portion of a transcript
where every line is prefixed with its ORIGINAL line number, like:

120: [00:04:05] Hello everyone.
121: [00:04:08] Today we discuss the budget.

Your task:
1. Segment this portion into coherent topic sections in chronological order.
2. Give each segment a concise title (8 words or less).
3. Write a 1-3 sentence summary for each segment.
4. Return the START and END line numbers using the ORIGINAL numbers shown above.

Return ONLY a valid JSON array — no markdown fences, no commentary:
[
  { "title": "...", "summary": "...", "start_line": 120, "end_line": 135 },
  ...
]

Rules:
- Use the exact line numbers shown in the input — do NOT renumber from 0.
- Every line in this portion must be covered — no gaps.
- First segment starts at the first line shown; last segment ends at the last line shown.
- Produce as many segments as needed — do not merge unrelated topics.
- IMPORTANT: The transcript is in Arabic. You MUST write ALL titles and summaries in Arabic. Do NOT use English under any circumstances.
"""

GAP_RESOLUTION_PROMPT = """\
You are an expert transcript analyst. Gaps were found in a segmentation.
For each gap you receive the last few lines of the previous segment,
the gap lines, and the first few lines of the next segment.

Decide for each gap:
- "previous" → gap continues the previous segment
- "next"     → gap starts the next segment
- "new"      → gap is a separate topic (provide title and summary)

Return ONLY a valid JSON array — no markdown fences, no commentary:
[
  { "gap_id": 0, "belongs_to": "previous"|"next"|"new", "title": "...", "summary": "..." },
  ...
]

Rules:
- "title" and "summary" only required when belongs_to is "new", otherwise set to "".
- IMPORTANT: The transcript is in Arabic. You MUST respond with titles and summaries in Arabic only. Do NOT use English.
"""

MERGE_BOUNDARY_PROMPT = """\
You are an expert transcript analyst. Two consecutive transcript chunks were
segmented independently. Where they overlap, the last segment of chunk A and
the first segment of chunk B may cover the same topic.

Given:
- Last segment of chunk A: title, summary, end_line
- First segment of chunk B: title, summary, start_line

Decide:
- "merge"  → they are the same topic (combine into one segment)
- "keep"   → they are different topics (keep both)

Return ONLY a valid JSON object — no markdown fences, no commentary:
{ "decision": "merge"|"keep", "title": "...", "summary": "..." }

Rules:
- If merging, provide a combined title and summary.
- If keeping, title and summary can be empty strings.
- IMPORTANT: The transcript is in Arabic. You MUST respond with titles and summaries in Arabic only. Do NOT use English.
"""


# ── Helpers ───────────────────────────────────────────────────

TIMESTAMP_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]")


def parse_lines(transcript: str) -> list[str]:
    return [l for l in transcript.splitlines() if l.strip()]


def extract_timestamp(line: str) -> str | None:
    m = TIMESTAMP_RE.match(line.strip())
    return m.group(1) if m else None


def clean_line(line: str) -> str:
    """Strip the [HH:MM:SS] timestamp prefix and return clean text."""
    return TIMESTAMP_RE.sub("", line).strip()


def parse_json_list(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def parse_json_obj(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def slice_lines(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[max(0, start) : min(len(lines) - 1, end) + 1])


def numbered_chunk(lines: list[str], start_idx: int, end_idx: int) -> str:
    """Return lines[start_idx..end_idx] each prefixed with its ORIGINAL index."""
    return "\n".join(
        f"{i}: {lines[i]}" for i in range(start_idx, min(end_idx + 1, len(lines)))
    )


# ── Gap detection & resolution ────────────────────────────────

def find_gaps(data: list[dict], first_line: int, last_line: int) -> list[dict]:
    data = sorted(data, key=lambda x: x["start_line"])
    gaps, gid = [], 0

    if data[0]["start_line"] > first_line:
        gaps.append({"gap_id": gid, "start_line": first_line, "end_line": data[0]["start_line"] - 1})
        gid += 1

    for i in range(len(data) - 1):
        if data[i + 1]["start_line"] > data[i]["end_line"] + 1:
            gaps.append({"gap_id": gid,
                         "start_line": data[i]["end_line"] + 1,
                         "end_line":   data[i + 1]["start_line"] - 1})
            gid += 1

    if data[-1]["end_line"] < last_line:
        gaps.append({"gap_id": gid,
                     "start_line": data[-1]["end_line"] + 1,
                     "end_line":   last_line})

    return gaps


def build_gap_message(gaps, data, lines, ctx=3):
    data  = sorted(data, key=lambda x: x["start_line"])
    parts = []
    for gap in gaps:
        gs, ge = gap["start_line"], gap["end_line"]
        prev   = next((s for s in reversed(data) if s["end_line"] < gs), None)
        nxt    = next((s for s in data if s["start_line"] > ge), None)
        block  = [f"--- GAP {gap['gap_id']} (lines {gs}–{ge}) ---"]
        if prev:
            block.append(f"[PREVIOUS — last {ctx} lines]\n"
                + slice_lines(lines, max(prev["start_line"], prev["end_line"] - ctx + 1), prev["end_line"]))
        block.append(f"[GAP LINES]\n{slice_lines(lines, gs, ge)}")
        if nxt:
            block.append(f"[NEXT — first {ctx} lines]\n"
                + slice_lines(lines, nxt["start_line"], min(nxt["end_line"], nxt["start_line"] + ctx - 1)))
        parts.append("\n\n".join(block))
    return ("\n\n" + "=" * 60 + "\n\n").join(parts)


def apply_gap_resolutions(data, gaps, resolutions, lines):
    data   = sorted(data, key=lambda x: x["start_line"])
    res_by = {r["gap_id"]: r for r in resolutions}
    for gap in gaps:
        gs, ge = gap["start_line"], gap["end_line"]
        res    = res_by.get(gap["gap_id"], {"belongs_to": "new", "title": "Uncategorized", "summary": ""})
        b      = res.get("belongs_to", "new")
        if b == "previous":
            for seg in reversed(data):
                if seg["end_line"] < gs:
                    seg["end_line"] = ge
                    break
        elif b == "next":
            for seg in data:
                if seg["start_line"] > ge:
                    seg["start_line"] = gs
                    break
        else:
            data.append({"title": res.get("title", "Uncategorized"),
                         "summary": res.get("summary", ""),
                         "start_line": gs, "end_line": ge})
    return sorted(data, key=lambda x: x["start_line"])


# ── Single-chunk segmentation ─────────────────────────────────

def segment_chunk(lines: list[str], start_idx: int, end_idx: int, client: Groq) -> list[dict]:
    """
    Segment lines[start_idx..end_idx] using original line numbers.
    Returns a list of dicts with start_line/end_line in original coordinates.
    """
    chunk_text = numbered_chunk(lines, start_idx, end_idx)

    r = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SEGMENTATION_PROMPT},
            {"role": "user",   "content": chunk_text},
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    data = sorted(parse_json_list(r.choices[0].message.content), key=lambda x: x["start_line"])

    # Clamp to actual chunk boundaries (model may hallucinate line numbers)
    for seg in data:
        seg["start_line"] = max(start_idx, seg["start_line"])
        seg["end_line"]   = min(end_idx,   seg["end_line"])

    # Resolve any gaps within this chunk
    gaps = find_gaps(data, start_idx, end_idx)
    if gaps:
        gap_msg = build_gap_message(gaps, data, lines)
        r2 = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": GAP_RESOLUTION_PROMPT},
                {"role": "user",   "content": gap_msg},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        data = apply_gap_resolutions(data, gaps, parse_json_list(r2.choices[0].message.content), lines)

    return sorted(data, key=lambda x: x["start_line"])


# ── Chunk boundary merge ──────────────────────────────────────

def maybe_merge_boundary(seg_a: dict, seg_b: dict, client: Groq) -> list[dict]:
    """
    Ask the model if the last segment of chunk A and the first segment of
    chunk B are actually the same topic and should be merged.
    """
    user_msg = (
        f"Chunk A — last segment:\n"
        f"  title: {seg_a['title']}\n"
        f"  summary: {seg_a['summary']}\n"
        f"  end_line: {seg_a['end_line']}\n\n"
        f"Chunk B — first segment:\n"
        f"  title: {seg_b['title']}\n"
        f"  summary: {seg_b['summary']}\n"
        f"  start_line: {seg_b['start_line']}"
    )

    r = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": MERGE_BOUNDARY_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=256,
    )

    result = parse_json_obj(r.choices[0].message.content)

    if result.get("decision") == "merge":
        merged = {
            "title":      result.get("title", seg_a["title"]),
            "summary":    result.get("summary", seg_a["summary"]),
            "start_line": seg_a["start_line"],
            "end_line":   seg_b["end_line"],
        }
        return [merged]
    else:
        return [seg_a, seg_b]


# ── Main segmentation pipeline ────────────────────────────────

def segment_transcript(transcript: str, client: Groq) -> list[dict]:
    """
    Full pipeline:
    1. Split transcript into overlapping chunks (≤ MAX_LINES_PER_CHUNK).
    2. Segment each chunk independently (original line numbers preserved).
    3. Merge boundaries between chunks.
    4. Attach real timestamps from the original transcript lines.
    """
    lines       = parse_lines(transcript)
    total_lines = len(lines)

    # ── Step 1: build chunk ranges ────────────────────────────
    ranges = []
    start  = 0
    while start < total_lines:
        end = min(start + MAX_LINES_PER_CHUNK - 1, total_lines - 1)
        ranges.append((start, end))
        if end == total_lines - 1:
            break
        start = end - OVERLAP_LINES + 1   # overlap so boundary topics are seen twice

    print(f"  Transcript: {total_lines} lines → {len(ranges)} chunk(s) "
          f"(max {MAX_LINES_PER_CHUNK} lines each, {OVERLAP_LINES}-line overlap).")

    # ── Step 2: segment each chunk ────────────────────────────
    all_chunk_segs = []
    for i, (cs, ce) in enumerate(ranges):
        print(f"  Segmenting chunk {i + 1}/{len(ranges)} "
              f"(lines {cs}–{ce})...", end="", flush=True)
        segs = segment_chunk(lines, cs, ce, client)
        print(f" {len(segs)} segment(s).")
        all_chunk_segs.append(segs)

    # ── Step 3: merge boundaries between consecutive chunks ───
    if len(all_chunk_segs) == 1:
        merged_segs = all_chunk_segs[0]
    else:
        merged_segs = all_chunk_segs[0]
        for i in range(1, len(all_chunk_segs)):
            next_segs = all_chunk_segs[i]

            # Remove overlap: drop segments from next_segs that start
            # before the end of the last non-overlapping line of merged_segs
            overlap_start = ranges[i][0]   # first line of this chunk
            next_segs = [s for s in next_segs if s["end_line"] >= overlap_start]

            if not next_segs:
                continue

            # Check if the boundary should be merged
            print(f"  Checking boundary between chunk {i} and {i + 1}...", end="", flush=True)
            result = maybe_merge_boundary(merged_segs[-1], next_segs[0], client)
            print(" merged." if len(result) == 1 else " kept separate.")

            merged_segs = merged_segs[:-1] + result + next_segs[1:]

    # ── Step 4: attach timestamps ─────────────────────────────
    segments = []
    for i, item in enumerate(merged_segs):
        start         = max(0, item.get("start_line", 0))
        end           = min(total_lines - 1, item.get("end_line", start))
        segment_lines = lines[start : end + 1]
        text          = " ".join(clean_line(l) for l in segment_lines if clean_line(l))

        start_time = next(
            (extract_timestamp(l) for l in segment_lines if extract_timestamp(l)),
            "00:00:00"
        )
        end_time = next(
            (extract_timestamp(l) for l in reversed(segment_lines) if extract_timestamp(l)),
            start_time
        )

        segments.append({
            "index":      i,
            "title":      item.get("title", f"Topic {i + 1}"),
            "summary":    item.get("summary", ""),
            "start_time": start_time,
            "end_time":   end_time,
            "text":       text,
        })

    return segments


# ── CLI ───────────────────────────────────────────────────────

def print_results(segments: list[dict]) -> None:
    print("\n" + "=" * 70)
    for seg in segments:
        print(f"\n  [{seg['index'] + 1}] {seg['title']}")
        print(f"       {seg['start_time']} → {seg['end_time']}")
        print(f"       {seg['summary']}")
    print("\n" + "=" * 70)


def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in .env")
        raise SystemExit(1)

    script_dir      = Path(__file__).parent
    transcript_path = script_dir / TRANSCRIPT_FILE
    if not transcript_path.exists():
        print(f"Error: {transcript_path} not found. Run transcribe.py first.")
        raise SystemExit(1)

    print(f"\n=== Step 2: Segmentation ===\n")
    print(f"  Reading: {transcript_path}")
    transcript = transcript_path.read_text(encoding="utf-8")

    client   = Groq(api_key=groq_api_key)
    segments = segment_transcript(transcript, client)

    print_results(segments)

    output_path = script_dir / OUTPUT_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"\n  Segments saved → {output_path}")
    print(f"\n=== Done. Run describe.py next. ===\n")


if __name__ == "__main__":
    main()