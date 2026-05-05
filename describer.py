#!/usr/bin/env python3
"""
describe.py  —  Step 3 of 3
=============================
Reads segments.json (output of segment.py) and generates a structured
content description using the pre-built titles and summaries.
 
No re-summarization — the summaries are already there.
 
Output:
    content_description.json
 
Requirements:
    pip install groq python-dotenv
"""
 
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
 
load_dotenv(Path(__file__).parent / ".env")
 
# ============================================================
#  CONFIG
# ============================================================
SEGMENTS_FILE = "segments.json"
OUTPUT_FILE   = "content_description.json"
GROQ_MODEL    = "llama-3.3-70b-versatile"
# ============================================================
 
 
def build_segments_summary(segments: list[dict]) -> str:
    """Format the segments into a compact numbered list for the LLM."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        start   = seg.get("start_time", "?")
        end     = seg.get("end_time",   "?")
        title   = seg.get("title",      f"Topic {i}")
        summary = seg.get("summary",    "")
        lines.append(f"{i}. [{title}]  ({start} → {end})")
        lines.append(f"   {summary}")
        lines.append("")
    return "\n".join(lines)
 
 
def generate_description(segments_text: str, client: Groq) -> dict:
    system_prompt = (
        "You are a professional content analyst. "
        "You will receive a numbered list of topic segments from a video, "
        "each with a title, timestamp range, and a short summary.\n\n"
        "Using only this information, produce a structured content description "
        "as a JSON object with exactly these keys:\n"
        "- summary: a concise 2-3 sentence description of the entire video (string)\n"
        "- target_audience: who would benefit most from watching (string)\n"
        "- tone_and_style: overall tone — educational, casual, formal, etc. (string)\n"
        "- seo_tags: 5-8 relevant keywords or tags (list of strings)\n\n"
        "Return ONLY a valid JSON object. No markdown fences, no commentary.\n\n"
        "IMPORTANT: The entire response MUST be in Arabic only. "
        "Do NOT use English, Russian, or any other language. "
        "Every single word in every field must be Arabic."
    )
 
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Here are the video segments:\n\n{segments_text}"},
        ],
        temperature=0.5,
        max_tokens=2048,
    )
 
    raw = response.choices[0].message.content.strip()
 
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
 
    return json.loads(raw)
 
 
def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in .env")
        raise SystemExit(1)
 
    script_dir    = Path(__file__).parent
    segments_path = script_dir / SEGMENTS_FILE
 
    if not segments_path.exists():
        print(f"Error: {segments_path} not found. Run segment.py first.")
        raise SystemExit(1)
 
    print(f"\n=== Step 3: Content Description ===\n")
    print(f"  Reading: {segments_path}")
 
    with open(segments_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
 
    print(f"  Loaded {len(segments)} segments.")
    segments_text = build_segments_summary(segments)
 
    print(f"  Sending to Groq ({GROQ_MODEL})...")
    client      = Groq(api_key=groq_api_key)
    description = generate_description(segments_text, client)
 
    print("\n" + "=" * 60)
    print("  CONTENT DESCRIPTION")
    print("=" * 60)
    print(json.dumps(description, ensure_ascii=False, indent=2))
    print("=" * 60)
 
    output_path = script_dir / OUTPUT_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(description, f, ensure_ascii=False, indent=2)
 
    print(f"\n  Saved → {output_path}")
    print(f"\n=== Pipeline complete. ===\n")
    print("  Output files:")
    print("    transcript.txt           (from transcribe.py)")
    print("    segments.json            (from segment.py)")
    print("    content_description.json (from describe.py)")
 
 
if __name__ == "__main__":
    main()