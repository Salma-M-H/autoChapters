"""
Topic Segmentation — Arabic & English
=======================================
Handles both Arabic (including unpunctuated transcribed speech) and English.

Key Arabic improvements:
  - Custom Arabic sentence splitter (no reliance on NLTK punkt)
  - Sliding-window chunker for unpunctuated speech transcripts
  - Arabic-aware depth-score segmentation
  - Bilingual LLM prompt (Arabic instructions when Arabic text detected)

Requirements:
    pip install scikit-learn nltk sentence-transformers requests numpy
"""

import re
import json
import requests
import numpy as np
import nltk
from dotenv import load_dotenv
import os

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
load_dotenv()


OPENROUTER_API_KEY        = os.getenv("OPENROUTER_API_KEY")
MODEL                     = "openai/gpt-oss-20b"
MIN_SEGMENT_SENTENCES     = 2     # minimum sentence-chunks per segment
MAX_TOKENS_REPLY          = 2000
# For unpunctuated Arabic speech: words per pseudo-sentence chunk
ARABIC_SPEECH_CHUNK_WORDS = 30


# ─────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────

def is_arabic(text: str) -> bool:
    """Returns True if the text is predominantly Arabic (>40% Arabic chars)."""
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    total_chars  = len(re.sub(r'\s', '', text))
    return (arabic_chars / total_chars) > 0.4 if total_chars > 0 else False


# ─────────────────────────────────────────────
# SENTENCE SPLITTING
# ─────────────────────────────────────────────

def split_arabic_sentences(text: str) -> list:
    """
    Splits Arabic text on sentence-ending punctuation (. ؟ ! newlines).
    Falls back to fixed-word-count chunks when the text is unpunctuated
    speech (e.g. YouTube transcripts), which have no usable boundaries.
    """
    text = re.sub(r'\n+', '\n', text.strip())

    # Split on Arabic/Latin sentence-ending punctuation
    parts = re.split(r'(?<=[.؟!\n])\s*', text)
    parts = [p.strip() for p in parts if p.strip()]

    # Heuristic: if very few parts or average chunk is huge → unpunctuated speech
    avg_words = float(np.mean([len(p.split()) for p in parts])) if parts else 0
    if len(parts) < 4 or avg_words > 50:
        print("   Detected unpunctuated speech → using word-count chunking")
        words = text.split()
        parts = [
            " ".join(words[i : i + ARABIC_SPEECH_CHUNK_WORDS])
            for i in range(0, len(words), ARABIC_SPEECH_CHUNK_WORDS)
        ]

    return [p for p in parts if len(p.split()) >= 3]


def split_sentences(text: str) -> list:
    """Language-aware sentence splitter."""
    if is_arabic(text):
        return split_arabic_sentences(text)
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


# ─────────────────────────────────────────────
# STEP 1 — COSINE SIMILARITY SEGMENTATION
# ─────────────────────────────────────────────

def cosine_segment(text: str) -> list:
    """
    Splits text into topic segments using depth-score boundary detection.

    Depth score at position i = how sharply the similarity dips at that
    gap relative to its neighbors. Boundaries are placed at the deepest
    dips. The cutoff (mean + 0.5*std) adapts to each text automatically.
    Works for Arabic (including transcribed speech) and English.
    """
    sentences = split_sentences(text)
    n = len(sentences)
    if n < 3:
        return [text]

    print(f"   Split into {n} sentence chunks")
    print("   Loading embedding model...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(sentences, show_progress_bar=False)

    # Pairwise similarity between adjacent chunks
    similarities = [
        float(cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0])
        for i in range(n - 1)
    ]

    # Depth score: how much does position i dip vs its neighbors?
    depth_scores = []
    for i in range(len(similarities)):
        left  = similarities[i - 1] if i > 0                      else similarities[i]
        right = similarities[i + 1] if i < len(similarities) - 1  else similarities[i]
        depth_scores.append((left - similarities[i]) + (right - similarities[i]))

    # Adaptive cutoff
    scores = np.array(depth_scores)
    cutoff = scores.mean() + 0.5 * scores.std()

    boundaries = [0]
    for i, d in enumerate(depth_scores):
        candidate = i + 1
        if d >= cutoff and (candidate - boundaries[-1]) >= MIN_SEGMENT_SENTENCES:
            boundaries.append(candidate)
    boundaries.append(n)

    return [
        " ".join(sentences[boundaries[i] : boundaries[i + 1]])
        for i in range(len(boundaries) - 1)
    ]


# ─────────────────────────────────────────────
# STEP 2 — BUILD PROMPT
# ─────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Conservative estimate: ~3 chars/token (covers Arabic & English)."""
    return len(text) // 3


def build_prompt(segments: list, arabic: bool = False) -> str:
    numbered = "\n\n".join(
        f"Segment {i}:\n{seg}" for i, seg in enumerate(segments)
    )

    if arabic:
        instruction = (
            "أنت مساعد لتقسيم النصوص حسب الموضوع.\n"
            "فيما يلي مقاطع نصية تم تقسيمها مسبقاً باستخدام تشابه جيب التمام.\n"
            "مهمتك: جمّع المقاطع التي تنتمي إلى نفس الموضوع معاً.\n\n"
            "القواعد:\n"
            "- خصص لكل مقطع رقم مجموعة موضوعية (topic_group) يبدأ من 0\n"
            "- المقاطع التي تشترك في نفس الموضوع تحصل على نفس الرقم\n"
            "- احتفظ بالترتيب الأصلي للمقاطع\n"
            "- أعد فقط مصفوفة JSON صالحة، بدون أي شرح أو علامات markdown\n\n"
            "الصيغة المطلوبة:\n"
            '[{"segment_index": 0, "topic_group": 0, "topic_label": "اسم الموضوع"}, ...]'
        )
    else:
        instruction = (
            "You are a text segmentation assistant.\n"
            "Below are text segments pre-split using cosine similarity.\n"
            "Group segments that belong to the same topic.\n\n"
            "Rules:\n"
            "- Assign each segment a topic_group (integer starting from 0)\n"
            "- Segments sharing the same topic get the same topic_group number\n"
            "- Preserve original segment order\n"
            "- Return ONLY a valid JSON array, no explanation, no markdown fences\n\n"
            "Format:\n"
            '[{"segment_index": 0, "topic_group": 0, "topic_label": "short topic name"}, ...]'
        )

    return f"{instruction}\n\nSegments:\n{numbered}"


# ─────────────────────────────────────────────
# STEP 3 — SEND TO OPENROUTER
# ─────────────────────────────────────────────

def call_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/topic-segmentation",
        "X-Title": "Topic Segmentation",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Reasoning: low"},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": MAX_TOKENS_REPLY,
        "temperature": 0.1,
    }

    print("   Sending request to OpenRouter...")
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────
# STEP 4 — PARSE LLM RESPONSE
# ─────────────────────────────────────────────

def parse_response(raw: str) -> list:
    """Strips markdown fences and parses JSON array."""
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse LLM response as JSON.\nError: {e}\nRaw:\n{raw}")


# ─────────────────────────────────────────────
# STEP 5 — APPLY GROUPING
# ─────────────────────────────────────────────

def apply_grouping(segments: list, decisions: list) -> list:
    """Merges original segment texts according to LLM topic group assignments."""
    decision_map = {d["segment_index"]: d for d in decisions}
    groups = {}

    for i, seg in enumerate(segments):
        d     = decision_map.get(i, {"topic_group": i, "topic_label": f"موضوع {i}" if is_arabic(seg) else f"Topic {i}"})
        gid   = d["topic_group"]
        label = d.get("topic_label", f"Topic {gid}")

        if gid not in groups:
            groups[gid] = {"topic_group": gid, "topic_label": label, "parts": []}
        groups[gid]["parts"].append(seg)

    return [
        {
            "topic_group": gid,
            "topic_label": groups[gid]["topic_label"],
            "text": " ".join(groups[gid]["parts"]).strip(),
        }
        for gid in sorted(groups)
    ]


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def segment_by_topic(text: str, verbose: bool = True) -> list:
    """
    Full pipeline: raw text → topic-labeled segments.
    Auto-detects Arabic and adjusts every step accordingly.

    Returns:
        list of {"topic_group": int, "topic_label": str, "text": str}
    """
    arabic = is_arabic(text)
    if verbose:
        lang = "Arabic" if arabic else "English"
        print(f"── Detected language: {lang}")

    # Step 1 — Cosine segmentation
    if verbose:
        print("── Step 1: Cosine similarity segmentation...")
    segments = cosine_segment(text)
    if verbose:
        print(f"   → {len(segments)} segments found")

    if len(segments) == 1:
        label = "الموضوع الرئيسي" if arabic else "Main Topic"
        print(f"   → Only 1 segment found, no LLM call needed.")
        return [{"topic_group": 0, "topic_label": label, "text": text.strip()}]

    # Token safety check
    prompt    = build_prompt(segments, arabic=arabic)
    estimated = estimate_tokens(prompt)
    if verbose:
        print(f"   → Estimated prompt size: ~{estimated:,} tokens")
    if estimated > 100_000:
        raise ValueError(
            f"Prompt too large (~{estimated:,} tokens). "
            "Use the compact pipeline (topic_segmentation_pipeline.py) instead."
        )

    # Step 2 — LLM call
    if verbose:
        print("── Step 2: Sending segments to LLM...")
    raw = call_openrouter(prompt)
    if verbose:
        print(f"   → Raw LLM response:\n{raw}\n")

    # Step 3 — Parse
    if verbose:
        print("── Step 3: Parsing LLM decisions...")
    decisions = parse_response(raw)

    # Step 4 — Apply grouping
    if verbose:
        print("── Step 4: Merging segments by topic group...")
    results = apply_grouping(segments, decisions)

    if verbose:
        print(f"\n✓ Done — {len(results)} final topic segments\n")
        for r in results:
            preview = r["text"][:100].replace("\n", " ")
            print(f'  [{r["topic_group"]}] {r["topic_label"]}: "{preview}..."')

    return results


# ─────────────────────────────────────────────
# EXTRACT HELPERS
# ─────────────────────────────────────────────

def get_texts(results: list) -> list:
    return [r["text"] for r in results]

def get_labels(results: list) -> list:
    return [r["topic_label"] for r in results]

def get_by_label(results: list, keyword: str) -> list:
    return [r for r in results if keyword.lower() in r["topic_label"].lower()]

def get_by_group(results: list, group_id: int):
    return next((r for r in results if r["topic_group"] == group_id), None)


# ─────────────────────────────────────────────
# EXAMPLE — paste your Arabic text here
# ─────────────────────────────────────────────

if __name__ == "__main__":
    arabic_text = """
    انت في غرفه ثم تنتقل الى غرفه اخرى وفجاه ينسى عقلك سبب نهوضك في المقام الاول
    يعرف هذا بتاثير العتبه وجد الباحثون ان عبور العتبه يعد بمثابه حدث عقلي فاصل
    مما يجعل دماغك يتخلى عن السياق القديم ويركز على الجديد يشبه الامر اغلاق علامه تبويب وفتح اخرى
    مما يصعب استرجاع الفكره الاصليه لكن لا تقلق اذا عدت الى حيث بدات غالبا ما يعيد دماغك تحميل المعلومات المفقوده

    هل سبق وان كنت تتعرض لحادث سياره وشعرت ان الزمن تباطا هذا هو تمدد الزمن
    ففي لحظات التوتر العالي يسجل دماغك تفاصيل اكثر بكثير من المعتاد
    مما يجعلك تشعر وكان الزمن يتمدد فكر فيها كانك تنتقل من 30 اطارا في الثانيه الى 120 اطارا في الثانيه
    هذا ليس مجرد شيء غريب بل اليه للبقاء تساعدك على التعامل مع التهديدات بشكل اكثر فعاليه

    تتعلم كلمه او مفهوما جديدا وفجاه تراه في كل مكان هذا ليس سحرا بل ظاهره بادر ماينهوف
    المعروفه ايضا بوهم التكرار دماغك يرشح المعلومات باستمرار وبمجرد ان يصبح شيء ما مهما يبدا باعطائه الاولويه
    ليس ان العالم تغير فجاه بل ان ادراكك استقبل تحديثا

    ذاكرتك ليست تسجيلا دقيقا بل هي اعاده بناء واحيانا يملا دماغك الفجوات بتفاصيل لم تحدث اصلا
    اظهرت الدراسات ان الناس يمكن اقناعهم بانهم شاهدوا او مروا باشياء لم تحدث ابدا حتى بتفاصيل حيه
    ولهذا السبب غالبا ما تكون الشهادات العينيه غير موثوقه
    """

    results = segment_by_topic(arabic_text, verbose=True)

    print("\n── Full JSON:")
    print(json.dumps(results, indent=2, ensure_ascii=False))