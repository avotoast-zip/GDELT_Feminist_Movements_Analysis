#!/usr/bin/env python3
"""
07_fetch_and_label.py
---------------------
Fetches each article and asks Gemini to code it, storing BOTH the fetched text
and the model's reasoning alongside every label.

Two design points you asked for:

  1. The article text travels WITH the label into the output file, so every row
     is auditable — you can read exactly what the model read. Hard rule: if the
     text did not fetch, the row is written with its fetch_status and is NOT
     sent to the API. A stance coded from a headline is not a stance, and
     silently headline-coded rows would corrupt your agreement statistics later.

  2. The model reasons BEFORE it commits to a stance, and the reasoning is
     stored. Key order matters: an LLM fills JSON keys in the order you specify,
     so "stance_reasoning" placed before "stance" is real reasoning-then-
     conclusion. Placed after, it is just a story invented to fit an answer
     already given.

The fetch is PARALLEL (the old sequential version would have taken days at
205k articles). Fetching and labeling are separate phases so you can stop after
fetching if the rate is bad.

INPUT   fetch_list.csv          (from 05a_dedupe_prefetch.py)
OUTPUT  labeled_articles.csv    (fetch status + text + label + reasoning)

SETUP
    pip install google-genai trafilatura pandas tqdm python-dotenv
    echo "GOOGLE_API_KEY=your-key" >> .env

USAGE
    python 07_fetch_and_label.py --limit 20      # PILOT: read all 20 reasonings
    python 07_fetch_and_label.py --fetch-only    # fetch everything, label later
    python 07_fetch_and_label.py                 # full run

LICENSE LINE (do not cross): only GDELT-sourced, openly fetched text goes
through this script. Never put Nexis Uni text into it.
"""

# ---------------------------------------------------------------------------
# Repository paths. Added when the project folder was reorganised (Aug 2026).
# Scripts live in scripts/ and resolve every input and output from the repo
# root, so they run from anywhere:  python scripts/17_lda_topics.py
# ---------------------------------------------------------------------------
from pathlib import Path as _Path
ROOT       = _Path(__file__).resolve().parents[1]
CODEBOOK   = ROOT / "codebook"
GEO        = ROOT / "assets" / "geo"
RAW        = ROOT / "data" / "raw"
INTERIM    = ROOT / "data" / "interim"
PROCESSED  = ROOT / "data" / "processed"
FIGURES    = ROOT / "outputs" / "figures"
REPORTS    = ROOT / "outputs" / "reports"
TABLES     = ROOT / "outputs" / "tables"
DASHBOARDS = ROOT / "outputs" / "dashboards"
for _d in (INTERIM, PROCESSED, FIGURES, REPORTS, TABLES, DASHBOARDS, GEO):
    _d.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------------------------------

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import trafilatura
from dotenv import load_dotenv
from tqdm import tqdm

warnings.filterwarnings("ignore")
load_dotenv(ROOT / ".env")

IN_FILE = INTERIM / "fetch_list.csv"
OUT_FILE = PROCESSED / "labeled_articles.csv"
MODEL = "gemini-2.5-flash"

MAX_STORE = 8000       # chars of text kept in the CSV for your audit
MAX_SEND = 5000        # chars sent to the model
FETCH_WORKERS = 16   # archive.org rate-limits; keep modest when wayback is on
API_WORKERS = 8        # parallel API calls; lower this if you hit rate limits
CHECKPOINT = 200

# ── RUBRIC ──────────────────────────────────────────────────────────────
RUBRIC = """You are coding a news article for an academic study of how global news
media covered the #MeToo movement and its backlash (2017-2019).

Read the article, then answer ONLY with a JSON object with these keys IN THIS
ORDER. The order matters: work through the evidence before committing to a stance.

1. "language": ISO 639-1 code of the article's language (e.g. "en", "fr", "pt").

2. "is_about_movement": true/false. True if the article concerns the #MeToo
   movement, its backlash, or a case explicitly framed as part of it. False if
   the movement is only mentioned in passing, or this is unrelated coverage
   (e.g. a general crime report with no movement framing). If false, still fill
   the other fields as best you can but set "stance" to "not_applicable".

3. "article_type": one of "straight_news", "opinion_editorial",
   "feature_analysis", "meme_roundup", "interview", "other".

4. "article_type_reasoning": one sentence. What in the text told you this?

5. "primary_focus": one short phrase - what or who the article is mainly about.

6. "figures": list of objects, one per named public figure, each
   {"name": ..., "portrayal": "sympathetic"|"critical"|"neutral",
    "why_mentioned": one short phrase}

7. "voice_balance": one of "movement_voices_dominate",
   "backlash_voices_dominate", "balanced", "no_quoted_voices".

8. "voice_balance_reasoning": one sentence naming who is quoted, at what length,
   and who gets the final word.

9. "evaluative_language": list of words or phrases where the JOURNALIST (not a
   quoted source) makes an evaluative choice - e.g. "rightly", "inflammatory",
   "so-called", "admitted", "claimed". Empty list if none. This is the main
   evidence for stance leaking through apparently neutral framing.

10. "stance_reasoning": TWO TO FOUR sentences. Apply the dominant-effect test
    explicitly: what is the reader's takeaway, and what in the text produces it?
    Weigh the voice balance and the evaluative language you just listed. If the
    call is close, say what would have tipped it the other way. Do NOT state the
    stance label here - reason toward it.

11. "stance": one of:
    "support"  - dominant effect validates the movement, centers survivors, or
                 frames mockery of backlash approvingly.
    "backlash" - dominant effect undermines the movement: false-accusation,
                 witch-hunt, due-process, men-as-victims framing given
                 unchallenged space.
    "neutral"  - factual reporting with no discernible stance.
    "satire"   - the article ITSELF is satirical/ironic, or is a pure meme
                 compilation whose own stance cannot be determined.
    "not_applicable" - "is_about_movement" is false.

    Decision aids:
    - Tone of subject matter is NOT stance: an article describing assault
      sympathetically is support even though its language is grim.
    - Mentioning accusations is not balance; balance means the other side gets
      real space.
    - Apply the dominant-effect test: what is the reader's takeaway?

12. "stance_confidence": "H", "M", or "L".

13. "confidence_reasoning": one sentence. If not "H", name the specific thing
    that is ambiguous - that is what a human coder will adjudicate.

14. "evidence": ONE short quote (under 15 words) from the article that best
    supports your stance call. Quote verbatim; do not paraphrase.

15. "movement_centrality": "primary" or "incidental".

Return ONLY the JSON object. No preamble, no markdown fences."""

SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "is_about_movement": {"type": "boolean"},
        "article_type": {"type": "string", "enum": [
            "straight_news", "opinion_editorial", "feature_analysis",
            "meme_roundup", "interview", "other"]},
        "article_type_reasoning": {"type": "string"},
        "primary_focus": {"type": "string"},
        "figures": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "portrayal": {"type": "string",
                              "enum": ["sympathetic", "critical", "neutral"]},
                "why_mentioned": {"type": "string"}},
            "required": ["name", "portrayal", "why_mentioned"]}},
        "voice_balance": {"type": "string", "enum": [
            "movement_voices_dominate", "backlash_voices_dominate",
            "balanced", "no_quoted_voices"]},
        "voice_balance_reasoning": {"type": "string"},
        "evaluative_language": {"type": "array", "items": {"type": "string"}},
        "stance_reasoning": {"type": "string"},
        "stance": {"type": "string", "enum": [
            "support", "backlash", "neutral", "satire", "not_applicable"]},
        "stance_confidence": {"type": "string", "enum": ["H", "M", "L"]},
        "confidence_reasoning": {"type": "string"},
        "evidence": {"type": "string"},
        "movement_centrality": {"type": "string",
                                "enum": ["primary", "incidental"]},
    },
    "required": ["language", "is_about_movement", "article_type",
                 "article_type_reasoning", "primary_focus", "figures",
                 "voice_balance", "voice_balance_reasoning",
                 "evaluative_language", "stance_reasoning", "stance",
                 "stance_confidence", "confidence_reasoning", "evidence",
                 "movement_centrality"],
    "propertyOrdering": ["language", "is_about_movement", "article_type",
                         "article_type_reasoning", "primary_focus", "figures",
                         "voice_balance", "voice_balance_reasoning",
                         "evaluative_language", "stance_reasoning", "stance",
                         "stance_confidence", "confidence_reasoning",
                         "evidence", "movement_centrality"],
}
FIELDS = SCHEMA["propertyOrdering"]


# ── FETCH ───────────────────────────────────────────────────────────────
BROWSER = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8,es;q=0.7,de;q=0.6",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _extract(html_text):
    if not html_text:
        return None, ""
    text = trafilatura.extract(html_text, include_comments=False,
                               include_tables=False, favor_recall=True)
    meta = trafilatura.extract_metadata(html_text)
    title = (meta.title if meta and meta.title else "") or ""
    return text, title


def fetch_one(url, use_wayback=True):
    """Direct request with browser headers; Wayback Machine as fallback.

    The pilot diagnosis showed 64 of 300 failures were HTTP 403 — servers
    refusing the default scraper user-agent, not dead pages. Sending normal
    browser headers recovered 62 of them. Countries with heavy consent walls
    benefit most: Canada went 12% -> 78%, Switzerland 13% -> 62%.
    """
    # ── attempt 1: direct, with browser headers ──
    code = None
    try:
        r = requests.get(url, headers=BROWSER, timeout=20, allow_redirects=True)
        code = r.status_code
        if r.ok and r.text:
            text, title = _extract(r.text)
            if text and len(text) >= 400:
                return (title, text[:MAX_STORE], "OK")
            if text:
                return (title, text[:MAX_STORE], "TOO_SHORT")
            return (title, "", "NO_TEXT")
    except Exception as e:
        code = type(e).__name__

    if not use_wayback:
        return ("", "", f"FAILED:{code}")

    # ── attempt 2: Wayback Machine ──
    # archive.org rate-limits hard. Keep WAYBACK_WORKERS low, and treat a
    # failure here as "unknown", not as proof the page was never archived.
    try:
        api = f"http://archive.org/wayback/available?url={url}&timestamp=20180601"
        j = requests.get(api, timeout=25).json()
        snap = j.get("archived_snapshots", {}).get("closest", {})
        if snap.get("available") and snap.get("url"):
            r2 = requests.get(snap["url"], headers=BROWSER, timeout=35)
            if r2.ok and r2.text:
                text, title = _extract(r2.text)
                if text and len(text) >= 400:
                    return (title, text[:MAX_STORE], "OK_WAYBACK")
                if text:
                    return (title, text[:MAX_STORE], "TOO_SHORT")
        return ("", "", f"DEAD:{code}")
    except Exception:
        return ("", "", f"WAYBACK_ERR:{code}")


# ── LABEL ───────────────────────────────────────────────────────────────
def label_one(client, types, row):
    prompt = (
        f"SOURCE COUNTRY: {row.get('source_country','')}\n"
        f"KEYWORDS THAT MATCHED THIS ARTICLE: {row.get('keywords_matched','')}\n"
        f"(The keywords are why this article entered the corpus. They may be "
        f"misleading - judge the article on its text, not on the keyword.)\n\n"
        f"HEADLINE: {row.get('fetched_title','')}\n\n"
        f"ARTICLE TEXT:\n{str(row.get('article_text',''))[:MAX_SEND]}")
    resp = client.models.generate_content(
        model=MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=RUBRIC,
            response_mime_type="application/json",
            response_schema=SCHEMA,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ))
    return json.loads(resp.text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N rows (use 20 for a pilot)")
    ap.add_argument("--fetch-only", action="store_true",
                    help="fetch text and stop; no API calls")
    ap.add_argument("--fetch-workers", type=int, default=FETCH_WORKERS)
    ap.add_argument("--api-workers", type=int, default=API_WORKERS)
    ap.add_argument("--no-wayback", action="store_true",
                    help="skip the archive.org fallback (faster, fewer articles)")
    a = ap.parse_args()

    df = pd.read_csv(IN_FILE, low_memory=False).drop_duplicates("url")
    if a.limit:
        df = df.head(a.limit)

    done = set()
    if os.path.exists(OUT_FILE):
        prev = pd.read_csv(OUT_FILE, low_memory=False)
        ok = prev["stance"].notna() & (prev["stance"].astype(str).str.len() > 0)
        done = set(prev.loc[ok, "url"])
        print(f"resuming: {len(done):,} already labeled")
        df = df[~df["url"].isin(done)]
    print(f"{len(df):,} articles to process\n")
    if not len(df):
        print("nothing to do.")
        return

    # ── PHASE 1: parallel fetch ──
    print(f"PHASE 1  fetching with {a.fetch_workers} workers")
    res = {}
    with ThreadPoolExecutor(max_workers=a.fetch_workers) as ex:
        futs = {ex.submit(fetch_one, u, not a.no_wayback): u for u in df["url"]}
        for f in tqdm(as_completed(futs), total=len(futs), desc="fetch"):
            res[futs[f]] = f.result()

    df = df.copy()
    df["fetched_title"] = df["url"].map(lambda u: res[u][0])
    df["article_text"] = df["url"].map(lambda u: res[u][1])
    df["fetch_status"] = df["url"].map(lambda u: res[u][2])
    df["text_chars"] = df["article_text"].str.len()

    print("\nfetch results:")
    print(df["fetch_status"].value_counts().to_string())
    usable = df["fetch_status"].isin(["OK", "OK_WAYBACK"])
    print(f"usable: {usable.sum():,}/{len(df):,} ({usable.mean()*100:.1f}%)")

    for f in FIELDS:
        df[f] = ""
    df["llm_error"] = ""

    if a.fetch_only:
        df.to_csv(OUT_FILE, index=False)
        print(f"\nfetch-only: wrote {OUT_FILE}. Re-run without --fetch-only to label.")
        return

    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        df.to_csv(OUT_FILE, index=False)
        print("\nNo GOOGLE_API_KEY in .env — text saved, labeling skipped.")
        return

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)

    # ── PHASE 2: parallel label, only on rows with real text ──
    work = df[usable]
    print(f"\nPHASE 2  labeling {len(work):,} articles with {a.api_workers} workers")

    def do(idx_row):
        i, row = idx_row
        for attempt in range(5):
            try:
                return i, label_one(client, types, row), ""
            except Exception as e:
                msg = str(e)
                if "429" in msg or "quota" in msg.lower() or "Resource" in type(e).__name__:
                    time.sleep(5 * (2 ** attempt))
                    continue
                return i, None, f"API_ERROR:{type(e).__name__}"
        return i, None, "API_ERROR:retries_exhausted"

    n_done = 0
    with ThreadPoolExecutor(max_workers=a.api_workers) as ex:
        futs = [ex.submit(do, (i, r)) for i, r in work.to_dict("index").items()]
        for fu in tqdm(as_completed(futs), total=len(futs), desc="label"):
            i, out, err = fu.result()
            if out:
                for f in FIELDS:
                    v = out.get(f, "")
                    df.at[i, f] = (json.dumps(v, ensure_ascii=False)
                                   if isinstance(v, (list, dict)) else v)
            else:
                df.at[i, "llm_error"] = err
            n_done += 1
            if n_done % CHECKPOINT == 0:
                df.to_csv(OUT_FILE, index=False)

    df.to_csv(OUT_FILE, index=False)

    lab = df[df["stance"].astype(str).str.len() > 0]
    print(f"\nlabeled: {len(lab):,}")
    if len(lab):
        print("\nLLM stance:")
        print(lab["stance"].value_counts().to_string())
        print("\nkeyword-based stance vs LLM stance:")
        print(pd.crosstab(lab["article_stance"], lab["stance"]).to_string())
        print("\nURL heuristic type vs LLM type:")
        print(pd.crosstab(lab["article_type_x"] if "article_type_x" in lab
                          else lab["article_type"], lab["article_type"]).to_string()[:800])
        rel = lab["is_about_movement"].astype(str).str.lower().isin(["true", "1"])
        print(f"\nactually about the movement: {rel.sum():,}/{len(lab):,} "
              f"({rel.mean()*100:.1f}%)")

    df.assign(ok=df["fetch_status"].isin(["OK", "OK_WAYBACK"])).groupby("source_country")["ok"] \
      .agg(["mean", "count"]).sort_values("mean").to_csv(TABLES / "fetch_rate_by_country.csv")
    print("\nwrote fetch_rate_by_country.csv (for your methods section)")
    print(f"wrote {OUT_FILE}")
    print("\nnext: python 08_dedupe.py")


if __name__ == "__main__":
    main()
