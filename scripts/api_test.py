#!/usr/bin/env python3
"""
api_test.py — 20-second check that the API works at all.
Run this BEFORE re-running the labeling, so you find out in 20 seconds
instead of staring at a 0% progress bar.

    python api_test.py
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
import os, sys, time, json

key = os.environ.get("GOOGLE_API_KEY")
if not key:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        key = os.environ.get("GOOGLE_API_KEY")
    except ImportError:
        pass
if not key:
    sys.exit("FAIL: no GOOGLE_API_KEY found in environment or .env")
print(f"key found: {key[:8]}...{key[-4:]}  (length {len(key)})")

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("FAIL: pip install google-genai")

client = genai.Client(api_key=key)
print("client created. sending one tiny request...")

t0 = time.time()
try:
    r = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Reply with exactly this JSON and nothing else: {\"ok\": true}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ))
    dt = time.time() - t0
    print(f"RESPONSE in {dt:.1f}s: {r.text.strip()[:80]}")
    print("\nPASS - the API works.")
    print(f"At {dt:.1f}s per call with 4 workers, 2,176 articles would take")
    print(f"roughly {2176*dt/4/60:.0f} minutes.")
except Exception as e:
    dt = time.time() - t0
    print(f"\nFAIL after {dt:.1f}s")
    print(f"  {type(e).__name__}: {str(e)[:300]}")
    print("\nCommon causes:")
    print("  - 400 API_KEY_INVALID  -> the key is wrong")
    print("  - 403 PERMISSION_DENIED -> key not enabled for the Gemini API")
    print("  - 429 RESOURCE_EXHAUSTED -> free-tier quota hit; wait or use --n 400")
    print("  - hangs with no error   -> network/proxy blocking the API")
