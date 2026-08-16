#!/usr/bin/env python3
"""
13_label_simple.py
------------------
No threads. No progress bar. One call at a time, one line printed per call.

Why: `ps` showed the threaded version using 5 seconds of CPU across 5 minutes of
wall clock - it was blocked on network calls that never returned. Two likely
causes, both removed here:
  - the SDK timeout was set inside a try/except that silently fell back to a
    client with NO timeout, so a hung connection blocked forever
  - four threads sharing one client and its HTTP connection pool

Sequential execution removes both, plus rate-limit bursts. At the ~2s per call
your preflight measured, 400 articles takes about 13 minutes.

    python 13_label_simple.py --n 400

Stop it any time with Ctrl-C. It saves every 20 calls and resumes from
sample_labeled.csv, so nothing is lost and you can stop when you run out of
time and still report what you have.
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
import math
import os
import socket
import sys
import time

import pandas as pd

# Belt and braces: force a socket-level timeout so no connection can hang
# forever regardless of what the SDK does with its own timeout setting.
socket.setdefaulttimeout(60)

IN_FILE = PROCESSED / "labeled_articles.csv"
FRAME = INTERIM / "sample_frame_simple.csv"
OUT = PROCESSED / "sample_labeled.csv"
MODEL = "gemini-2.5-flash"
MAX_SEND = 4000
SAVE_EVERY = 20

RUBRIC = """You are coding a news article for a study of how global media covered the
#MeToo movement (2017-2019).

Answer ONLY with a JSON object, keys in this exact order:

1. "is_about_movement": true/false. TRUE if the article concerns the #MeToo
   movement, a comparable feminist movement, or a case explicitly framed as part
   of one. FALSE for routine crime reporting, court reports, or gender-violence
   statistics with no movement framing. Be strict: a single assault case with no
   movement framing is FALSE even though its subject is gender violence.

2. "movement_reasoning": one sentence naming the specific cue that decided it.

3. "article_type": "straight_news" | "opinion_editorial" | "feature_analysis" |
   "interview" | "other".

4. "stance_reasoning": two sentences. What is the reader's takeaway and what
   produces it? Do not state the label here.

5. "stance": "support" | "backlash" | "neutral" | "not_applicable".
   Use "not_applicable" when is_about_movement is false.

6. "stance_confidence": "H" | "M" | "L".

Return ONLY the JSON object."""

SCHEMA = {
    "type": "object",
    "properties": {
        "is_about_movement": {"type": "boolean"},
        "movement_reasoning": {"type": "string"},
        "article_type": {"type": "string", "enum": [
            "straight_news", "opinion_editorial", "feature_analysis",
            "interview", "other"]},
        "stance_reasoning": {"type": "string"},
        "stance": {"type": "string", "enum": [
            "support", "backlash", "neutral", "not_applicable"]},
        "stance_confidence": {"type": "string", "enum": ["H", "M", "L"]},
    },
    "required": ["is_about_movement", "movement_reasoning", "article_type",
                 "stance_reasoning", "stance", "stance_confidence"],
    "propertyOrdering": ["is_about_movement", "movement_reasoning",
                         "article_type", "stance_reasoning", "stance",
                         "stance_confidence"],
}
FIELDS = SCHEMA["propertyOrdering"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def build_sample(n):
    print("reading fetched file (chunked, ~1 min)...", flush=True)
    parts = []
    for ch in pd.read_csv(IN_FILE,
                          usecols=["url", "source_country", "has_hashtag_term",
                                   "fetch_status", "keywords_matched"],
                          chunksize=200_000, low_memory=False):
        parts.append(ch[ch["fetch_status"].isin(["OK", "OK_WAYBACK"])])
    df = pd.concat(parts, ignore_index=True)
    df["match_type"] = df["has_hashtag_term"].map({True: "hashtag", False: "generic"})
    print(f"  usable: {len(df):,}", flush=True)

    half = n // 2
    g = df[df["match_type"] == "generic"].sample(n=half, random_state=20260731)
    h = df[df["match_type"] == "hashtag"].sample(n=n - half, random_state=20260731)
    # interleave so a partial run still has both types
    samp = pd.concat([g.reset_index(drop=True), h.reset_index(drop=True)],
                     keys=["g", "h"]).swaplevel().sort_index().reset_index(drop=True)

    print("attaching article text...", flush=True)
    want = set(samp["url"])
    tparts = []
    for ch in pd.read_csv(IN_FILE, usecols=["url", "fetched_title", "article_text"],
                          chunksize=50_000, low_memory=False):
        hit = ch[ch["url"].isin(want)]
        if len(hit):
            tparts.append(hit)
    txt = pd.concat(tparts, ignore_index=True).drop_duplicates("url")
    samp = samp.merge(txt, on="url", how="left")
    samp = samp[samp["article_text"].notna()].reset_index(drop=True)
    samp.to_csv(FRAME, index=False)
    print(f"  sample ready: {len(samp):,}\n", flush=True)
    return samp


def report(path=OUT):
    df = pd.read_csv(path, low_memory=False)
    lab = df[df["stance"].astype(str).str.len() > 0].copy()
    if not len(lab):
        print("nothing labeled yet.")
        return
    lab["about"] = lab["is_about_movement"].astype(str).str.lower().isin(["true", "1"])
    L = ["", "=" * 62,
         "IS THE GENERIC-VOCABULARY MATERIAL ABOUT THE MOVEMENT?",
         "=" * 62]
    for mt in ["hashtag", "generic"]:
        s = lab[lab["match_type"] == mt]
        if not len(s):
            continue
        k, n = int(s["about"].sum()), len(s)
        lo, hi = wilson(k, n)
        L.append(f"  {mt.upper():9s} n={n:4d}   about the movement: {k:4d}  "
                 f"{k/n*100:5.1f}%   [95% CI {lo*100:.0f}-{hi*100:.0f}]")
    g = lab[lab["match_type"] == "generic"]
    if len(g):
        r = g["about"].mean()
        L += ["", f"  Of ~50,700 fetched generic-matched articles, roughly",
              f"  {int(50700*r):,} are movement coverage and {int(50700*(1-r)):,} are not."]
    am = lab[lab["about"]]
    if len(am):
        L += ["", "STANCE among articles about the movement:"]
        for k2, v in am["stance"].value_counts().items():
            L.append(f"  {k2:16s} {v:4d}  {v/len(am)*100:5.1f}%")
    L += ["", "SPOT-CHECK before you present this:",
          "  open sample_labeled.csv, read movement_reasoning for 10 rows",
          "  where match_type=generic and is_about_movement=False."]
    out = "\n".join(L)
    print(out)
    open(REPORTS / "sample_report.txt", "w").write(out)
    print("\nwrote sample_report.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    if a.report_only:
        report()
        return

    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            key = os.environ.get("GOOGLE_API_KEY")
        except ImportError:
            pass
    if not key:
        sys.exit("no GOOGLE_API_KEY")

    from google import genai
    from google.genai import types

    # Do NOT swallow a timeout failure silently - that is what hid the hang.
    try:
        client = genai.Client(api_key=key,
                              http_options=types.HttpOptions(timeout=45_000))
        print("client: SDK timeout set to 45s", flush=True)
    except Exception as e:
        client = genai.Client(api_key=key)
        print(f"client: SDK timeout NOT supported ({type(e).__name__}); "
              f"relying on the 60s socket timeout instead", flush=True)

    if os.path.exists(FRAME):
        samp = pd.read_csv(FRAME, low_memory=False)
        if len(samp) < a.n:
            samp = build_sample(a.n)
    else:
        samp = build_sample(a.n)
    samp = samp.head(a.n)

    done = set()
    rows = []
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT, low_memory=False)
        prev = prev[prev["stance"].astype(str).str.len() > 0]
        done = set(prev["url"])
        rows = prev.to_dict("records")
        print(f"resuming: {len(done)} already done", flush=True)

    todo = samp[~samp["url"].isin(done)]
    print(f"to label: {len(todo)}   (sequential, one at a time)\n", flush=True)

    n_ok = n_err = 0
    t0 = time.time()
    try:
        for pos, (_, row) in enumerate(todo.iterrows(), 1):
            prompt = (f"COUNTRY: {row.get('source_country','')}\n"
                      f"MATCHED KEYWORDS: {row.get('keywords_matched','')}\n"
                      f"(judge the article on its text, not the keyword)\n\n"
                      f"HEADLINE: {row.get('fetched_title','')}\n\n"
                      f"TEXT:\n{str(row.get('article_text',''))[:MAX_SEND]}")
            t1 = time.time()
            rec = row.to_dict()
            try:
                r = client.models.generate_content(
                    model=MODEL, contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=RUBRIC,
                        response_mime_type="application/json",
                        response_schema=SCHEMA,
                        temperature=0.0,
                        thinking_config=types.ThinkingConfig(thinking_budget=0)))
                out = json.loads(r.text)
                rec.update(out)
                n_ok += 1
                flag = "MOVEMENT" if out.get("is_about_movement") else "not-mvmt"
                print(f"[{pos:4d}/{len(todo)}] {time.time()-t1:4.1f}s  "
                      f"{row['match_type']:8s} {str(row['source_country'])[:13]:13s} "
                      f"{flag}", flush=True)
            except Exception as e:
                msg = str(e)
                code = ("429-ratelimit" if "429" in msg else
                        "403-forbidden" if "403" in msg else
                        "timeout" if "timeout" in msg.lower() else
                        type(e).__name__)
                rec["llm_error"] = code
                for f in FIELDS:
                    rec.setdefault(f, "")
                n_err += 1
                print(f"[{pos:4d}/{len(todo)}] FAIL {code}: {msg[:100]}", flush=True)
                if "429" in msg:
                    print("        rate limited - pausing 15s", flush=True)
                    time.sleep(15)
            rows.append(rec)
            if pos % SAVE_EVERY == 0:
                pd.DataFrame(rows).to_csv(OUT, index=False)
                el = time.time() - t0
                rate = el / pos
                left = (len(todo) - pos) * rate / 60
                print(f"        --- saved. {n_ok} ok, {n_err} err, "
                      f"{rate:.1f}s/call, ~{left:.0f} min left ---", flush=True)
    except KeyboardInterrupt:
        print("\nstopped by you - saving what we have...", flush=True)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nlabeled {n_ok}, failed {n_err}, in {(time.time()-t0)/60:.1f} min")
    report()


if __name__ == "__main__":
    main()
