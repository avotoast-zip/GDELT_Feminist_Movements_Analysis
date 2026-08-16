#!/usr/bin/env python3
"""
07b_wayback_retry.py
--------------------
A second, patient pass at the Wayback Machine for the rows where archive.org
errored out rather than answering.

WHY THIS EXISTS
  Your 500-article pilot came back 64.6% usable, but 141 rows were
  WAYBACK_ERR — archive.org refused or timed out before telling us whether a
  snapshot exists. Only 19 rows got an actual "no archive" answer. So those 141
  are UNEXAMINED, not dead, and treating them as dead would understate your
  recoverable corpus by a lot.

  archive.org rate-limits hard and does not like concurrency. This script goes
  slowly on purpose: few workers, real backoff on 429, and it uses the CDX
  index rather than the availability API, which is more reliable and lets us
  ask for a snapshot that actually returned HTTP 200 within your study window.

  It also requests the snapshot with the `id_` modifier, which returns the
  original page without archive.org's navigation banner — cleaner extraction.

  Safe to stop and re-run: it only touches rows that are still unresolved, and
  it checkpoints as it goes.

USAGE
    python 07b_wayback_retry.py                  # patient defaults
    python 07b_wayback_retry.py --workers 2      # even gentler if you see 429s
    python 07b_wayback_retry.py --limit 200      # try it on a subset first
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
import os
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import trafilatura
from tqdm import tqdm

warnings.filterwarnings("ignore")

FILE = PROCESSED / "labeled_articles.csv"
MAX_STORE = 8000
CHECKPOINT = 50

BROWSER = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

CDX = "http://web.archive.org/cdx/search/cdx"


def cdx_lookup(url, tries=4):
    """Ask the CDX index for one archived snapshot that returned HTTP 200.
    Returns a timestamp string, None if genuinely not archived, or
    'RATELIMIT' if archive.org would not answer."""
    params = {
        "url": url, "output": "json", "limit": 1,
        "filter": "statuscode:200", "fl": "timestamp",
        "from": "2017", "to": "2021", "collapse": "digest",
    }
    for i in range(tries):
        try:
            r = requests.get(CDX, params=params, timeout=30, headers=BROWSER)
            if r.status_code == 429:
                time.sleep((2 ** i) * 5 + random.random() * 3)
                continue
            if not r.ok:
                time.sleep(2 + random.random() * 2)
                continue
            rows = r.json()
            if len(rows) <= 1:          # header row only == not archived
                return None
            return rows[1][0]
        except Exception:
            time.sleep((2 ** i) * 3 + random.random() * 2)
    return "RATELIMIT"


def fetch_snapshot(url, ts):
    """`id_` gives the original bytes without the archive.org banner."""
    wb = f"http://web.archive.org/web/{ts}id_/{url}"
    for i in range(3):
        try:
            r = requests.get(wb, headers=BROWSER, timeout=45)
            if r.status_code == 429:
                time.sleep((2 ** i) * 5)
                continue
            if r.ok and r.text:
                text = trafilatura.extract(r.text, include_comments=False,
                                           include_tables=False,
                                           favor_recall=True)
                meta = trafilatura.extract_metadata(r.text)
                title = (meta.title if meta and meta.title else "") or ""
                return text, title
            return None, ""
        except Exception:
            time.sleep(2 + random.random() * 2)
    return None, ""


def resolve(url):
    ts = cdx_lookup(url)
    if ts is None:
        return ("", "", "DEAD_confirmed")       # archive says: never captured
    if ts == "RATELIMIT":
        return ("", "", "WAYBACK_ERR")          # still unexamined; retry later
    text, title = fetch_snapshot(url, ts)
    if text and len(text) >= 400:
        return (title, text[:MAX_STORE], "OK_WAYBACK")
    if text:
        return (title, text[:MAX_STORE], "TOO_SHORT")
    return ("", "", "WAYBACK_THIN")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3,
                    help="keep this low: archive.org rate-limits concurrency")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    if not os.path.exists(FILE):
        raise SystemExit(f"{FILE} not found — run 07_fetch_and_label.py first.")
    df = pd.read_csv(FILE, low_memory=False)

    # only rows we never got a real answer for
    unresolved = df["fetch_status"].astype(str).str.startswith(
        ("WAYBACK_ERR", "DEAD:", "FAILED:", "FETCH_FAILED", "NO_TEXT"))
    todo = df[unresolved]
    if a.limit:
        todo = todo.head(a.limit)

    print(f"{len(df):,} rows in file")
    print(f"{unresolved.sum():,} unresolved; retrying {len(todo):,} "
          f"with {a.workers} workers")
    print("this is deliberately slow — archive.org throttles concurrency\n")
    if not len(todo):
        print("nothing to retry.")
        return

    before_ok = df["fetch_status"].isin(["OK", "OK_WAYBACK"]).sum()

    results, n = {}, 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(resolve, u): u for u in todo["url"]}
        for f in tqdm(as_completed(futs), total=len(futs), desc="wayback"):
            u = futs[f]
            results[u] = f.result()
            n += 1
            if n % CHECKPOINT == 0:
                for uu, (t, x, s) in results.items():
                    i = df.index[df["url"] == uu]
                    if len(i) and s != "WAYBACK_ERR":
                        df.loc[i, ["fetched_title", "article_text",
                                   "fetch_status"]] = [t, x, s]
                df.to_csv(FILE, index=False)

    for uu, (t, x, s) in results.items():
        i = df.index[df["url"] == uu]
        if len(i) and s != "WAYBACK_ERR":
            df.loc[i, ["fetched_title", "article_text", "fetch_status"]] = [t, x, s]
    df["text_chars"] = df["article_text"].fillna("").astype(str).str.len()
    df.to_csv(FILE, index=False)

    after_ok = df["fetch_status"].isin(["OK", "OK_WAYBACK"]).sum()
    print("\n" + "=" * 58)
    print(f"usable: {before_ok:,} -> {after_ok:,}  (+{after_ok-before_ok:,})")
    print(f"rate  : {before_ok/len(df)*100:.1f}% -> {after_ok/len(df)*100:.1f}%")
    print("=" * 58)
    print("\nstatus now:")
    print(df["fetch_status"].value_counts().to_string())

    still = df["fetch_status"].astype(str).str.startswith("WAYBACK_ERR").sum()
    if still:
        print(f"\n{still:,} rows are STILL rate-limited, not dead.")
        print("Re-run this script (it only retries those) — ideally later,")
        print("or with --workers 2. Each pass recovers more.")
    dead = (df["fetch_status"] == "DEAD_confirmed").sum()
    print(f"\n{dead:,} rows are confirmed never archived — those are truly gone.")
    print("\nwhen you are done: python 07_fetch_and_label.py   (labels what fetched)")


if __name__ == "__main__":
    main()
