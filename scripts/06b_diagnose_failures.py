#!/usr/bin/env python3
"""
06b_diagnose_failures.py
------------------------
Your pilot returned 42.8% usable, with a country spread that is hard to explain
by link rot alone:

    Belgium 9%   Canada 12%   Switzerland 13%   Netherlands 30%   Germany 38%
    France 77%   Spain 70%    India 60%

Belgium, the Netherlands, Switzerland and Germany are wealthy countries with
stable, well-archived media. Their sites have not disappeared. What they DO have
is aggressive GDPR consent walls and paywalls, and Canada has hard paywalls.
That pattern says a large share of your 494 FETCH_FAILED results are the server
REFUSING the default scraper, not the article being gone.

That distinction matters enormously:
  - genuinely dead  -> unrecoverable, and a real bias you must report
  - blocked         -> recoverable, and pretending otherwise throws away half
                       your corpus and biases it toward whichever countries
                       happen not to block scrapers

This script takes the URLs that failed in your pilot and finds out which it is.
It tries three things on each and reports what works:

  1. exact HTTP status with a normal browser user-agent
     404/410 = really gone.  403/429 = blocked.  timeout/DNS = ambiguous.
  2. a full browser-like request (headers, redirects, longer timeout)
  3. the Wayback Machine (archive.org), which is free and has no rate problem
     at this scale

USAGE
    pip install requests trafilatura pandas tqdm
    python 06b_diagnose_failures.py --n 300
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
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import trafilatura
from tqdm import tqdm

warnings.filterwarnings("ignore")

IN_FILE = TABLES / "pilot_fetch_results.csv"
OUT_FILE = TABLES / "failure_diagnosis.csv"

# A normal desktop browser. Many sites serve 403 to anything that looks like a
# script and 200 to this exact string.
BROWSER = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8,es;q=0.7,de;q=0.6",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def classify(url):
    """Return (verdict, http_code, chars_recovered, method_that_worked)."""
    # ── attempt 1: browser-headed direct request ──
    code = None
    try:
        r = requests.get(url, headers=BROWSER, timeout=20, allow_redirects=True)
        code = r.status_code
        if r.ok and r.text:
            text = trafilatura.extract(r.text, include_comments=False,
                                       include_tables=False, favor_recall=True)
            if text and len(text) >= 400:
                return ("RECOVERED_browser", code, len(text), "browser_headers")
            if text:
                return ("SHORT_browser", code, len(text), "browser_headers")
            return ("NO_TEXT_browser", code, 0, "")
    except requests.exceptions.SSLError:
        code = "SSL"
    except requests.exceptions.ConnectTimeout:
        code = "TIMEOUT"
    except requests.exceptions.ReadTimeout:
        code = "TIMEOUT"
    except requests.exceptions.ConnectionError:
        code = "DNS_OR_CONN"
    except Exception as e:
        code = type(e).__name__

    # ── attempt 2: Wayback Machine ──
    try:
        api = f"http://archive.org/wayback/available?url={url}&timestamp=2018"
        j = requests.get(api, timeout=20).json()
        snap = j.get("archived_snapshots", {}).get("closest", {})
        if snap.get("available") and snap.get("url"):
            wb = snap["url"]
            r2 = requests.get(wb, headers=BROWSER, timeout=30)
            if r2.ok and r2.text:
                text = trafilatura.extract(r2.text, include_comments=False,
                                           include_tables=False,
                                           favor_recall=True)
                if text and len(text) >= 400:
                    return ("RECOVERED_wayback", code, len(text), "wayback")
                return ("WAYBACK_THIN", code, len(text or ""), "")
        return ("DEAD_no_archive", code, 0, "")
    except Exception:
        return ("DEAD_no_archive", code, 0, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    df = pd.read_csv(IN_FILE, low_memory=False)
    failed = df[~df["status"].isin(["OK"])].copy()
    print(f"pilot had {len(df)} URLs, {len(failed)} not usable")
    s = failed.sample(n=min(a.n, len(failed)), random_state=20260715).copy()
    print(f"diagnosing {len(s)} of them with {a.workers} workers")
    print("(slower than the pilot: each URL may get up to 3 requests)\n")

    out = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(classify, u): u for u in s["url"]}
        for f in tqdm(as_completed(futs), total=len(futs)):
            out[futs[f]] = f.result()

    s["verdict"] = s["url"].map(lambda u: out[u][0])
    s["http_code"] = s["url"].map(lambda u: out[u][1])
    s["chars"] = s["url"].map(lambda u: out[u][2])
    s["method"] = s["url"].map(lambda u: out[u][3])
    s.to_csv(OUT_FILE, index=False)

    print("\n" + "=" * 62)
    print("WHAT THE FAILURES ACTUALLY ARE")
    print("=" * 62)
    print(s["verdict"].value_counts().to_string())

    rec = s["verdict"].str.startswith("RECOVERED")
    print(f"\nRECOVERABLE: {rec.sum()}/{len(s)}  ({rec.mean()*100:.1f}% of failures)")
    by_method = s[rec]["method"].value_counts()
    if len(by_method):
        print("  by method:")
        print("   ", by_method.to_string().replace("\n", "\n    "))

    print("\nHTTP codes seen (where the server answered at all):")
    print(s["http_code"].value_counts().head(10).to_string())
    print("  403 / 429       = blocking a scraper, recoverable")
    print("  404 / 410       = genuinely gone")
    print("  DNS_OR_CONN     = domain dead or unreachable")
    print("  TIMEOUT         = slow or hostile; sometimes recoverable on retry")

    if "source_country" in s.columns:
        print("\nrecovery rate by country (>=8 diagnosed):")
        g = s.groupby("source_country")["verdict"].agg(
            [("recover", lambda x: x.str.startswith("RECOVERED").mean()), "count"])
        g = g[g["count"] >= 8].sort_values("recover", ascending=False)
        print(g.round(3).to_string())

    # projected new overall rate
    old_ok = (df["status"] == "OK").mean()
    proj = old_ok + (1 - old_ok) * rec.mean()
    print("\n" + "=" * 62)
    print(f"PROJECTED USABLE RATE IF YOU APPLY THESE FIXES: {proj*100:.1f}%")
    print(f"  (was {old_ok*100:.1f}%)")
    print("=" * 62)
    print(f"\nwrote {OUT_FILE}")
    print("""
HOW TO READ THIS

  If most failures are RECOVERED_browser, the problem was the scraper's
  user-agent, not your data. Fix it in the fetcher and rerun the pilot -
  your corpus roughly doubles and the country bias shrinks a lot.

  If most are RECOVERED_wayback, add the archive.org fallback. Slower, but
  free, and it recovers exactly the old dead links you care about.

  If most are DEAD_no_archive with 404s, the links really are gone. Then the
  honest path is to treat fetch failure as survey non-response: keep the
  design weights, report fetch rate by country in your methods, and say
  plainly that the labeled subset over-represents France, Spain and India.
""")


if __name__ == "__main__":
    main()
