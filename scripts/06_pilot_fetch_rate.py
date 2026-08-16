#!/usr/bin/env python3
"""
06_pilot_fetch_rate.py
----------------------
RUN THIS BEFORE THE FULL FETCH. It answers the one question that decides
everything downstream: how many of your article links still work?

Your articles were published Oct 2017 - Dec 2019. It is now 2026. Those links
are 6 to 9 years old. Some will be dead, and the deaths will NOT be evenly
spread — small outlets, paywalled outlets and non-English outlets die faster
than nytimes.com. If half your links are dead and the dead ones cluster in
Latin America and Africa, then what you end up labeling is a biased slice of
your corpus, and you need to know that before you start, not after.

This fetches a random 1,000 URLs in parallel and reports success overall, by
country, by year, and by hashtag-vs-generic match, then extrapolates the full
run time.

USAGE
    pip install trafilatura pandas tqdm
    python 06_pilot_fetch_rate.py --n 1000 --workers 24
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

import numpy as np
import pandas as pd
import trafilatura
from tqdm import tqdm

warnings.filterwarnings("ignore")

IN_FILE = INTERIM / "fetch_list.csv"
OUT_FILE = TABLES / "pilot_fetch_results.csv"
SEED = 20260715


def fetch_one(url):
    t0 = time.time()
    try:
        dl = trafilatura.fetch_url(url)
        if dl is None:
            return ("FETCH_FAILED", 0, time.time() - t0)
        text = trafilatura.extract(dl, include_comments=False,
                                   include_tables=False, favor_recall=True)
        if not text:
            return ("NO_TEXT", 0, time.time() - t0)
        if len(text) < 400:
            # usually a paywall stub or a cookie-consent page: it "succeeds"
            # but contains nothing you can code a stance from
            return ("TOO_SHORT", len(text), time.time() - t0)
        return ("OK", len(text), time.time() - t0)
    except Exception as e:
        return (f"ERR:{type(e).__name__}", 0, time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()

    df = pd.read_csv(IN_FILE, low_memory=False)
    rng = np.random.default_rng(SEED)
    s = df.loc[rng.choice(df.index, size=min(a.n, len(df)), replace=False)].copy()
    s["year"] = pd.to_datetime(s["published"], errors="coerce", utc=True).dt.year

    print(f"corpus: {len(df):,} articles")
    print(f"fetching a random {len(s)} with {a.workers} workers\n")

    t0 = time.time()
    res = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch_one, u): u for u in s["url"]}
        for f in tqdm(as_completed(futs), total=len(futs)):
            res[futs[f]] = f.result()
    elapsed = time.time() - t0

    s["status"] = s["url"].map(lambda u: res[u][0])
    s["chars"] = s["url"].map(lambda u: res[u][1])
    s.to_csv(OUT_FILE, index=False)

    ok = s["status"].eq("OK")
    print("\n" + "=" * 60)
    print(f"OVERALL USABLE: {ok.sum()}/{len(s)}  ({ok.mean()*100:.1f}%)")
    print("=" * 60)
    print("\nwhat happened:")
    print(s["status"].value_counts().to_string())

    print("\nby year published:")
    print(s.groupby("year")["status"].apply(lambda x: (x == "OK").mean()).round(3).to_string())
    print("  -> if 2017 is much worse than 2019, link rot is time-dependent")
    print("     and your surviving corpus skews toward the later period.")

    if "source_country" in s.columns:
        g = s.groupby("source_country")["status"].agg(
            [("rate", lambda x: (x == "OK").mean()), "count"])
        g = g[g["count"] >= 10].sort_values("rate")
        if len(g):
            print("\nworst countries (>=10 sampled):")
            print(g.head(10).round(3).to_string())
            print("\nbest countries:")
            print(g.tail(5).round(3).to_string())
            print("  -> a big spread here IS the bias. Report it in your methods.")

    if "has_hashtag_term" in s.columns:
        print("\nby hashtag vs generic-vocabulary match:")
        print(s.groupby("has_hashtag_term")["status"]
              .apply(lambda x: (x == "OK").mean()).round(3).to_string())

    per_url = elapsed / len(s)
    n_full = len(df)
    hrs = n_full * per_url / 3600
    print(f"\nFULL RUN ESTIMATE at {a.workers} workers:")
    print(f"  {n_full:,} URLs  ->  ~{hrs:.1f} hours,  ~{int(n_full*ok.mean()):,} usable articles")
    print(f"  (measured {per_url*1000:.0f} ms per URL effective)")

    print("""
WHAT TO DO WITH THIS NUMBER

  above 65% usable : go ahead with the full fetch, note the bias, move on.

  40 to 65%        : go ahead, but you MUST report fetch rate by country in
                     your methods, and treat the labeled set as a biased
                     subsample of the corpus rather than the corpus itself.

  below 40%        : do not brute-force it. Options, in order of preference:
                     (a) add a Wayback Machine fallback for dead links
                         (the archive.org CDX API is free),
                     (b) narrow scope to the hashtag-matched articles,
                     (c) label a stratified sample instead of everything and
                         reweight, which is statistically honest and far cheaper.
""")


if __name__ == "__main__":
    main()
