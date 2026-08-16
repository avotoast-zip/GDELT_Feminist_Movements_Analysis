#!/usr/bin/env python3
"""
15_export_samples.py
--------------------
Pulls a readable sample of fetched articles to send to a professor.

    python 15_export_samples.py                 # 120 articles
    python 15_export_samples.py --n 200
    python 15_export_samples.py --per-country 3 # 3 from every country with text

Makes two files:
    article_samples.csv    full text, opens in Excel
    article_samples.html   formatted for reading in a browser

The sample is spread across countries and across both match types, so it shows
the range of what is in the corpus rather than 120 articles from the US.
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
import html as H

import pandas as pd

IN_FILE = PROCESSED / "labeled_articles.csv"
TEXT_CHARS = 100_000   # full text in the CSV


def load():
    cols = ["url", "source_country", "outlet", "published", "keywords_matched",
            "has_hashtag_term", "fetched_title", "article_text", "fetch_status",
            "text_chars"]
    parts = []
    for ch in pd.read_csv(IN_FILE, usecols=cols, chunksize=100_000, low_memory=False):
        ch = ch[ch["fetch_status"].isin(["OK", "OK_WAYBACK"])]
        ch = ch[ch["article_text"].notna()]
        parts.append(ch)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--per-country", type=int, default=None,
                    help="take N from every country instead of a fixed total")
    ap.add_argument("--min-chars", type=int, default=800,
                    help="skip very short texts so samples are readable")
    a = ap.parse_args()

    print("reading fetched articles (~1 min)...")
    df = load()
    df = df[df["text_chars"] >= a.min_chars]
    print(f"  usable and long enough: {len(df):,}")

    if a.per_country:
        picks = []
        for c, g in df.groupby("source_country"):
            picks.append(g.sample(n=min(a.per_country, len(g)), random_state=42))
        samp = pd.concat(picks, ignore_index=True)
    else:
        # spread across the biggest countries, then fill from everywhere
        top = df["source_country"].value_counts().head(20).index
        per = max(2, a.n // (len(top) + 4))
        picks = []
        for c in top:
            g = df[df["source_country"] == c]
            # half hashtag, half general words where possible
            for mt in [True, False]:
                gg = g[g["has_hashtag_term"] == mt]
                if len(gg):
                    picks.append(gg.sample(n=min(max(1, per // 2), len(gg)),
                                           random_state=42))
        samp = pd.concat(picks, ignore_index=True).drop_duplicates("url")
        if len(samp) < a.n:
            rest = df[~df["url"].isin(samp["url"])].sample(
                n=min(a.n - len(samp), len(df)), random_state=42)
            samp = pd.concat([samp, rest], ignore_index=True)
        samp = samp.head(a.n)

    samp = samp.sort_values(["source_country", "published"]).reset_index(drop=True)
    samp["match_type"] = samp["has_hashtag_term"].map({True: "hashtag",
                                                       False: "general words"})

    out_cols = ["source_country", "outlet", "published", "match_type",
                "keywords_matched", "fetched_title", "url", "article_text"]
    samp[out_cols].to_csv(PROCESSED / "article_samples.csv", index=False)

    # ── readable HTML ──
    css = """body{font-family:-apple-system,Segoe UI,sans-serif;max-width:860px;
    margin:0 auto;padding:24px 20px 60px;background:#FBFAF7;color:#1A1420;line-height:1.55}
    h1{font-size:23px;margin-bottom:2px}
    .sub{color:#6E5A5C;margin-bottom:22px;font-size:14px}
    .c{background:#4A1418;color:#fff;padding:8px 14px;margin:26px 0 0;
       border-radius:5px 5px 0 0;font-weight:700;font-size:13px;letter-spacing:.5px}
    .card{background:#fff;border:1px solid #E4DCDD;border-top:none;padding:15px 18px}
    .meta{font-size:12px;color:#6E5A5C;margin-bottom:5px}
    .kw{background:#F5EEE0;color:#8A6A16;padding:2px 7px;border-radius:3px;
        font-weight:600;font-size:12px}
    .tag{background:#EDE3E4;color:#7A1420;padding:2px 7px;border-radius:3px;font-size:11px}
    .ttl{font-weight:700;font-size:15px;margin:6px 0 8px}
    .txt{font-size:13.5px;white-space:pre-wrap}
    a{color:#7A1420;font-size:12px;word-break:break-all}"""
    parts = [f"<!doctype html><meta charset='utf-8'><title>Article samples</title>",
             f"<style>{css}</style>",
             f"<h1>Article samples</h1>",
             f"<div class='sub'>{len(samp)} articles from "
             f"{samp['source_country'].nunique()} countries. "
             f"Full downloaded text. Corpus: 107,481 articles with text.</div>"]
    for c, g in samp.groupby("source_country"):
        parts.append(f"<div class='c'>{H.escape(str(c))} &nbsp;·&nbsp; {len(g)}</div>")
        for _, r in g.iterrows():
            body = H.escape(str(r["article_text"])[:4000])
            parts.append(f"""<div class='card'>
              <div class='meta'>{H.escape(str(r['outlet']))} &nbsp;·&nbsp;
                {str(r['published'])[:10]} &nbsp;·&nbsp;
                <span class='tag'>{r['match_type']}</span> &nbsp;
                <span class='kw'>{H.escape(str(r['keywords_matched']))}</span></div>
              <div class='ttl'>{H.escape(str(r['fetched_title'] or '(no headline)'))[:200]}</div>
              <div class='txt'>{body}</div>
              <div style='margin-top:8px'><a href='{H.escape(str(r['url']))}'>{H.escape(str(r['url'])[:110])}</a></div>
            </div>""")
    open(PROCESSED / "article_samples.html", "w").write("\n".join(parts))

    print(f"\nwrote article_samples.csv  ({len(samp)} articles, full text)")
    print(f"      article_samples.html ({len(samp)} articles, formatted)")
    print(f"\ncountries covered: {samp['source_country'].nunique()}")
    print(samp['match_type'].value_counts().to_string())
    print("\ntop countries in the sample:")
    print(samp['source_country'].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
