#!/usr/bin/env python3
"""
16_rematch_fulltext.py
----------------------
Searches the downloaded article text for every keyword in the codebook.

WHY THIS MATTERS

Until now we only searched the web address and the headline. 80% of articles
have no headline saved, so most were matched on the web address alone.

A web address is short. It holds one or two words. The article body holds
hundreds.

This is why Brazil looked wrong. #EleNao was one of the largest feminist
protests in Brazilian history. It matched 4 articles in the whole corpus.
Not because Brazil did not cover it. Because the hashtag was in the article
text, and we never looked there.

This script fixes that. It reads the downloaded text of all 107,481 articles
and finds every keyword that appears.

WHAT CHANGES AFTER THIS

  - Articles gain keywords they always had
  - The hashtag vs general-words split will move
  - Countries with low hashtag rates may rise
  - Some articles will match both types for the first time

INPUT
    labeled_articles.csv      the downloaded text
    metoo_keywords_v3.csv     the codebook

OUTPUT
    rematched_articles.csv    same articles, new columns:
                                kw_urltitle   what we found before
                                kw_fulltext   what the article text contains
                                kw_combined   both together
                                hashtag_urltitle / hashtag_fulltext
                                gained_hashtag  True if the text revealed one
    rematch_report.txt        before and after numbers

RUN
    pip install pandas regex pyahocorasick
    python 16_rematch_fulltext.py

Takes about 10 to 20 minutes. No API. No cost.
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

import html as H
import re
import sys
import unicodedata

import pandas as pd
import regex

try:
    import ahocorasick
    HAVE_AC = True
except ImportError:
    HAVE_AC = False

IN_FILE = PROCESSED / "labeled_articles.csv"
KEYWORDS = CODEBOOK / "metoo_keywords_v3.csv"
OUT_FILE = PROCESSED / "rematched_articles.csv"
REPORT = REPORTS / "rematch_report.txt"

MAX_TEXT = 20_000     # chars of body text to search per article


def build_matchers():
    kw = pd.read_csv(KEYWORDS)
    bad = kw["kw_match_lower"].astype(str).str.strip().isin(
        ["generic", "generic russian tags", "n/a", "none", "na"])
    kw = kw[~bad].copy()

    bound = {}
    for t in kw.loc[kw["match_mode"] == "regex", "kw_match_lower"].unique():
        bound[str(t)] = regex.compile(
            r"(?:^|[^\p{L}\p{N}_])" + regex.escape(str(t)) + r"(?:[^\p{L}\p{N}_]|$)")

    terms = [str(t) for t in kw["kw_match_lower"].unique()]
    if HAVE_AC:
        A = ahocorasick.Automaton()
        for t in terms:
            A.add_word(t, t)
        A.make_automaton()
        gate = lambda blob: {t for _, t in A.iter(blob)}
    else:
        print("NOTE: pyahocorasick not installed. This will be slow.")
        print("      pip install pyahocorasick")
        big = re.compile("|".join(re.escape(t) for t in
                                  sorted(terms, key=len, reverse=True)))
        gate = lambda blob: set(big.findall(blob))

    modes = dict(zip(kw["kw_match_lower"].astype(str), kw["match_mode"]))
    display = {}
    for _, r in kw.iterrows():
        display.setdefault(str(r["kw_match_lower"]), r["keyword"])
    is_tag = {}
    for _, r in kw.iterrows():
        is_tag.setdefault(str(r["kw_match_lower"]), bool(r["is_hashtag"]))
    return kw, bound, gate, modes, display, is_tag


def norm(s):
    s = str(s)
    if "&" in s:
        s = H.unescape(s)
    return s.lower()


def main():
    kw, bound, gate, modes, display, is_tag = build_matchers()
    print(f"codebook: {len(modes)} unique terms "
          f"({'aho-corasick' if HAVE_AC else 'regex fallback'})\n")

    cols = ["url", "source_country", "outlet", "published", "keywords_matched",
            "has_hashtag_term", "fetched_title", "article_text", "fetch_status",
            "text_chars"]
    out_chunks = []
    n_seen = 0
    print("scanning article text...")
    for ch in pd.read_csv(IN_FILE, usecols=cols, chunksize=20_000, low_memory=False):
        ch = ch[ch["fetch_status"].isin(["OK", "OK_WAYBACK"])]
        ch = ch[ch["article_text"].notna()]
        if not len(ch):
            continue

        found_terms = []
        for txt in ch["article_text"]:
            blob = norm(txt)[:MAX_TEXT]
            hits = set()
            for t in gate(blob):
                if modes.get(t) == "substr" or (t in bound and bound[t].search(blob)):
                    hits.add(t)
            found_terms.append(hits)

        ch = ch.copy()
        ch["_ft"] = found_terms
        ch["kw_fulltext"] = ch["_ft"].apply(
            lambda S: " | ".join(sorted(display.get(t, t) for t in S)))
        ch["hashtag_fulltext"] = ch["_ft"].apply(
            lambda S: any(is_tag.get(t, False) for t in S))
        ch["n_kw_fulltext"] = ch["_ft"].apply(len)
        out_chunks.append(ch.drop(columns=["_ft", "article_text"]))
        n_seen += len(ch)
        print(f"  {n_seen:,} scanned", flush=True)

    df = pd.concat(out_chunks, ignore_index=True)
    df = df.rename(columns={"keywords_matched": "kw_urltitle",
                            "has_hashtag_term": "hashtag_urltitle"})

    def combine(r):
        a = set(x.strip() for x in str(r["kw_urltitle"]).split(" | ") if x.strip())
        b = set(x.strip() for x in str(r["kw_fulltext"]).split(" | ") if x.strip())
        return " | ".join(sorted(a | b))
    df["kw_combined"] = df.apply(combine, axis=1)
    df["hashtag_combined"] = df["hashtag_urltitle"].fillna(False) | df["hashtag_fulltext"]
    df["gained_hashtag"] = (~df["hashtag_urltitle"].fillna(False)) & df["hashtag_fulltext"]
    df.to_csv(OUT_FILE, index=False)

    # ── report ──
    L = []
    def w(s=""):
        print(s)
        L.append(s)

    n = len(df)
    w("=" * 64)
    w("RE-MATCHING AGAINST FULL ARTICLE TEXT")
    w("=" * 64)
    w(f"articles with downloaded text: {n:,}")
    w("")
    w("HASHTAG SHARE")
    w(f"  before (web address + headline): {df['hashtag_urltitle'].fillna(False).mean()*100:5.1f}%")
    w(f"  after  (+ article text)        : {df['hashtag_combined'].mean()*100:5.1f}%")
    w(f"  articles that gained a hashtag : {df['gained_hashtag'].sum():,}")
    w("")
    w("KEYWORDS PER ARTICLE")
    before = df["kw_urltitle"].astype(str).apply(
        lambda s: len([x for x in s.split(" | ") if x.strip()]))
    after = df["kw_combined"].astype(str).apply(
        lambda s: len([x for x in s.split(" | ") if x.strip()]))
    w(f"  before: {before.mean():.2f}    after: {after.mean():.2f}")
    w("")

    w("BRAZILIAN HASHTAGS  (the test case)")
    BR = ["#EleNão", "#MeuPrimeiroAssédio", "#ChegaDeFiuFiu", "#MeuAmigoSecreto",
          "#MexeuComUmaMexeuComTodas", "#DeixaElaTrabalhar", "#PrimeiroAssédio",
          "#EuTambém"]
    for t in BR:
        b = df["kw_urltitle"].astype(str).str.contains(re.escape(t), na=False).sum()
        a2 = df["kw_combined"].astype(str).str.contains(re.escape(t), na=False).sum()
        w(f"  {t:28s} before {b:5,}   after {a2:5,}")
    w("")

    w("HASHTAG SHARE BY COUNTRY  (n>=300 with text)")
    g = df[df["source_country"] != "UNKNOWN"].groupby("source_country").agg(
        n=("url", "size"),
        before=("hashtag_urltitle", lambda s: s.fillna(False).mean()),
        after=("hashtag_combined", "mean"))
    g = g[g["n"] >= 300]
    g["change_pp"] = (g["after"] - g["before"]) * 100
    g = g.sort_values("change_pp", ascending=False)
    w(f"  {'country':20s} {'n':>6s} {'before':>7s} {'after':>7s} {'change':>8s}")
    for c, r in g.head(15).iterrows():
        w(f"  {c[:20]:20s} {int(r['n']):6,} {r['before']*100:6.1f}% "
          f"{r['after']*100:6.1f}% {r['change_pp']:+7.1f}pp")
    w("")
    w("TERMS MOST OFTEN FOUND ONLY IN THE BODY TEXT")
    from collections import Counter
    cnt = Counter()
    for ut, ft in zip(df["kw_urltitle"].astype(str), df["kw_fulltext"].astype(str)):
        a = set(x.strip() for x in ut.split(" | ") if x.strip())
        b = set(x.strip() for x in ft.split(" | ") if x.strip())
        for t in (b - a):
            cnt[t] += 1
    for t, c2 in cnt.most_common(20):
        w(f"  {t:34s} {c2:6,}")

    w("")
    w(f"wrote {OUT_FILE}")
    open(REPORT, "w").write("\n".join(L))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
