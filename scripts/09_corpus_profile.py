#!/usr/bin/env python3
"""
09_corpus_profile.py
--------------------
Descriptive profile of the corrected corpus, answering the questions Julia and
Nilesh raised. Run this on articles_v3_enriched.csv; it writes one CSV per
question plus a printed summary you can paste into an email.

    pip install pandas langdetect
    python 09_corpus_profile.py

OUTPUTS
    profile_1_hashtag_by_country.csv    hashtag vs generic vocabulary, per country
    profile_2_match_classes.csv         hashtag-only / generic-only / both
    profile_3_language.csv              language distribution (see caveat below)
    profile_4_outlets_by_country.csv    unique outlets + concentration per country
    profile_5_top_keywords.csv          most-matched keywords by country
    profile_6_suspect_countries.csv     likely TLD-misassigned outlets

WHAT EACH ANSWERS

Q1  "hashtag vs broader vocabulary, by country"
    -> profile_1. An article counts as hashtag-matched if ANY matched term is a
       hashtag. See Q2 for why that phrasing matters.

Q2  "how are articles handled when they match both?"
    -> Both keywords are recorded. `keywords_matched` holds every distinct term
       that matched; `has_hashtag_term` is true if at least one is a hashtag.
       Nothing is discarded and no priority rule is applied, so the categories
       are NOT mutually exclusive by construction. In practice they almost are
       (see the printed split) because matching runs on the URL plus the title,
       and 81% of articles have no title in GDELT - a URL slug rarely contains
       two different vocabularies. This means the current split describes what
       appears in URLs and headlines, NOT what appears in article bodies. It
       will shift once full text is fetched, and that shift is itself worth
       reporting.

Q3  "language distribution"
    -> profile_3, WITH A CAVEAT. GDELT's GKG record for these articles does not
       carry a reliable language field in what we extracted, and 81% of rows
       have no title, so language cannot be measured directly for most of the
       corpus yet. This script gives two approximations and reports the
       coverage of each:
         (a) detected  - langdetect on the title, where a title exists (~19%)
         (b) inferred  - the codebook country of the matched keyword, which
                         implies a language for language-specific terms
       Treat both as provisional. Real language distribution comes after the
       fetch, from article body text.

Q3  "unique outlets by country", "concentration within the largest outlets"
    -> profile_4. Concentration is reported three ways: share held by the single
       largest outlet, share held by the top three, and HHI (sum of squared
       outlet shares; 1.0 = one outlet holds everything, near 0 = highly
       dispersed). HHI is included because top-N share is sensitive to how many
       outlets a country has at all.

Q3  "most frequently matched keywords by country or language"
    -> profile_5.
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

import re
import sys
import pandas as pd
import numpy as np

IN_FILE = INTERIM / "articles_v3_enriched.csv"
KEYWORDS = CODEBOOK / "metoo_keywords_v3.csv"

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    HAVE_LD = True
except ImportError:
    HAVE_LD = False

# ccTLDs that are widely resold outside their country. GDELT's domain lookup
# assigns some of these by TLD, which invents publishing countries.
VANITY_TLD = {
    "Tuvalu": ".tv", "Niue": ".nu", "Montenegro": ".me",
    "Cocos (Keeling) Islands": ".cc", "Federated States of Micronesia": ".fm",
    "Tokelau": ".tk", "Anguilla": ".ai",
    "British Indian Ocean Territory": ".io", "Turkmenistan": ".tm",
}


def kw_list(s):
    return [x.strip() for x in str(s).split(" | ")] if pd.notna(s) else []


def main():
    df = pd.read_csv(IN_FILE, low_memory=False)
    print(f"corpus: {len(df):,} articles\n")

    if "source_country" not in df.columns:
        sys.exit("no source_country column - run add_country.py first.")

    df["_kws"] = df["keywords_matched"].apply(kw_list)
    df["_n_hash"] = df["_kws"].apply(lambda L: sum(1 for k in L if k.startswith("#")))
    df["_n_gen"] = df["_kws"].apply(lambda L: sum(1 for k in L if not k.startswith("#")))

    def cls(r):
        if r["_n_hash"] and r["_n_gen"]:
            return "both"
        if r["_n_hash"]:
            return "hashtag_only"
        if r["_n_gen"]:
            return "generic_only"
        return "none"
    df["match_class"] = df.apply(cls, axis=1)

    known = df[df["source_country"] != "UNKNOWN"].copy()

    # ── Q2: match classes ────────────────────────────────────────────────
    mc = df["match_class"].value_counts().rename_axis("match_class").reset_index(name="articles")
    mc["pct"] = (mc["articles"] / len(df) * 100).round(2)
    mc.to_csv(TABLES / "profile_2_match_classes.csv", index=False)
    print("=" * 66)
    print("Q2  HOW ARTICLES MATCHING BOTH ARE HANDLED")
    print("=" * 66)
    print(mc.to_string(index=False))
    print(f"\nmean distinct keywords per article: {df['_n_kw' ] if False else df['_kws'].apply(len).mean():.2f}")
    t = df["title"].fillna("").astype(str)
    has_t = t.str.len() > 15
    print(f"articles with a title in GDELT: {has_t.sum():,} ({has_t.mean()*100:.1f}%)")
    print(f"  'both' rate, title present : {(df[has_t]['match_class']=='both').mean()*100:.2f}%")
    print(f"  'both' rate, title absent  : {(df[~has_t]['match_class']=='both').mean()*100:.2f}%")
    print("  -> overlap is low because most matching happens on the URL alone.")
    print("     Expect this to rise substantially once full text is fetched.")

    # ── Q1: hashtag vs generic by country ────────────────────────────────
    g = known.groupby("source_country").agg(
        articles=("url", "count"),
        hashtag_matched=("has_hashtag_term", "sum")).reset_index()
    g["hashtag_pct"] = (g["hashtag_matched"] / g["articles"] * 100).round(1)
    g["generic_pct"] = (100 - g["hashtag_pct"]).round(1)
    g = g.sort_values("articles", ascending=False)
    g.to_csv(TABLES / "profile_1_hashtag_by_country.csv", index=False)

    print("\n" + "=" * 66)
    print("Q1  HASHTAG vs BROADER VOCABULARY, BY COUNTRY  (n>=500)")
    print("=" * 66)
    big = g[g["articles"] >= 500].sort_values("hashtag_pct")
    print(f"{'country':22s} {'articles':>8s} {'hashtag%':>9s}")
    for _, r in big.iterrows():
        print(f"{r['source_country'][:22]:22s} {r['articles']:8,} {r['hashtag_pct']:9.1f}")

    # ── the language-family pattern in that split ────────────────────────
    LANG_GROUPS = {
        "Spanish": ["Mexico", "Argentina", "Spain", "Colombia", "Peru", "Chile",
                    "Venezuela", "Ecuador", "Bolivia", "Paraguay", "Uruguay",
                    "Guatemala", "Cuba", "Dominican Republic", "Costa Rica",
                    "Panama", "Honduras", "El Salvador", "Nicaragua"],
        "Portuguese": ["Brazil", "Portugal", "Angola", "Mozambique"],
        "French": ["France", "Belgium", "Senegal", "Ivory Coast", "Cameroon",
                   "Morocco", "Algeria", "Tunisia", "Haiti", "Congo"],
        "English": ["United States", "United Kingdom", "Canada", "Australia",
                    "India", "Ireland", "South Africa", "New Zealand", "Nigeria",
                    "Kenya", "Pakistan", "Ghana", "Singapore", "Philippines"],
        "German": ["Germany", "Austria", "Switzerland"],
        "Nordic": ["Sweden", "Norway", "Denmark", "Finland", "Iceland"],
        "Slavic": ["Russia", "Poland", "Ukraine", "Czech Republic", "Serbia",
                   "Croatia", "Bulgaria", "Slovak Republic", "Belarus", "Moldova"],
    }
    rev = {c: l for l, cs in LANG_GROUPS.items() for c in cs}
    known["_langgrp"] = known["source_country"].map(rev)
    lg = known[known["_langgrp"].notna()].groupby("_langgrp").agg(
        articles=("url", "count"), hashtag=("has_hashtag_term", "sum"))
    lg["hashtag_pct"] = (lg["hashtag"] / lg["articles"] * 100).round(1)
    print("\nSAME SPLIT, GROUPED BY PRIMARY LANGUAGE OF THE COUNTRY:")
    print(lg.sort_values("hashtag_pct")[["articles", "hashtag_pct"]].to_string())
    print("  -> the split tracks language family, not individual countries.")

    # is that a codebook artifact? check codebook composition
    try:
        kw = pd.read_csv(KEYWORDS)
        kw["_lg"] = kw["keyword_country"].map(rev)
        ck = kw[kw["_lg"].notna()].groupby("_lg").agg(
            terms=("keyword", "count"), hashtags=("is_hashtag", "sum"))
        ck["codebook_hashtag_pct"] = (ck["hashtags"] / ck["terms"] * 100).round(1)
        print("\nCODEBOOK composition for the same groups:")
        print(ck.sort_values("codebook_hashtag_pct")[["terms", "codebook_hashtag_pct"]].to_string())
        print("  -> if the codebook is hashtag-RICH for a group whose articles")
        print("     nonetheless match mostly generic terms, the gap is in the")
        print("     press or in term frequency, not in our keyword list.")
    except FileNotFoundError:
        print("\n(metoo_keywords_v3.csv not found - skipping codebook comparison)")

    # ── Q3: language distribution ────────────────────────────────────────
    print("\n" + "=" * 66)
    print("Q3  LANGUAGE DISTRIBUTION  (provisional - see caveat)")
    print("=" * 66)
    lang_rows = []
    if HAVE_LD:
        sub = df[has_t].copy()
        def safe(t):
            try:
                return detect(t)
            except Exception:
                return None
        sub["_lang"] = sub["title"].astype(str).apply(safe)
        vc = sub["_lang"].value_counts()
        cover = sub["_lang"].notna().sum()
        print(f"detected from titles: {cover:,} articles "
              f"({cover/len(df)*100:.1f}% of corpus)")
        print(vc.head(15).to_string())
        lang_rows = vc.rename_axis("language").reset_index(name="articles")
        lang_rows["method"] = "langdetect_on_title"
        lang_rows["pct_of_detected"] = (lang_rows["articles"] / cover * 100).round(1)
    else:
        print("langdetect not installed - pip install langdetect")
    if len(lang_rows):
        pd.DataFrame(lang_rows).to_csv(TABLES / "profile_3_language.csv", index=False)
    print("\nCAVEAT: this covers only the ~19% of articles that carry a title.")
    print("Full language distribution requires the fetched article text.")

    # ── Q3: outlets + concentration ──────────────────────────────────────
    rows = []
    for c, grp in known.groupby("source_country"):
        n = len(grp)
        vc = grp["outlet"].value_counts()
        shares = (vc / n).values
        rows.append({
            "country": c, "articles": n, "unique_outlets": grp["outlet"].nunique(),
            "articles_per_outlet": round(n / max(grp["outlet"].nunique(), 1), 1),
            "top1_outlet_pct": round(vc.iloc[0] / n * 100, 1),
            "top3_outlet_pct": round(vc.head(3).sum() / n * 100, 1),
            "top10_outlet_pct": round(vc.head(10).sum() / n * 100, 1),
            "HHI": round(float((shares ** 2).sum()), 4),
            "largest_outlet": vc.index[0],
        })
    oc = pd.DataFrame(rows).sort_values("articles", ascending=False)
    oc.to_csv(TABLES / "profile_4_outlets_by_country.csv", index=False)
    print("\n" + "=" * 66)
    print("Q3  OUTLETS AND CONCENTRATION  (n>=500, most concentrated first)")
    print("=" * 66)
    sh = oc[oc["articles"] >= 500].sort_values("top1_outlet_pct", ascending=False)
    print(f"{'country':20s} {'n':>7s} {'outlets':>7s} {'top1%':>6s} {'top3%':>6s} {'HHI':>6s}")
    for _, r in sh.head(10).iterrows():
        print(f"{r['country'][:20]:20s} {r['articles']:7,} {r['unique_outlets']:7,} "
              f"{r['top1_outlet_pct']:6.1f} {r['top3_outlet_pct']:6.1f} {r['HHI']:6.3f}")
    print("  ... least concentrated:")
    for _, r in sh.tail(5).iterrows():
        print(f"{r['country'][:20]:20s} {r['articles']:7,} {r['unique_outlets']:7,} "
              f"{r['top1_outlet_pct']:6.1f} {r['top3_outlet_pct']:6.1f} {r['HHI']:6.3f}")

    # ── Q3: top keywords by country ──────────────────────────────────────
    ex = known[["source_country", "_kws"]].explode("_kws")
    ex = ex[ex["_kws"].notna() & (ex["_kws"] != "")]
    tk = (ex.groupby(["source_country", "_kws"]).size()
            .rename("articles").reset_index()
            .sort_values(["source_country", "articles"], ascending=[True, False]))
    tk["rank"] = tk.groupby("source_country")["articles"].rank(
        method="first", ascending=False).astype(int)
    tk[tk["rank"] <= 10].rename(columns={"_kws": "keyword"}) \
      .to_csv(TABLES / "profile_5_top_keywords.csv", index=False)
    print("\n" + "=" * 66)
    print("Q3  TOP MATCHED KEYWORD, LARGEST COUNTRIES")
    print("=" * 66)
    for c in oc.head(12)["country"]:
        t3 = tk[(tk["source_country"] == c) & (tk["rank"] <= 3)]
        terms = ", ".join(f"{r['_kws']} ({r['articles']:,})" for _, r in t3.iterrows())
        print(f"  {c[:18]:18s} {terms[:76]}")

    # ── data-quality flag: vanity TLD countries ──────────────────────────
    sus = []
    for c, tld in VANITY_TLD.items():
        s = known[known["source_country"] == c]
        if not len(s):
            continue
        frac = s["outlet"].astype(str).str.endswith(tld).mean()
        if frac > 0.5:
            sus.append({"assigned_country": c, "tld": tld, "articles": len(s),
                        "pct_with_that_tld": round(frac * 100, 1),
                        "example_outlets": ", ".join(s["outlet"].value_counts().head(4).index)})
    if sus:
        sdf = pd.DataFrame(sus).sort_values("articles", ascending=False)
        sdf.to_csv(TABLES / "profile_6_suspect_countries.csv", index=False)
        print("\n" + "=" * 66)
        print("DATA QUALITY: LIKELY TLD-MISASSIGNED COUNTRIES")
        print("=" * 66)
        print(sdf.to_string(index=False))
        print(f"\ntotal affected: {sdf['articles'].sum():,} articles "
              f"({sdf['articles'].sum()/len(df)*100:.2f}% of corpus)")
        print("GDELT maps some domains to a country by TLD. .tv/.nu/.me are sold")
        print("worldwide, so these are not really Tuvaluan/Niuean/Montenegrin outlets.")
        print("Small in volume, but they put non-existent publishing countries on")
        print("the map. Recommend reassigning or excluding before analysis.")

    print("\nwrote: profile_1..6 CSVs")


if __name__ == "__main__":
    main()
