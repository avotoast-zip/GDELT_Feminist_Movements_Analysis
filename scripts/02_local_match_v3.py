#!/usr/bin/env python3
"""
local_match_v3.py
-----------------
Replaces BigQuery Step 2 and Step 3 entirely. Runs on your laptop.

WHY THIS EXISTS
  BigQuery already did the part it is good at: scanning 27 months of GDELT and
  cutting it down to the candidate tables (cand_2017q4 ... cand_2019q4). That
  work is done and never needs to run again.
  The remaining work — matching 1,751 keyword patterns against each candidate —
  is CPU-dense on almost no data, and the sandbox's on-demand model caps CPU
  *per byte scanned*. That ratio is the same at any scale, which is why
  splitting by quarter moved the numbers (20,555 vs 6,400) but not the verdict.
  The sandbox cannot run this join. Your laptop can, in minutes.

INPUT   cand_*.csv          the nine candidate tables, exported from BigQuery
        metoo_keywords_v3.csv   the corrected codebook (same file you uploaded)

OUTPUT  keyword_hits_v3.csv     one row per article x keyword match
        articles_v3.csv         one row per article (replaces your old export)
        + the five regression checks printed, including 4e against your old
          keyword_hits_enriched.csv if it is in the same folder

USAGE
    pip install pandas pyahocorasick regex tqdm
    python local_match_v3.py
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

import glob
import html
import os
import re as stdre
import sys

import pandas as pd
import regex
from tqdm import tqdm

try:
    import ahocorasick
    HAVE_AC = True
except ImportError:
    HAVE_AC = False

KEYWORDS = CODEBOOK / "metoo_keywords_v3.csv"
CAND_GLOB = str(RAW / "cand_*.csv")
V2_FILE = INTERIM / "keyword_hits_enriched.csv"     # optional, for check 4e
OUT_HITS = INTERIM / "keyword_hits_v3.csv"
OUT_ARTICLES = INTERIM / "articles_v3.csv"


# ── 1. load codebook, build matchers ────────────────────────────────────
def load_keywords():
    kw = pd.read_csv(KEYWORDS)
    # the same guard as the SQL: annotations can never re-enter
    bad = kw["kw_match_lower"].str.strip().isin(
        ["generic", "generic russian tags", "n/a", "none", "na"])
    bad |= kw["kw_match"].astype(str).str.match(r"^\(.*\)$")
    kw = kw[~bad].copy()

    # one compiled Unicode-boundary regex per bounded term, built once —
    # this is exactly what BigQuery could not do (it recompiled per row-pair)
    bound = {}
    for t in kw.loc[kw["match_mode"] == "regex", "kw_match_lower"].unique():
        bound[t] = regex.compile(
            r"(?:^|[^\p{L}\p{N}_])" + regex.escape(t) + r"(?:[^\p{L}\p{N}_]|$)")

    # Aho-Corasick automaton over ALL terms: one pass per blob finds every
    # term that occurs as a substring (the STRPOS gate, but all terms at once)
    terms = kw["kw_match_lower"].unique()
    if HAVE_AC:
        A = ahocorasick.Automaton()
        for t in terms:
            A.add_word(t, t)
        A.make_automaton()
        gate = lambda blob: {t for _, t in A.iter(blob)}
    else:
        # fallback: single alternation regex, longest-first
        big = stdre.compile("|".join(
            stdre.escape(t) for t in sorted(terms, key=len, reverse=True)))
        gate = lambda blob: set(big.findall(blob))
        print("NOTE: pyahocorasick not installed; using slower regex fallback."
              "  pip install pyahocorasick  is ~10x faster.")

    modes = dict(zip(kw["kw_match_lower"], kw["match_mode"]))
    return kw, bound, gate, modes


# ── 2. decode + match one candidate file ────────────────────────────────
ENTITY = stdre.compile(r"&#?[0-9A-Za-z]{1,8};")

def process(path, bound, gate, modes):
    df = pd.read_csv(path)
    # BigQuery Drive exports name columns exactly as the table: url, pub_ts,
    # domain, title_raw
    df = df.drop_duplicates(subset="url", keep="first")

    hits = []
    for row in tqdm(df.itertuples(index=False), total=len(df),
                    desc=os.path.basename(path), unit="rows"):
        title_raw = "" if pd.isna(row.title_raw) else str(row.title_raw)
        was_encoded = bool(ENTITY.search(title_raw))
        title = html.unescape(title_raw) if "&" in title_raw else title_raw
        blob = (title + " " + str(row.url)).lower()

        present = gate(blob)                      # substring gate, one pass
        for t in present:
            if modes[t] == "substr" or bound[t].search(blob):
                hits.append((row.url, row.pub_ts, row.domain,
                             title, was_encoded, t))
    return pd.DataFrame(hits, columns=[
        "url", "pub_ts", "domain", "title", "title_was_encoded",
        "kw_match_lower"])


# ── 3. main ─────────────────────────────────────────────────────────────
def main():
    files = sorted(glob.glob(CAND_GLOB))
    if not files:
        sys.exit(f"no {CAND_GLOB} files here. Export the nine cand_* tables "
                 f"from BigQuery first (see instructions).")
    print(f"candidate files: {len(files)}")

    kw, bound, gate, modes = load_keywords()
    print(f"keywords: {len(kw)} rows, {len(modes)} unique terms, "
          f"{'aho-corasick' if HAVE_AC else 'regex fallback'} gate\n")

    parts = [process(f, bound, gate, modes) for f in files]
    hits = pd.concat(parts, ignore_index=True)
    # cross-quarter dedup (an URL crawled in two quarters appears twice)
    hits = hits.sort_values("pub_ts").drop_duplicates(
        subset=["url", "kw_match_lower"], keep="first")

    # attach every codebook row for the matched term (one hit per country
    # listing, same semantics as the SQL join)
    hits = hits.merge(
        kw[["kw_match_lower", "keyword", "keyword_raw", "keyword_country",
            "is_hashtag", "stance_original", "stance_primary"]],
        on="kw_match_lower", how="left")
    hits.to_csv(OUT_HITS, index=False)
    print(f"\n{OUT_HITS}: {len(hits):,} match rows, "
          f"{hits['url'].nunique():,} distinct articles")

    # ── article-level table (Step 3 equivalent) ──
    def agg(g):
        kws = sorted(g["keyword"].unique())
        st = {k: s for k, s in zip(g["keyword"], g["stance_primary"])}
        return pd.Series({
            "title": g["title"].iloc[0],
            "published": g["pub_ts"].iloc[0],
            "outlet": g["domain"].iloc[0],
            "title_was_encoded": g["title_was_encoded"].iloc[0],
            "keywords_matched": " | ".join(kws),
            "keyword_stances": " | ".join(f"{k} [{st[k]}]" for k in kws),
            "n_keywords": len(kws),
            "n_support": sum(1 for k in kws if st[k] == "Support"),
            "n_backlash": sum(1 for k in kws if st[k] == "Backlash"),
            "has_hashtag_term": bool(g["is_hashtag"].any()),
        })
    print("aggregating to article level (a few minutes)...")
    articles = hits.groupby("url", sort=False).apply(agg).reset_index()

    # attach source_country from the GDELT lookup if it is present, so the
    # dashboard map has data. Falls back to UNKNOWN if the lookup is missing.
    if os.path.exists(CODEBOOK / "sourcesbycountry.csv"):
        look = pd.read_csv(CODEBOOK / "sourcesbycountry.csv")
        look.columns = [c.strip() for c in look.columns]
        look["Domain"] = look["Domain"].astype(str).str.lower().str.strip()
        look = look.dropna(subset=["Domain"]).drop_duplicates("Domain", keep="first")
        cmap = dict(zip(look["Domain"], look["CountryName"]))
        articles["source_country"] = (articles["outlet"].astype(str).str.lower()
                                      .str.strip().map(cmap).fillna("UNKNOWN"))
        print(f"  source_country attached; "
              f"{(articles['source_country']!='UNKNOWN').sum():,} resolved")
    else:
        articles["source_country"] = "UNKNOWN"
        print("  NOTE: sourcesbycountry.csv not found — source_country set to "
              "UNKNOWN (map will be blank). Export the lookup and rerun, or run "
              "add_country.py.")
    articles.to_csv(OUT_ARTICLES, index=False)
    print(f"{OUT_ARTICLES}: {len(articles):,} articles")

    # ── regression checks 4a-4d ──
    print("\n" + "=" * 62)
    print("REGRESSION CHECKS")
    print("=" * 62)
    g = (hits["kw_match_lower"] == "generic").sum()
    print(f"4a  'generic' rows            : {g}   (must be 0)")

    v = hits[hits["kw_match_lower"] == "viol"]
    pat = regex.compile(r"(?:^|[^\p{L}\p{N}_])viol(?:[^\p{L}\p{N}_]|$)")
    bad = sum(1 for _, r in v.iterrows()
              if not pat.search((str(r["title"]) + " " + str(r["url"])).lower()))
    print(f"4b  bad 'viol' rows           : {bad}   (must be 0; v2 had 5,984)")

    for t in ["米兔", "미투", "私も", "我也是", "몰카"]:
        n = (hits["kw_match_lower"] == t).sum()
        print(f"4c  {t:6s} matches            : {n:,}   (v2 had 0)")

    enc = hits[hits["title_was_encoded"] &
               ~hits["kw_match_lower"].str.fullmatch(r"[\x00-\x7f]+")]
    print(f"4d  non-ASCII terms recovered from encoded titles: {len(enc):,} rows")

    # ── 4e: v2 vs v3 by country needs the outlet->country lookup ──
    if os.path.exists(V2_FILE):
        v2 = pd.read_csv(V2_FILE)
        c2 = v2.groupby("source_country")["url"].nunique()
        # v3 country via the same GDELT lookup, downloaded once
        lk = CODEBOOK / "sourcesbycountry.csv"
        if os.path.exists(lk):
            look = pd.read_csv(lk)
            look["Domain"] = look["Domain"].str.lower()
            articles2 = articles.merge(look[["Domain", "CountryName"]],
                                       left_on="outlet", right_on="Domain",
                                       how="left")
            articles2["source_country"] = articles2["CountryName"].fillna("UNKNOWN")
            c3 = articles2.groupby("source_country")["url"].nunique()
            cmp = pd.DataFrame({"v2": c2, "v3": c3}).fillna(0).astype(int)
            cmp["delta"] = cmp["v3"] - cmp["v2"]
            cmp["pct"] = (cmp["delta"] / cmp["v2"].replace(0, pd.NA) * 100).round(1)
            cmp = cmp.reindex(cmp["delta"].abs().sort_values(ascending=False).index)
            cmp.to_csv(TABLES / "check_4e_v2_vs_v3.csv")
            print("\n4e  v2 vs v3 by country (top 25 by |delta|) "
                  "-> check_4e_v2_vs_v3.csv")
            print(cmp.head(25).to_string())
        else:
            print("\n4e  skipped: download the lookup first —")
            print("    in BigQuery run: SELECT Domain, CountryName FROM "
                  "`gdelt-bq.extra.sourcesbycountry`")
            print("    save the result as sourcesbycountry.csv next to this "
                  "script and re-run (only 4e reruns instantly).")
    else:
        print(f"\n4e  skipped: {V2_FILE} not found in this folder.")

    print("\nnext: run the article-type/stance enrichment on articles_v3.csv,")
    print("rebuild the dashboard, and retire every v2 artifact.")


if __name__ == "__main__":
    main()
