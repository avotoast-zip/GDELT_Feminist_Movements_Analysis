#!/usr/bin/env python3
"""
03_enrich_v3.py
---------------
Turns articles_v3.csv (from local_match_v3.py) into the analysis-ready file the
dashboard reads. This is the same enrichment that produced keyword_hits_enriched
for v2, re-run on the corrected corpus, with every derivation recomputed from
scratch rather than trusted from the old file.

Adds:
  article_stance   Support / Backlash / Mixed / Other, from the keyword stances
  article_type     News/Reporting / Opinion-Editorial / Analysis, multilingual heuristic
  has_hashtag_term already present from the matcher; passed through
  n_support, n_backlash, n_keywords  recomputed as DISTINCT-keyword counts

Note: v3 has NO generic/viol-entity artifacts to flag, because they were fixed
at source. The flag columns from v2 (flag_generic_artifact, flag_entity_fp) are
therefore gone. `clean` is not needed either — the whole v3 corpus is clean.

USAGE
    pip install pandas
    python 03_enrich_v3.py
    -> articles_v3_enriched.csv
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
import pandas as pd

IN_FILE = INTERIM / "articles_v3.csv"
OUT_FILE = INTERIM / "articles_v3_enriched.csv"


# ── stance ───────────────────────────────────────────────────────────────
def parse_stances(s):
    if pd.isna(s):
        return {}
    d = {}
    for part in str(s).split(" | "):
        m = re.match(r"^(.*) \[(.+)\]$", part.strip())
        if m:
            d[m.group(1)] = m.group(2)
    return d


# ── article type (multilingual heuristic, v2) ────────────────────────────
OPINION_SEGMENTS = [
    "opinion", "opinions", "editorial", "editorials", "op-ed", "oped",
    "commentary", "comment", "column", "columns", "columnists", "blog",
    "blogs", "voices", "viewpoint", "perspective", "commentisfree",
    "idees", "id%C3%A9es", "chroniques", "chronique", "tribunes", "tribune",
    "edito", "editos", "debats", "meinung", "meinungen", "kommentar",
    "kommentare", "kolumne", "kolumnen", "debatte", "gastbeitrag",
    "columnas", "columna", "tribuna", "firmas", "articulistas",
    "opiniao", "opini%C3%A3o", "colunas", "coluna", "cronicas",
    "opinioni", "commenti", "editoriali", "rubriche", "opinie",
    "ledare", "debatt", "kronika", "kr%C3%B6nika", "meninger", "kronikk",
    "debat", "synspunkt", "felietony", "komentarze", "yazarlar",
    "kose-yazisi", "gorus", "mneniya", "kolumnisty", "kolom", "opini", "tajuk",
]
ANALYSIS_SEGMENTS = ["analysis", "analyse", "analisis", "an%C3%A1lisis", "analiz"]


def seg_re(lst):
    return re.compile(r"/(" + "|".join(re.escape(s) for s in lst) + r")(/|$|\?)", re.I)


RX_OPINION = seg_re(OPINION_SEGMENTS)
RX_ANALYSIS = seg_re(ANALYSIS_SEGMENTS)
TITLE_PREFIX = re.compile(
    r"^\s*(opinion|editorial|op-ed|commentary|column|opini[oó]n|tribune|"
    r"[ée]dito(rial)?|chronique|meinung|kommentar|kolumne|opini[aã]o|"
    r"cr[oô]nica|opinie|ledare|kr[oö]nika|debatt|kronikk|g[oö]r[uü]ş|yorum)"
    r"\s*[:|\u2013\u2014-]", re.I)
TITLE_INLINE = re.compile(
    r"\|\s*(opinion|commentary|editorial|meinung|opini[oó]n)\s*($|\|)", re.I)


def article_type(url, title):
    u, t = str(url), str(title)
    if RX_ANALYSIS.search(u):
        return "Analysis"
    if RX_OPINION.search(u) or TITLE_PREFIX.search(t) or TITLE_INLINE.search(t):
        return "Opinion/Editorial"
    return "News/Reporting"


def main():
    df = pd.read_csv(IN_FILE)
    print(f"loaded {len(df):,} articles")

    sm = df["keyword_stances"].apply(parse_stances)
    df["n_support"] = sm.apply(lambda d: sum(1 for v in d.values() if v == "Support"))
    df["n_backlash"] = sm.apply(lambda d: sum(1 for v in d.values() if v == "Backlash"))
    df["n_keywords"] = sm.apply(len)

    def stance(r):
        if r["n_support"] > 0 and r["n_backlash"] > 0:
            return "Mixed"
        if r["n_backlash"] > 0:
            return "Backlash"
        if r["n_support"] > 0:
            return "Support"
        return "Other"
    df["article_stance"] = df.apply(stance, axis=1)

    df["article_type"] = df.apply(
        lambda r: article_type(r["url"], r["title"]), axis=1)

    df.to_csv(OUT_FILE, index=False)
    print(f"wrote {OUT_FILE}\n")
    print("article_stance:")
    print(df["article_stance"].value_counts().to_string())
    print("\narticle_type:")
    print(df["article_type"].value_counts().to_string())
    print("\nhashtag share:", f"{df['has_hashtag_term'].mean()*100:.1f}%")
    if "source_country" in df.columns:
        print("countries:", df[df["source_country"] != "UNKNOWN"]["source_country"].nunique())
    print("\nnext: python 04_build_dashboard.py")


if __name__ == "__main__":
    main()
