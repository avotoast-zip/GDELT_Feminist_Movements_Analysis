#!/usr/bin/env python3
"""
14_spotcheck.py
---------------
Prints the model's own reasoning for the calls that matter, so you can verify
the 17% number in about 10 minutes instead of scrolling a CSV.

    python 14_spotcheck.py

Shows 10 generic-matched articles the model called NOT about the movement, then
3 it called MOVEMENT, then any hashtag-matched ones it called NOT - those last
ones are where a false negative would show up most obviously.

WHAT YOU ARE CHECKING
  For each one: do you agree? If the model is calling genuine movement coverage
  "not about the movement", the 17% is too low and you should say so rather than
  present it. If they really are routine crime reports, the number holds.
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

import pandas as pd
import textwrap

F = PROCESSED / "sample_labeled.csv"


def show(row, i):
    kw = str(row.get("keywords_matched", ""))
    ctry = str(row.get("source_country", ""))
    ttl = str(row.get("fetched_title", "") or "(no headline)")[:150]
    why = str(row.get("movement_reasoning", ""))
    txt = str(row.get("article_text", ""))[:340].replace("\n", " ")
    print(f"\n[{i}] {ctry}  ·  matched: {kw}")
    print(f"    HEADLINE: {ttl}")
    print(textwrap.fill(f"MODEL SAYS: {why}", 78,
                        initial_indent="    ", subsequent_indent="                "))
    print(textwrap.fill(f"TEXT: {txt}…", 78,
                        initial_indent="    ", subsequent_indent="          "))
    print(f"    {row.get('url','')[:76]}")


def main():
    d = pd.read_csv(F, low_memory=False)
    d = d[d["stance"].astype(str).str.len() > 0]
    d["about"] = d["is_about_movement"].astype(str).str.lower().isin(["true", "1"])

    print("=" * 78)
    print("A. GENERIC-MATCHED, model says NOT about the movement")
    print("   These are the 83%. If they are really routine crime reports,")
    print("   the 17% figure holds.")
    print("=" * 78)
    sub = d[(d["match_type"] == "generic") & (~d["about"])]
    for i, (_, r) in enumerate(sub.sample(n=min(10, len(sub)), random_state=7).iterrows(), 1):
        show(r, i)

    print("\n\n" + "=" * 78)
    print("B. GENERIC-MATCHED, model says IS about the movement")
    print("   The 17%. Check these look genuinely different from block A.")
    print("=" * 78)
    sub2 = d[(d["match_type"] == "generic") & (d["about"])]
    for i, (_, r) in enumerate(sub2.sample(n=min(3, len(sub2)), random_state=7).iterrows(), 1):
        show(r, i)

    print("\n\n" + "=" * 78)
    print("C. HASHTAG-MATCHED, model says NOT about the movement")
    print("   Possible false negatives. If these look wrong, the model is")
    print("   too strict and 17% is an underestimate.")
    print("=" * 78)
    sub3 = d[(d["match_type"] == "hashtag") & (~d["about"])]
    if len(sub3):
        for i, (_, r) in enumerate(sub3.sample(n=min(4, len(sub3)), random_state=7).iterrows(), 1):
            show(r, i)
    else:
        print("\n  (none - the model called every hashtag article movement coverage)")

    print("\n\n" + "=" * 78)
    print(f"counts: generic not-movement {len(sub)}, generic movement {len(sub2)}, "
          f"hashtag not-movement {len(sub3)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
