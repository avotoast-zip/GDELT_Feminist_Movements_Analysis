#!/usr/bin/env python3
"""
05a_dedupe_prefetch.py
----------------------
Cheap deduplication BEFORE fetching, so you don't spend hours downloading the
same article twenty times.

This handles only what can be judged without the article body:

  Stage A  URL variants — http vs https, www vs not, trailing slash, tracking
           query strings. Same page, different string. Always safe to collapse.

  Stage B  Identical titles from the SAME outlet — a re-crawl or a duplicate
           listing. Also safe to collapse.

  Stage C  Identical titles from DIFFERENT outlets — wire syndication (AP,
           Reuters, AFP). Almost always the same text, but this is where your
           editorial-overlay rule could bite: a paper could run the wire story
           under the same headline and append a comment. So these are CLUSTERED
           and FLAGGED, not collapsed, unless you pass --collapse-syndication.

Nothing is deleted. You get a map of every URL to its cluster, and a
fetch list that skips the certain duplicates.

The authoritative dedup still happens AFTER fetching, in 05_dedupe.py, which
compares full body text and applies the editorial-overlay rule properly.

USAGE
    python 05a_dedupe_prefetch.py
    python 05a_dedupe_prefetch.py --collapse-syndication   # fetch fewer

OUTPUT
    prefetch_dedupe_map.csv    every url -> cluster_id, stage, is_representative
    fetch_list.csv             the URLs actually worth fetching
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
import re
import unicodedata
from collections import defaultdict

import pandas as pd

IN_FILE = INTERIM / "articles_v3_enriched.csv"
OUT_MAP = INTERIM / "prefetch_dedupe_map.csv"
OUT_FETCH = INTERIM / "fetch_list.csv"


def norm_url(x):
    """Strip the things that differ without changing the page."""
    x = str(x)
    x = re.sub(r"^https?://", "", x)
    x = re.sub(r"^www\.", "", x)
    x = x.split("?")[0].split("#")[0].rstrip("/")
    return x.lower()


def norm_title(t):
    if not isinstance(t, str):
        return ""
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collapse-syndication", action="store_true",
                    help="also collapse same-title-different-outlet groups "
                         "(fewer fetches, small risk of losing an editorial variant)")
    a = ap.parse_args()

    df = pd.read_csv(IN_FILE, low_memory=False)
    n0 = len(df)
    print(f"{n0:,} articles in")

    df["_nu"] = df["url"].apply(norm_url)
    df["_nt"] = df["title"].apply(norm_title)
    df["_has_title"] = df["_nt"].str.len() > 15

    # union-find over article indices
    parent = list(range(len(df)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        parent[find(x)] = find(y)

    stage = ["" for _ in range(len(df))]

    # ── Stage A: URL variants ──
    by_nu = defaultdict(list)
    for i, k in enumerate(df["_nu"]):
        by_nu[k].append(i)
    a_count = 0
    for grp in by_nu.values():
        if len(grp) > 1:
            for j in grp[1:]:
                union(grp[0], j)
                stage[j] = "url_variant"
                a_count += 1
    print(f"  Stage A  url variants collapsed        : {a_count:,}")

    # ── Stage B: same title, same outlet ──
    b_count = 0
    sub = df[df["_has_title"]]
    by_to = defaultdict(list)
    for i, (t, o) in zip(sub.index, zip(sub["_nt"], sub["outlet"])):
        by_to[(t, str(o).lower())].append(i)
    for grp in by_to.values():
        if len(grp) > 1:
            for j in grp[1:]:
                if find(j) != find(grp[0]):
                    union(grp[0], j)
                    stage[j] = stage[j] or "same_title_same_outlet"
                    b_count += 1
    print(f"  Stage B  same title, same outlet       : {b_count:,}")

    # ── Stage C: same title, different outlet (syndication) ──
    by_t = defaultdict(list)
    for i, t in zip(sub.index, sub["_nt"]):
        by_t[t].append(i)
    synd_groups, c_count = 0, 0
    synd_flag = ["" for _ in range(len(df))]
    for t, grp in by_t.items():
        outlets = df.loc[grp, "outlet"].nunique()
        if len(grp) > 1 and outlets > 1:
            synd_groups += 1
            for j in grp:
                synd_flag[j] = "syndicated"
            if a.collapse_syndication:
                for j in grp[1:]:
                    if find(j) != find(grp[0]):
                        union(grp[0], j)
                        stage[j] = stage[j] or "syndication"
                        c_count += 1
    print(f"  Stage C  syndication groups found      : {synd_groups:,}"
          f"{'  (collapsed ' + format(c_count, ',') + ')' if a.collapse_syndication else '  (flagged, NOT collapsed)'}")

    df["_cluster"] = [find(i) for i in range(len(df))]
    df["_stage"] = stage
    df["_syndicated"] = synd_flag

    # representative = longest title, then first url (stable)
    reps = set()
    for cid, grp in df.groupby("_cluster").groups.items():
        idx = list(grp)
        best = max(idx, key=lambda i: (len(str(df.at[i, "_nt"])), -i))
        reps.add(best)
    df["_is_rep"] = [i in reps for i in df.index]

    keep = [c for c in ["url", "outlet", "source_country", "title"] if c in df.columns]
    out = df[keep].copy()
    out["cluster_id"] = df["_cluster"]
    out["collapsed_by"] = df["_stage"]
    out["syndicated"] = df["_syndicated"]
    out["is_representative"] = df["_is_rep"]
    out.to_csv(OUT_MAP, index=False)

    fetch = df[df["_is_rep"]].drop(columns=[c for c in df.columns if c.startswith("_")])
    fetch.to_csv(OUT_FETCH, index=False)

    n1 = len(fetch)
    print(f"\n{n0:,} articles -> {n1:,} to fetch  "
          f"({n0-n1:,} certain duplicates skipped, {(n0-n1)/n0*100:.1f}%)")
    print(f"\nwrote {OUT_MAP}  (every url, its cluster, why it collapsed)")
    print(f"wrote {OUT_FETCH}  (the list to feed the fetcher)")
    if not a.collapse_syndication:
        n_syn = (df["_syndicated"] == "syndicated").sum()
        print(f"\n{n_syn:,} articles are flagged as syndicated but kept, so the")
        print("post-fetch dedup can check them for editorial overlays. Re-run with")
        print("--collapse-syndication if you would rather fetch fewer.")
    print("\nnext: pilot_fetch_rate.py, then 01_fetch_and_label.py on fetch_list.csv")


if __name__ == "__main__":
    main()
