#!/usr/bin/env python3
"""
08_dedupe.py
------------
Finds articles that are the same story, or a light rewrite of it, and groups
them — WITHOUT collapsing cases where one version carries an editorial or
opinion overlay. Per your rule: same wire copy = duplicate; same wire copy with
a columnist's framing bolted on = a genuinely different article, because the
added opinion changes what the piece means.

This runs AFTER fetching (needs article body text), on labeled_sample.csv or
any file with columns: url, source_country, title, article_text, and — if you
have run the LLM labeler — article_type, stance, evaluative_language.

HOW IT DECIDES

  Stage 1  Exact duplicates: identical normalized body text -> same story.
  Stage 2  Near-duplicates: MinHash/LSH over shingles of the body finds pairs
           above a Jaccard threshold (default 0.80). This is how you catch wire
           copy republished with a changed headline or a trimmed intro.
  Stage 3  Editorial guard: within a near-duplicate cluster, split back apart
           any member that diverges editorially from the rest. A member is kept
           SEPARATE if, relative to its cluster:
             - its article_type is opinion/editorial/analysis while others are
               straight_news (an opinion overlay), OR
             - its LLM stance differs from the cluster's majority, OR
             - it carries evaluative_language the shared base text does not
               (the columnist's own loaded words), OR
             - it has a substantial unique text segment the others lack (the
               added commentary), measured as text not covered by the shared core.
           These are exactly the "same article, different light" cases.

  Output   dedupe_map.csv    url -> cluster_id, is_representative, split_reason
           deduped_articles.csv   one representative row per final cluster,
                                   with dup_count and dup_urls for the audit trail

Nothing is deleted. Deduplication is a LABELING, not a deletion — you keep every
row and every reason, so the choice is reversible and defensible.

USAGE
    pip install datasketch pandas regex tqdm
    python 05_dedupe.py --threshold 0.80
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
import json
import re
import unicodedata

import pandas as pd
from tqdm import tqdm

try:
    from datasketch import MinHash, MinHashLSH
    HAVE_DS = True
except ImportError:
    HAVE_DS = False

IN_FILE = PROCESSED / "labeled_articles.csv"   # from 07_fetch_and_label.py
OUT_MAP = INTERIM / "dedupe_map.csv"
OUT_ARTICLES = PROCESSED / "deduped_articles.csv"
SHINGLE = 5          # words per shingle
NUM_PERM = 128       # MinHash permutations; 128 is the usual accuracy/speed point


def normalize(text):
    """Lowercase, strip accents, collapse whitespace/punctuation. For matching
    the SHARED story, not for display — accents and case do not change identity."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def shingles(norm_text, k=SHINGLE):
    words = norm_text.split()
    if len(words) < k:
        return set(words) or {norm_text}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def make_minhash(sh):
    m = MinHash(num_perm=NUM_PERM)
    for s in sh:
        m.update(s.encode("utf-8"))
    return m


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── editorial-divergence test ────────────────────────────────────────────
def parse_eval(v):
    try:
        x = json.loads(v) if isinstance(v, str) else v
        return set(w.lower().strip() for w in x) if isinstance(x, list) else set()
    except Exception:
        return set()


OPINION_TYPES = {"opinion_editorial", "feature_analysis"}


def split_reasons(member, cluster_members, shared_shingles):
    """Return list of reasons this member should stay separate from its cluster.
    Empty list = it is a true duplicate and folds in."""
    reasons = []

    others = [m for m in cluster_members if m["url"] != member["url"]]
    if not others:
        return reasons

    # 1. opinion overlay: this one is opinion while the cluster core is news
    m_type = str(member.get("article_type", "")).lower()
    other_types = [str(o.get("article_type", "")).lower() for o in others]
    if m_type in OPINION_TYPES and any(t not in OPINION_TYPES and t for t in other_types):
        reasons.append("opinion_overlay")

    # 2. stance divergence from the cluster majority
    m_stance = str(member.get("stance", "")).lower()
    other_stances = [str(o.get("stance", "")).lower() for o in others if o.get("stance")]
    if m_stance and other_stances:
        maj = max(set(other_stances), key=other_stances.count)
        if m_stance != maj and m_stance not in ("", "not_applicable"):
            reasons.append(f"stance_differs({m_stance}_vs_{maj})")

    # 3. unique evaluative language the shared base does not carry
    m_eval = parse_eval(member.get("evaluative_language", ""))
    shared_eval = set.intersection(
        *[parse_eval(o.get("evaluative_language", "")) for o in others]) if others else set()
    unique_eval = m_eval - shared_eval
    if len(unique_eval) >= 2:
        reasons.append(f"added_evaluative_language({len(unique_eval)})")

    # 4. substantial unique body segment not covered by the shared core
    m_sh = member["_shingles"]
    unique_frac = 1.0 - (len(m_sh & shared_shingles) / len(m_sh)) if m_sh else 0.0
    if unique_frac >= 0.35:     # a third or more of this text is not in the shared core
        reasons.append(f"unique_text({unique_frac:.0%})")

    return reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="Jaccard similarity to call two bodies near-duplicate")
    ap.add_argument("--infile", default=IN_FILE)
    a = ap.parse_args()

    df = pd.read_csv(a.infile)
    # only rows with real body text can be compared
    text_col = "article_text" if "article_text" in df.columns else "text_excerpt"
    df = df[df[text_col].notna() & (df[text_col].astype(str).str.len() > 200)].copy()
    df = df.reset_index(drop=True)
    print(f"{len(df)} articles with usable body text")

    df["_norm"] = df[text_col].apply(normalize)
    df["_shingles"] = df["_norm"].apply(shingles)

    # ── Stage 1: exact dupes via normalized-text hash ──
    df["_exact"] = df["_norm"].apply(lambda t: hash(t))

    # ── Stage 2: near-dup pairs ──
    if not HAVE_DS:
        print("datasketch not installed; falling back to O(n^2) Jaccard.")
        print("  pip install datasketch   is strongly recommended for >2000 rows.")
        pairs = []
        rows = df.to_dict("records")
        for i in tqdm(range(len(rows)), desc="pairwise"):
            for j in range(i + 1, len(rows)):
                if jaccard(rows[i]["_shingles"], rows[j]["_shingles"]) >= a.threshold:
                    pairs.append((i, j))
    else:
        lsh = MinHashLSH(threshold=a.threshold, num_perm=NUM_PERM)
        mh = {}
        for i, sh in tqdm(enumerate(df["_shingles"]), total=len(df), desc="minhash"):
            m = make_minhash(sh)
            mh[i] = m
            lsh.insert(str(i), m)
        pairs = []
        for i in tqdm(range(len(df)), desc="query"):
            for j in lsh.query(mh[i]):
                j = int(j)
                if j > i:
                    pairs.append((i, j))

    # ── union-find into clusters ──
    parent = list(range(len(df)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        parent[find(x)] = find(y)

    # exact-text matches always union
    from collections import defaultdict
    by_exact = defaultdict(list)
    for i, h in enumerate(df["_exact"]):
        by_exact[h].append(i)
    for grp in by_exact.values():
        for j in grp[1:]:
            union(grp[0], j)
    for i, j in pairs:
        union(i, j)

    df["_cluster"] = [find(i) for i in range(len(df))]

    # ── Stage 3: editorial guard — split divergent members back out ──
    recs = df.to_dict("records")
    by_cluster = defaultdict(list)
    for r in recs:
        by_cluster[r["_cluster"]].append(r)

    rows_out = []
    next_split_id = df["_cluster"].max() + 1
    for cid, members in by_cluster.items():
        if len(members) == 1:
            m = members[0]
            rows_out.append((m["url"], cid, True, ""))
            continue
        # shared core = shingles common to ALL members (the wire base)
        shared = set.intersection(*[m["_shingles"] for m in members])

        # test every member for editorial divergence FIRST, before choosing a
        # representative — otherwise the opinion overlay, which is often the
        # longest text, gets picked as rep and its divergence is never checked.
        divergent, plain = [], []
        for m in members:
            reasons = split_reasons(m, members, shared)
            (divergent if reasons else plain).append((m, reasons))

        # representative of the TRUE-duplicate core = longest plain member;
        # if every member diverged, fall back to the longest overall.
        pool = plain if plain else [(m, []) for m in members]
        rep_url = max(pool, key=lambda mr: len(str(mr[0][text_col])))[0]["url"]

        for m, reasons in plain:
            rows_out.append((m["url"], cid, m["url"] == rep_url, ""))
        for m, reasons in divergent:
            # editorially distinct -> its own cluster, representative of itself
            rows_out.append((m["url"], next_split_id, True, ";".join(reasons)))
            next_split_id += 1

    dmap = pd.DataFrame(rows_out, columns=[
        "url", "cluster_id", "is_representative", "split_reason"])
    dmap.to_csv(OUT_MAP, index=False)

    # ── representative-only table with audit trail ──
    merged = df.merge(dmap, on="url")
    reps = merged[merged["is_representative"]].copy()
    dup_urls = (merged.groupby("cluster_id")["url"]
                .apply(lambda s: " ; ".join(s)).to_dict())
    dup_count = merged.groupby("cluster_id")["url"].count().to_dict()
    reps["dup_count"] = reps["cluster_id"].map(dup_count)
    reps["dup_urls"] = reps["cluster_id"].map(dup_urls)
    drop_internal = [c for c in reps.columns if c.startswith("_")]
    reps.drop(columns=drop_internal).to_csv(OUT_ARTICLES, index=False)

    # ── report ──
    n_clusters = dmap["cluster_id"].nunique()
    n_splits = (dmap["split_reason"] != "").sum()
    print(f"\n{len(df)} articles -> {n_clusters} distinct after dedup")
    print(f"  collapsed: {len(df) - n_clusters} redundant copies removed")
    print(f"  editorial-divergence splits preserved: {n_splits}")
    if n_splits:
        print("\nwhy things were kept separate:")
        allr = Counter()
        for s in dmap.loc[dmap["split_reason"] != "", "split_reason"]:
            for r in s.split(";"):
                allr[re.sub(r"\(.*\)", "", r)] += 1
        for r, n in allr.most_common():
            print(f"  {r:28s} {n}")
    big = merged.groupby("cluster_id").size().sort_values(ascending=False)
    print("\nlargest duplicate clusters (story republished most):")
    for cid, n in big.head(5).items():
        if n > 1:
            ex = merged[merged["cluster_id"] == cid].iloc[0]
            print(f"  {n:3d}x  {str(ex.get('title',''))[:64]}")
    print(f"\nwrote {OUT_MAP} and {OUT_ARTICLES}")
    print("Nothing deleted. dedupe_map.csv keeps every url and every reason.")


if __name__ == "__main__":
    from collections import Counter
    main()
