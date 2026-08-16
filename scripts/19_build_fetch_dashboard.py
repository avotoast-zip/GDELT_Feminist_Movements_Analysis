#!/usr/bin/env python3
"""
19_build_fetch_dashboard.py
---------------------------
Builds a second dashboard page covering the fetching step and the duplicate
removal step. The first dashboard covered the collected corpus. This one covers
what happened after.

It contains:
  - how many articles were downloaded and why the rest failed
  - download success by country and by year
  - how the download changed the shape of the corpus
  - duplicate counts at three thresholds: 100%, 90% and 80%
  - which domains produced the most duplicate copies
  - a flag for pages that were not really articles

RUN
    python 19_build_fetch_dashboard.py

NEEDS
    labeled_articles.csv    the download results
    dedupe_map.csv          from 08_dedupe.py

OUTPUT
    metoo_dashboard_fetch.html      open in any browser, works offline

The threshold table is the slow part. It re-groups every duplicate cluster at
each cutoff. Expect 10 to 25 minutes. Use --skip-thresholds to build the page
without it, or --thresholds-from file.json to reuse a previous run.
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
import os
import re
import unicodedata
from collections import defaultdict

import pandas as pd

ARTICLES = PROCESSED / "labeled_articles.csv"
MAP = INTERIM / "dedupe_map.csv"
OUT = DASHBOARDS / "metoo_dashboard_fetch.html"
STALE = re.compile(r"\b(202[3-6])\b")

BLOCKED = ["FAILED:403", "FAILED:429", "FAILED:406", "FAILED:401", "FAILED:402"]


def norm(x):
    x = unicodedata.normalize("NFKD", str(x))
    x = "".join(c for c in x if not unicodedata.combining(c)).lower()
    x = re.sub(r"https?://\S+", " ", x)
    x = re.sub(r"[^\w\s]", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def shingles(x, k=5):
    w = x.split()
    return set(w) if len(w) < k else {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)}


def threshold_table(m, tmap):
    """Re-group each duplicate cluster at 100%, 90% and 80% overlap.
    The 80% clustering is a superset of the stricter ones, so we only ever
    compare inside an existing cluster. That keeps this feasible."""
    sizes = m.groupby("cluster_id").size()
    multi = sizes[sizes > 1]
    norms, shs = {}, {}
    for u, txt in tmap.items():
        n = norm(txt)[:20000]
        norms[u] = n
        shs[u] = shingles(n)

    by_cluster = defaultdict(list)
    for cid, u in zip(m["cluster_id"], m["url"]):
        by_cluster[cid].append(u)

    def partition(urls, th):
        par = {u: u for u in urls}
        def f(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for i in range(len(urls)):
            for j in range(i + 1, len(urls)):
                a, b = shs.get(urls[i], set()), shs.get(urls[j], set())
                if not a or not b:
                    continue
                same = (norms.get(urls[i]) == norms.get(urls[j])) if th >= 1.0 \
                    else (len(a & b) / len(a | b) >= th)
                if same:
                    par[f(urls[i])] = f(urls[j])
        groups = defaultdict(list)
        for u in urls:
            groups[f(u)].append(u)
        return list(groups.values())

    res = {}
    for th, label in [(1.0, "100"), (0.90, "90"), (0.80, "80")]:
        kept = removed = 0
        for cid in multi.index:
            urls = [u for u in by_cluster[cid] if u in shs]
            if len(urls) < 2:
                kept += len(urls)
                continue
            sample = urls[:120]          # cap the giant clusters
            parts = partition(sample, th)
            scale = len(urls) / len(sample)
            k = len(parts) * scale
            kept += k
            removed += len(urls) - k
        res[label] = {"removed": int(round(removed)),
                      "unique": int(round(kept)) + int((sizes == 1).sum())}
        print(f"  {label}% overlap: {int(round(removed)):,} removed")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-thresholds", action="store_true")
    ap.add_argument("--thresholds-from", default=None)
    a = ap.parse_args()

    print("reading download results...")
    cols = ["url", "source_country", "outlet", "published", "fetch_status",
            "text_chars", "has_hashtag_term", "article_text"]
    parts = []
    for ch in pd.read_csv(ARTICLES, usecols=cols, chunksize=100_000, low_memory=False):
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    df["ok"] = df["fetch_status"].isin(["OK", "OK_WAYBACK"])
    print(f"  {len(df):,} attempted, {df['ok'].sum():,} downloaded")

    st = df["fetch_status"].astype(str)
    outcome = {
        "Downloaded": int(df["ok"].sum()),
        "Page is gone (404, 410)": int(st.isin(["FAILED:404", "FAILED:410"]).sum()),
        "Website blocked us (403, 429)": int(st.isin(BLOCKED).sum()),
        "Network or server error": int(st.str.contains(
            "Timeout|ConnectionError|SSLError|FAILED:5|Redirect|Chunked", na=False).sum()),
        "No usable text": int(st.isin(["TOO_SHORT", "NO_TEXT"]).sum()),
    }

    ok = df[df["ok"]].copy()
    stale = ok["article_text"].astype(str).str.slice(0, 6000).str.contains(STALE, na=False)
    n_stale = int(stale.sum())

    known = df[df["source_country"] != "UNKNOWN"]
    by_country = (known.groupby("source_country")
                  .agg(attempted=("url", "size"), got=("ok", "sum")))
    by_country["rate"] = (by_country["got"] / by_country["attempted"] * 100).round(1)
    by_country = by_country[by_country["attempted"] >= 300].sort_values("rate")

    df["year"] = pd.to_datetime(df["published"], errors="coerce", utc=True).dt.year
    by_year = df.groupby("year").agg(attempted=("url", "size"), got=("ok", "sum"))
    by_year["rate"] = (by_year["got"] / by_year["attempted"] * 100).round(1)

    before = known["source_country"].value_counts(normalize=True) * 100
    after = ok[ok["source_country"] != "UNKNOWN"]["source_country"].value_counts(
        normalize=True) * 100
    shift = pd.DataFrame({"before": before, "after": after}).fillna(0)
    shift["change"] = (shift["after"] - shift["before"]).round(1)
    shift = shift.head(12).round(1)

    hash_before = df["has_hashtag_term"].mean() * 100
    hash_after = ok["has_hashtag_term"].mean() * 100

    # ── duplicates ──
    thresholds = None
    dom = []
    if os.path.exists(MAP):
        m = pd.read_csv(MAP, low_memory=False)
        sizes = m.groupby("cluster_id").size()
        n_unique = int(sizes.nunique() if False else len(sizes))
        n_removed = len(m) - len(sizes)

        mm = m.merge(ok[["url", "outlet", "source_country"]], on="url", how="left")
        mm["csize"] = mm["cluster_id"].map(sizes)
        dupd = mm[mm["csize"] > 1]
        g = dupd.groupby("outlet").agg(copies=("url", "size"),
                                       stories=("cluster_id", "nunique"),
                                       country=("source_country", "first")).reset_index()
        g["redundant"] = g["copies"] - g["stories"]
        dom = g.sort_values("redundant", ascending=False).head(18).to_dict("records")

        if a.thresholds_from and os.path.exists(a.thresholds_from):
            thresholds = json.load(open(a.thresholds_from))
            print("  thresholds loaded from file")
        elif not a.skip_thresholds:
            print("computing duplicate counts at 100%, 90%, 80% (slow)...")
            need = set(m[m["cluster_id"].isin(sizes[sizes > 1].index)]["url"])
            tmap = dict(zip(ok[ok["url"].isin(need)]["url"],
                            ok[ok["url"].isin(need)]["article_text"]))
            thresholds = threshold_table(m, tmap)
            json.dump(thresholds, open(TABLES / "dedupe_thresholds.json", "w"), indent=1)
    else:
        m = None
        n_unique = n_removed = 0

    D = {
        "attempted": int(len(df)), "downloaded": int(df["ok"].sum()),
        "rate": round(df["ok"].mean() * 100, 1),
        "unique_after_dedupe": n_unique, "dupes_removed": n_removed,
        "stale": n_stale, "stale_pct": round(n_stale / max(len(ok), 1) * 100, 1),
        "hash_before": round(hash_before, 1), "hash_after": round(hash_after, 1),
        "outcome": outcome,
        "by_country": [{"c": i, "a": int(r.attempted), "g": int(r.got), "r": float(r.rate)}
                       for i, r in by_country.iterrows()],
        "by_year": [{"y": int(i), "a": int(r.attempted), "r": float(r.rate)}
                    for i, r in by_year.iterrows() if pd.notna(i)],
        "shift": [{"c": i, "b": float(r.before), "a": float(r.after), "d": float(r.change)}
                  for i, r in shift.iterrows()],
        "thresholds": thresholds,
        "domains": dom,
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(D, ensure_ascii=False,
                                                   separators=(",", ":")))
    open(OUT, "w").write(html)
    print(f"\nwrote {OUT}  ({len(html)//1024} KB)")
    print("Open it in a browser. It works offline.")


TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>#MeToo corpus - fetching and duplicates</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#F0F0EC;color:#17171E;font-family:'Segoe UI',system-ui,sans-serif;font-size:13px;line-height:1.45}
.wrap{max-width:1060px;margin:0 auto;padding:16px 16px 60px}
header{border-bottom:3px solid #17171E;padding-bottom:10px}
.eyebrow{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#A2196B;margin-bottom:6px;font-family:monospace}
h1{font-size:22px;font-weight:900;letter-spacing:-.02em;line-height:1.1}
h1 span{color:#5A5A66;font-weight:400}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:#DCDCD4;border:1px solid #DCDCD4;margin:14px 0}
.kpi{background:#FCFCFA;padding:11px 9px}
.kpi b{display:block;font-size:20px;font-weight:900;letter-spacing:-.02em}
.kpi small{font-size:10px;color:#5A5A66;line-height:1.3;display:block}
.kpi.g b{color:#3E6B73}.kpi.r b{color:#B5432A}.kpi.a b{color:#A2196B}
section{margin-top:22px}
h2{font-size:13px;font-weight:700;margin-bottom:2px}
.sub{font-size:11px;color:#5A5A66;margin-bottom:8px;max-width:78ch}
.card{background:#FCFCFA;border:1px solid #DCDCD4;padding:12px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:#17171E;color:#F0F0EC;padding:6px 8px;text-align:left;font-weight:600;font-size:10.5px;position:sticky;top:0}
td{padding:4px 8px;border-bottom:1px solid #E6E6E0}
td.r{text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:#EDEFEA}
.scroll{max-height:400px;overflow-y:auto;border:1px solid #DCDCD4}
.bar{height:15px;background:#3E6B73;display:inline-block;vertical-align:middle;border-radius:2px}
.bar.r{background:#B5432A}.bar.a{background:#A2196B}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.note{border-left:3px solid #A2196B;padding:10px 14px;background:#FCFCFA;font-size:11.5px;color:#5A5A66;margin-top:12px;line-height:1.55}
.note b{color:#17171E}
.big{font-size:26px;font-weight:900;letter-spacing:-.02em}
footer{margin-top:26px;padding-top:12px;border-top:1px solid #DCDCD4;font-family:monospace;font-size:10.5px;color:#8A8A92}
.pos{color:#3E6B73;font-weight:700}.neg{color:#B5432A;font-weight:700}
</style></head><body><div class="wrap">
<header>
<div class="eyebrow">Stage 2 - after collection - fetching and duplicate removal</div>
<h1>Downloading the articles <span>and removing duplicates</span></h1>
</header>
<div class="kpis" id="kpis"></div>

<section><h2>What happened to each link</h2>
<p class="sub">The links are 6 to 9 years old. Only half the failures are dead pages. Blocked and error results can still be recovered.</p>
<div class="card"><table id="outcome"></table></div></section>

<section><h2>Duplicate articles found at three thresholds</h2>
<p class="sub">Two articles are compared by how much of their wording they share. 100% means the text is identical. 80% allows a changed headline or a trimmed paragraph.</p>
<div class="card"><table id="thresh"></table></div>
<div class="note" id="threshnote"></div></section>

<section><h2>Which sites produced the most duplicate copies</h2>
<p class="sub">Copies is how many articles that site contributed to duplicate groups. Stories is how many distinct stories those copies represent. Redundant is the difference.</p>
<div class="card"><table id="dom"></table></div></section>

<div class="grid2">
<section><h2>Download success by country</h2>
<p class="sub">Countries with 300 or more links tried. Sorted worst first.</p>
<div class="scroll"><table id="ctry"></table></div></section>
<section><h2>How the corpus changed</h2>
<p class="sub">Share of all articles before and after downloading.</p>
<div class="card"><table id="shift"></table></div>
<div style="height:10px"></div>
<h2>Download success by year published</h2>
<div class="card"><table id="year"></table></div></section>
</div>

<div class="note" id="stalenote"></div>
<footer id="foot"></footer></div>
<script>
const D=__DATA__;
document.getElementById('kpis').innerHTML=`
<div class="kpi"><b>${D.attempted.toLocaleString()}</b><small>links tried</small></div>
<div class="kpi g"><b>${D.downloaded.toLocaleString()}</b><small>downloaded</small></div>
<div class="kpi"><b>${D.rate}%</b><small>success rate</small></div>
<div class="kpi r"><b>${D.dupes_removed.toLocaleString()}</b><small>duplicate copies</small></div>
<div class="kpi a"><b>${D.unique_after_dedupe.toLocaleString()}</b><small>unique articles</small></div>
<div class="kpi r"><b>${D.stale_pct}%</b><small>wrong page served</small></div>`;

let o='<tr><th>Result</th><th class=r>Articles</th><th class=r>Share</th><th style="width:38%"></th></tr>';
const tot=D.attempted, mx=Math.max(...Object.values(D.outcome));
for(const [k,v] of Object.entries(D.outcome)){
  const cls=k.startsWith('Downloaded')?'':'r';
  o+=`<tr><td>${k}</td><td class=r>${v.toLocaleString()}</td><td class=r>${(v/tot*100).toFixed(1)}%</td>
  <td><span class="bar ${cls}" style="width:${v/mx*100}%"></span></td></tr>`;}
document.getElementById('outcome').innerHTML=o;

if(D.thresholds){
  let t='<tr><th>Overlap required</th><th class=r>Duplicate copies found</th><th class=r>Unique articles left</th><th class=r>Share removed</th></tr>';
  const order=['100','90','80'];
  const lbl={'100':'100% (identical text)','90':'90% or more','80':'80% or more'};
  for(const k of order){ if(!D.thresholds[k])continue;
    const r=D.thresholds[k];
    t+=`<tr><td><b>${lbl[k]}</b></td><td class=r>${r.removed.toLocaleString()}</td>
    <td class=r>${r.unique.toLocaleString()}</td>
    <td class=r>${(r.removed/D.downloaded*100).toFixed(1)}%</td></tr>`;}
  document.getElementById('thresh').innerHTML=t;
  const a=D.thresholds['100'],c=D.thresholds['80'];
  document.getElementById('threshnote').innerHTML=
   `<b>How to read this.</b> ${a?a.removed.toLocaleString():'-'} articles are word-for-word identical to another article.
    Loosening to 80% finds ${c?(c.removed-a.removed).toLocaleString():'-'} more, which are the same story with a changed
    headline or a trimmed paragraph. The corpus uses the 80% cutoff.`;
} else {
  document.getElementById('thresh').innerHTML='<tr><td>Threshold table not computed. Run without --skip-thresholds.</td></tr>';
}

let dm='<tr><th>#</th><th>Site</th><th>Country</th><th class=r>Copies</th><th class=r>Stories</th><th class=r>Redundant</th><th style="width:26%"></th></tr>';
const dmx=D.domains.length?D.domains[0].redundant:1;
D.domains.forEach((d,i)=>{dm+=`<tr><td>${i+1}</td><td><b>${d.outlet}</b></td><td>${d.country||''}</td>
 <td class=r>${d.copies.toLocaleString()}</td><td class=r>${d.stories.toLocaleString()}</td>
 <td class=r><b>${d.redundant.toLocaleString()}</b></td>
 <td><span class="bar r" style="width:${d.redundant/dmx*100}%"></span></td></tr>`;});
document.getElementById('dom').innerHTML=dm;

let ct='<tr><th>Country</th><th class=r>Tried</th><th class=r>Got</th><th class=r>Rate</th><th style="width:30%"></th></tr>';
D.by_country.forEach(c=>{ct+=`<tr><td>${c.c}</td><td class=r>${c.a.toLocaleString()}</td>
 <td class=r>${c.g.toLocaleString()}</td><td class=r><b>${c.r}%</b></td>
 <td><span class="bar" style="width:${c.r}%"></span></td></tr>`;});
document.getElementById('ctry').innerHTML=ct;

let sh='<tr><th>Country</th><th class=r>Before</th><th class=r>After</th><th class=r>Change</th></tr>';
D.shift.forEach(s=>{const cl=s.d>=0?'pos':'neg';
 sh+=`<tr><td>${s.c}</td><td class=r>${s.b.toFixed(1)}%</td><td class=r>${s.a.toFixed(1)}%</td>
 <td class="r ${cl}">${s.d>=0?'+':''}${s.d.toFixed(1)}pp</td></tr>`;});
document.getElementById('shift').innerHTML=sh;

let yr='<tr><th>Year</th><th class=r>Tried</th><th class=r>Rate</th><th style="width:34%"></th></tr>';
D.by_year.forEach(y=>{yr+=`<tr><td>${y.y}</td><td class=r>${y.a.toLocaleString()}</td>
 <td class=r><b>${y.r}%</b></td><td><span class="bar" style="width:${y.r}%"></span></td></tr>`;});
document.getElementById('year').innerHTML=yr;

document.getElementById('stalenote').innerHTML=
 `<b>Pages that were not really articles.</b> ${D.stale.toLocaleString()} downloaded articles (${D.stale_pct}%)
  mention a year between 2023 and 2026, even though every article was published between 2017 and 2019.
  The original page was deleted and the website served its current content instead. These should be
  filtered out before any text analysis.<br><br>
  <b>Hashtag share.</b> Before downloading, ${D.hash_before}% of articles had matched a hashtag.
  After downloading it is ${D.hash_after}%. Hashtag articles were harder to download than
  general-vocabulary ones, so the surviving set leans further toward general vocabulary.`;

document.getElementById('foot').textContent=
 'FADS  ·  #MeToo corpus  ·  fetching and duplicate removal  ·  generated '+new Date().toISOString().slice(0,10);
</script></body></html>"""


if __name__ == "__main__":
    main()
