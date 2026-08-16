#!/usr/bin/env python3
"""
21_name_topics.py
-----------------
LDA finds word groups. It cannot name them. This script handles the naming.

STEP 1
    python 21_name_topics.py --make-template

    Writes topic_names.csv. One row per topic, per language, with:
      language, topic, top_words, suggested_name, your_name

    suggested_name is my guess. your_name is empty and yours to fill.

STEP 2
    Open topic_names.csv in Excel. Read the top_words column. Type a name in
    your_name. If my suggestion is fine, copy it across. If a topic is junk,
    write DROP.

STEP 3
    python 21_name_topics.py --apply

    Re-renders every chart using your names instead of "Topic 1, Topic 2".
    Topics marked DROP are excluded from the country and time charts.

NEEDS
    lda_topics_<lang>.csv and lda_article_topics.csv from 17_lda_topics.py
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
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TEMPLATE = CODEBOOK / "topic_names.csv"

# My suggestions, keyed by a distinctive word in the topic. Matched loosely, so
# if your run differs slightly these still mostly land.
SUGGEST = [
    # (language, must contain these words, suggested name)
    ("en", ["kavanaugh", "investigation"], "Legal and political investigations"),
    ("en", ["music", "kelly"], "Music industry allegations"),
    ("en", ["workplace", "employees"], "Workplace harassment and policy"),
    ("en", ["film", "actor", "india"], "Film industry allegations"),
    ("en", ["nirbhaya", "delhi"], "Indian criminal cases"),
    ("en", ["trump", "weinstein"], "US politics and Weinstein"),
    ("en", ["like", "think", "know"], "DROP  (function words, no theme)"),
    ("en", ["university", "students"], "Campus and prevention"),

    ("es", ["feminista", "redes"], "Feminist politics and social media"),
    ("es", ["metoo", "movimiento"], "The MeToo movement"),
    ("es", ["proteccion", "protección", "ley"], "Gender violence law and policy"),
    ("es", ["vox", "madrid"], "Spanish party politics"),
    ("es", ["puebla", "fiscalia"], "Mexican femicide investigations"),
    ("es", ["familia", "madre", "hija"], "Personal and family narrative"),
    ("es", ["pareja", "cuerpo", "policia"], "Individual femicide cases"),
    ("es", ["juicio", "juez", "penal"], "Court proceedings"),

    ("fr", ["comme", "aussi", "peut"], "General discussion of harassment"),
    ("fr", ["weinstein"], "Weinstein"),
    ("fr", ["cour", "proces", "prison"], "Criminal trials"),
    ("fr", ["parquet", "judiciaire", "garde"], "Police investigations"),
    ("fr", ["balancetonporc", "muller"], "#BalanceTonPorc and the Muller case"),
    ("fr", ["haenel", "polanski"], "Haenel, Polanski and the film industry"),
    ("fr", ["ramadan", "tariq"], "Tariq Ramadan case"),
    ("fr", ["darmanin"], "Darmanin case"),

    ("pt", ["juri", "julgamento", "reu"], "Court trials"),
    ("pt", ["campanha", "sociedade"], "Public campaigns and debate"),
    ("pt", ["tatiane", "advogada"], "A single named case"),
    ("pt", ["metoo", "assedio"], "The MeToo movement"),
    ("pt", ["penha", "lei", "projeto"], "Maria da Penha law and legislation"),
    ("pt", ["dados", "numero", "seguranca"], "Violence statistics"),
    ("pt", ["letra", "selecao", "volei"], "DROP  (sport and entertainment, unrelated)"),
    ("pt", ["suspeito", "preso"], "Police cases"),

    ("de", ["belastigung", "debatte", "gewalt"], "General debate on harassment"),
    ("de", ["wedel"], "Dieter Wedel case"),
    ("de", ["jahres", "magazin", "swift"], "Time Person of the Year"),
    ("de", ["weinstein"], "Weinstein"),
    ("de", ["mueller", "russland", "hexenjagd"], "US politics, witch-hunt framing"),
    ("de", ["www", "http", "presseportal"], "DROP  (press release boilerplate)"),
    ("de", ["deneuve", "berlinale"], "Deneuve letter and the Berlinale"),
    ("de", ["ukraine", "biden", "newsblog"], "DROP  (unrelated news blog)"),

    ("sv", ["svt", "fakta", "relevant"], "DROP  (broadcaster boilerplate)"),
    ("sv", ["virtanen", "wallin", "fortal"], "Virtanen and Wallin defamation case"),
    ("sv", ["akademien", "kulturprofilen"], "Swedish Academy and Kulturprofilen"),
    ("sv", ["upprop", "uppropet", "kvinnliga"], "The upprop campaigns"),
    ("sv", ["vald", "procent", "kvinnors"], "Violence statistics"),
    ("sv", ["man", "nagon", "varit"], "DROP  (function words, no theme)"),

    ("it", ["weinstein"], "Weinstein"),
]


def _strip(t):
    import unicodedata
    t = unicodedata.normalize("NFKD", str(t).lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def suggest_for(lang, words):
    w = _strip(words)
    best, score = "", 0
    for l, keys, name in SUGGEST:
        if l != lang:
            continue
        hits = sum(1 for k in keys if _strip(k) in w)
        if hits > score:
            best, score = name, hits
    return best if score >= 1 else ""


def make_template():
    rows = []
    for f in sorted(glob.glob(str(TABLES / "lda_topics_*.csv"))):
        lang = re.search(r"lda_topics_(.+)\.csv", f).group(1)
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            n = int(re.search(r"(\d+)", str(r["topic"])).group(1))
            rows.append({
                "language": lang,
                "topic": n,
                "top_words": r["top_words"],
                "suggested_name": suggest_for(lang, r["top_words"]),
                "your_name": "",
            })
    if not rows:
        raise SystemExit("no lda_topics_*.csv found. Run 17_lda_topics.py first.")
    t = pd.DataFrame(rows)
    t.to_csv(TEMPLATE, index=False)
    print(f"wrote {TEMPLATE}  ({len(t)} topics across "
          f"{t['language'].nunique()} languages)\n")
    print("NEXT:")
    print(f"  1. open {TEMPLATE} in Excel")
    print("  2. read top_words, type a name in your_name")
    print("  3. copy suggested_name across if you agree with it")
    print("  4. write DROP for topics that are junk")
    print("  5. run: python 21_name_topics.py --apply")
    print("\nmy suggestions, for reference:")
    for lang, g in t.groupby("language"):
        print(f"\n  {lang.upper()}")
        for _, r in g.iterrows():
            s = r["suggested_name"] or "(no suggestion, you decide)"
            print(f"    {r['topic']}. {s}")
            print(f"       {str(r['top_words'])[:78]}")


def apply_names():
    if not os.path.exists(TEMPLATE):
        raise SystemExit(f"{TEMPLATE} not found. Run --make-template first.")
    names = pd.read_csv(TEMPLATE)
    names["final"] = names["your_name"].fillna("").astype(str).str.strip()
    blank = names["final"] == ""
    names.loc[blank, "final"] = names.loc[blank, "suggested_name"].fillna("").astype(str)
    still = names["final"] == ""
    if still.any():
        print(f"note: {int(still.sum())} topics have no name; "
              f"they keep 'Topic N'")
        names.loc[still, "final"] = "Topic " + names.loc[still, "topic"].astype(str)

    art = pd.read_csv(PROCESSED / "lda_article_topics.csv", low_memory=False)
    key = {(r["language"], r["topic"]): r["final"] for _, r in names.iterrows()}
    art["topic_name"] = [key.get((l, t), f"Topic {t}")
                         for l, t in zip(art["lang"], art["main_topic"])]
    art["dropped"] = art["topic_name"].str.startswith("DROP")
    art.to_csv(PROCESSED / "lda_article_topics_named.csv", index=False)
    print(f"wrote lda_article_topics_named.csv "
          f"({int((~art['dropped']).sum()):,} articles in kept topics, "
          f"{int(art['dropped'].sum()):,} in dropped ones)")

    art["published"] = pd.to_datetime(art["published"], errors="coerce", utc=True)
    keep = art[~art["dropped"]]

    for lang, g in keep.groupby("lang"):
        labels = sorted(g["topic_name"].unique())
        if len(labels) < 2:
            continue

        # topic mix by country
        top = g["source_country"].value_counts()
        top = top[top >= 30].head(12).index
        sub = g[g["source_country"].isin(top)]
        if len(top) >= 2:
            m = pd.crosstab(sub["source_country"], sub["topic_name"],
                            normalize="index") * 100
            m = m.reindex(top)
            fig, ax = plt.subplots(figsize=(max(8, .9 * len(labels) + 4),
                                            max(3.2, .42 * len(m))))
            im = ax.imshow(m.values, aspect="auto", cmap="RdPu")
            ax.set_yticks(range(len(m)))
            ax.set_yticklabels(m.index, fontsize=9)
            ax.set_xticks(range(len(m.columns)))
            ax.set_xticklabels([c[:26] for c in m.columns], fontsize=8,
                               rotation=35, ha="right")
            ax.set_title(f"Topic mix by country, {lang.upper()}",
                         fontsize=12, fontweight="bold", loc="left")
            fig.colorbar(im, ax=ax, shrink=.8, label="% of that country's articles")
            fig.tight_layout()
            fig.savefig(FIGURES / f"named_by_country_{lang}.png", dpi=135)
            plt.close(fig)

        # topics over time
        g2 = g.copy()
        g2["month"] = g2["published"].dt.strftime("%Y-%m")
        mt = pd.crosstab(g2["month"], g2["topic_name"], normalize="index") * 100
        mt = mt.sort_index()
        if len(mt) >= 3:
            fig, ax = plt.subplots(figsize=(11.5, 4.4))
            for c in mt.columns:
                ax.plot(mt.index, mt[c], linewidth=1.8, label=c[:34])
            step = max(1, len(mt) // 12)
            ax.set_xticks(range(0, len(mt), step))
            ax.set_xticklabels([mt.index[i] for i in range(0, len(mt), step)],
                               rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("% of that month's articles")
            ax.set_title(f"Topics over time, {lang.upper()}",
                         fontsize=12, fontweight="bold", loc="left")
            ax.legend(fontsize=7.5, ncol=2)
            ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(FIGURES / f"named_over_time_{lang}.png", dpi=135)
            plt.close(fig)
        print(f"  {lang}: wrote named_by_country_{lang}.png, "
              f"named_over_time_{lang}.png")

    # a plain summary table you can paste anywhere
    out = names[["language", "topic", "final", "top_words"]].rename(
        columns={"final": "name"})
    out.to_csv(TABLES / "topic_names_final.csv", index=False)
    print("\nwrote topic_names_final.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-template", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.make_template:
        make_template()
    elif a.apply:
        apply_names()
    else:
        print("use --make-template first, then --apply after you edit the file")


if __name__ == "__main__":
    main()
