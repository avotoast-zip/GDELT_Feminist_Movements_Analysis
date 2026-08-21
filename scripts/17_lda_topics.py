#!/usr/bin/env python3
"""
17_lda_topics.py
----------------
Finds topics in the downloaded article text, makes word clouds, and charts how
topics vary by country and over time.

WHAT LDA DOES

You give it a pile of articles. It finds groups of words that tend to appear
together. Each group is a topic. Each article gets a score for how much of each
topic it contains.

You choose how many topics to look for. It gives back word groups like
{trial, court, testimony, verdict, judge}. You name the topics yourself. LDA
does not name them.

THE LANGUAGE PROBLEM, AND HOW THIS SCRIPT HANDLES IT

LDA counts words. Our corpus is in dozens of languages. Run it on everything at
once and Spanish articles group with Spanish articles, French with French. You
get a map of languages, not a map of themes.

So this script runs LDA SEPARATELY FOR EACH LANGUAGE by default. You get English
topics, Spanish topics, French topics, and so on. Those are comparable to each
other by theme even though the words differ.

Use --pooled if you want the single-model version anyway, to show what it looks
like.

TWO FILTERS APPLIED FIRST

1. Articles whose text mentions a year from 2023 to 2026 are dropped. Every
   article was published between 2017 and 2019. A later year means the website
   served today's content instead of the old article. About 7% of downloads.
   Leaving them in would put current news into the topics.

2. Articles under 800 characters are dropped. Too short to carry a topic.

RUN
    pip install scikit-learn wordcloud matplotlib langdetect pandas
    python 17_lda_topics.py                    # per language, 8 topics each
    python 17_lda_topics.py --topics 12
    python 17_lda_topics.py --languages en es fr
    python 17_lda_topics.py --pooled           # one model over everything

OUTPUT
    lda_topics_<lang>.csv        top words per topic
    lda_wordcloud_<lang>.png     word clouds, one panel per topic
    lda_by_country_<lang>.png    which topics dominate which countries
    lda_over_time_<lang>.png     topics rising and falling by month
    lda_article_topics.csv       every article and its strongest topic
    lda_report.txt               the summary
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
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
try:
    import ftfy
    HAVE_FTFY = True
except ImportError:
    HAVE_FTFY = False
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

IN_FILE = PROCESSED / "labeled_articles.csv"
MIN_CHARS = 800
MIN_ARTICLES_PER_LANG = 300
STALE = re.compile(r"\b(202[3-6])\b")
# Stopword lists, boilerplate filters and language detection are shared with
# 22_phrases.py so the two analyses stay comparable. See _textutils.py.
from _textutils import (STOP, BOILERPLATE, ACCENTED, DEFAULT_STOP,
                        _strip_accents, stop_for, detect_langs, repair_rot47)

def load_and_clean():
    cols = ["url", "source_country", "outlet", "published", "article_text",
            "fetch_status", "text_chars", "keywords_matched", "has_hashtag_term"]
    parts = []
    for ch in pd.read_csv(IN_FILE, usecols=cols, chunksize=100_000, low_memory=False):
        ch = ch[ch["fetch_status"].isin(["OK", "OK_WAYBACK"])]
        ch = ch[ch["article_text"].notna()]
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df)
    df = df[df["text_chars"] >= MIN_CHARS]
    n1 = len(df)
    stale = df["article_text"].astype(str).str.slice(0, 6000).str.contains(STALE, na=False)
    df = df[~stale]
    n2 = len(df)

    # Some pages were saved with broken character encoding. Spanish "prisión"
    # became "prisiÃ³n", Portuguese "violência" became "violãªncia". Left alone,
    # LDA builds entire topics out of the broken fragments. 5% of articles are
    # affected. ftfy repairs them.
    if HAVE_FTFY:
        bad = df["article_text"].astype(str).str.contains("Ã|Â|ð|ï¼", regex=True, na=False)
        if bad.any():
            df.loc[bad, "article_text"] = df.loc[bad, "article_text"].astype(str).apply(ftfy.fix_text)
            print(f"  repaired broken encoding in {int(bad.sum()):,} articles")
    else:
        print("  NOTE: ftfy not installed. Broken-encoding articles will create")
        print("        junk topics. Run: pip install ftfy")

    # Some US local-news platforms serve the article body ROT47-encoded as a
    # scraper deterrent: "kAmp?5 r2C=D@?" is "<p>And Carlson". Those articles
    # pass the length filter and then build topics out of nothing. Decode them.
    fixed = df["article_text"].astype(str).apply(repair_rot47)
    n_rot = int(sum(c for _, c in fixed))
    if n_rot:
        df["article_text"] = [t for t, _ in fixed]
        print(f"  decoded ROT47-obfuscated body text in {n_rot:,} articles")

    # Some mojibake cannot be repaired, usually Chinese or Japanese text that
    # was double-encoded. ftfy leaves it as strings like "ï¼å". If a lot of the
    # text still looks like that after repair, the article is unreadable and
    # would otherwise become its own topic. Drop it.
    def broken_ratio(t):
        t = str(t)[:2000]
        if not t:
            return 1.0
        bad = sum(1 for c in t if c in "ÃÂðï¼å¼è¾ºæºå½ç")
        return bad / len(t)
    br = df["article_text"].apply(broken_ratio)
    unreadable = br > 0.06
    if unreadable.any():
        print(f"  dropped {int(unreadable.sum()):,} articles with unreadable text")
    df = df[~unreadable]

    # Pages that were never articles: registration forms, site homepages,
    # error pages. They survived the date filter because they carry no date.
    JUNK = ["united states of america us virgin islands",
            "this site could be risky", "enable javascript",
            "your browser is out of date", "select your country",
            "subscribe to continue reading"]
    jl = df["article_text"].astype(str).str.slice(0, 1500).str.lower()
    junk = jl.apply(lambda t: any(j in t for j in JUNK))
    df = df[~junk]

    print(f"  with text            : {n0:,}")
    print(f"  long enough          : {n1:,}  (dropped {n0-n1:,} short)")
    print(f"  not stale content    : {n2:,}  (dropped {stale.sum():,} wrong-article)")
    print(f"  not a junk page      : {len(df):,}  (dropped {int(junk.sum()):,})")
    return df.reset_index(drop=True)



def run_lda(texts, n_topics, lang, max_features=8000):
    vec = CountVectorizer(max_df=0.55, min_df=8, max_features=max_features,
                          stop_words=stop_for(lang),
                          token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
    X = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                    learning_method="online", max_iter=25,
                                    n_jobs=-1)
    W = lda.fit_transform(X)
    vocab = vec.get_feature_names_out()
    return lda, W, vocab


def top_words(lda, vocab, k=15):
    out = []
    for i, comp in enumerate(lda.components_):
        idx = comp.argsort()[::-1][:k]
        out.append([(vocab[j], float(comp[j])) for j in idx])
    return out


def wordclouds(tw, lang, n_topics):
    ncol = 3
    nrow = (n_topics + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.2, nrow * 2.8))
    axes = axes.ravel() if n_topics > 1 else [axes]
    for i, words in enumerate(tw):
        freq = {w: v for w, v in words}
        wc = WordCloud(width=520, height=330, background_color="white",
                       colormap="RdGy", prefer_horizontal=0.95).generate_from_frequencies(freq)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Topic {i+1}", fontsize=11, fontweight="bold")
        axes[i].axis("off")
    for j in range(len(tw), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Topics in {lang.upper()} articles", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / f"lda_wordcloud_{lang}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def chart_by_country(sub, lang, n_topics):
    top = sub["source_country"].value_counts()
    top = top[top >= 40].head(12).index
    if len(top) < 2:
        return
    m = (sub[sub["source_country"].isin(top)]
         .groupby("source_country")[[f"t{i}" for i in range(n_topics)]].mean())
    m = m.loc[top]
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.45 * len(m))))
    im = ax.imshow(m.values, aspect="auto", cmap="RdPu")
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m.index, fontsize=9)
    ax.set_xticks(range(n_topics))
    ax.set_xticklabels([f"T{i+1}" for i in range(n_topics)], fontsize=9)
    ax.set_title(f"Topic mix by country, {lang.upper()}", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="average share")
    fig.tight_layout()
    fig.savefig(FIGURES / f"lda_by_country_{lang}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def chart_over_time(sub, lang, n_topics):
    s = sub.copy()
    s["month"] = pd.to_datetime(s["published"], errors="coerce", utc=True).dt.strftime("%Y-%m")
    m = s.groupby("month")[[f"t{i}" for i in range(n_topics)]].mean().sort_index()
    if len(m) < 3:
        return
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i in range(n_topics):
        ax.plot(m.index, m[f"t{i}"], label=f"Topic {i+1}", linewidth=1.7)
    ax.set_xticks(range(0, len(m), max(1, len(m) // 12)))
    ax.set_xticklabels([m.index[i] for i in range(0, len(m), max(1, len(m) // 12))],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("average share of articles")
    ax.set_title(f"Topics over time, {lang.upper()}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / f"lda_over_time_{lang}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=int, default=8)
    ap.add_argument("--languages", nargs="*", default=None,
                    help="e.g. --languages en es fr. Default: all with enough articles")
    ap.add_argument("--pooled", action="store_true",
                    help="one model over all languages (shows the language problem)")
    ap.add_argument("--max-per-lang", type=int, default=25_000,
                    help="cap per language to keep it fast")
    a = ap.parse_args()

    print("loading and filtering...")
    df = load_and_clean()

    # Detecting language on every article is far too slow. Subsample first.
    # We only need enough per language to fit a topic model.
    budget = min(len(df), max(a.max_per_lang * 6, 30_000))
    work = df.sample(n=budget, random_state=42).copy()
    print(f"\ndetecting language on a {budget:,}-article subsample "
          f"(a few minutes)...")
    work["lang"] = detect_langs(work["article_text"])
    work = work[work["lang"].notna()]
    # A Russian article misdetected as Portuguese put a whole junk topic into
    # the PT model. Keep only languages we have stopwords for, so every model
    # is properly cleaned.
    known = set(STOP) | set(BOILERPLATE)
    unknown = (~work["lang"].isin(known)).sum()
    if unknown:
        print(f"  {unknown:,} articles are in languages we have no stopword list "
              f"for; they are excluded")
    work = work[work["lang"].isin(known)]
    df = work

    counts = df["lang"].value_counts()
    print("\narticles per language:")
    print(counts.head(12).to_string())

    L = ["LDA TOPIC MODELLING", "=" * 60,
         f"articles used: {len(df):,}", ""]

    if a.pooled:
        langs = ["pooled"]
        groups = {"pooled": df.sample(n=min(a.max_per_lang, len(df)), random_state=42)}
        L.append("MODE: pooled (all languages in one model)")
        L.append("Expect topics that separate by language, not by theme.")
    else:
        langs = a.languages or [l for l, c in counts.items()
                                if c >= MIN_ARTICLES_PER_LANG][:6]
        groups = {}
        for l in langs:
            g = df[df["lang"] == l]
            if len(g) < MIN_ARTICLES_PER_LANG:
                print(f"  skipping {l}: only {len(g)} articles")
                continue
            groups[l] = g.sample(n=min(a.max_per_lang, len(g)), random_state=42)
        L.append(f"MODE: separate model per language ({', '.join(groups)})")
    L.append("")

    all_assign = []
    for lang, sub in groups.items():
        print(f"\n=== {lang.upper()}  ({len(sub):,} articles) ===")
        lda, W, vocab = run_lda(sub["article_text"].astype(str).str.slice(0, 6000),
                                a.topics, lang if lang != "pooled" else "en")
        tw = top_words(lda, vocab)
        L.append(f"--- {lang.upper()}  n={len(sub):,} ---")
        for i, words in enumerate(tw):
            ws = ", ".join(w for w, _ in words[:12])
            print(f"  Topic {i+1}: {ws}")
            L.append(f"  Topic {i+1}: {ws}")
        L.append("")

        pd.DataFrame({"topic": [f"Topic {i+1}" for i in range(a.topics)],
                      "top_words": [", ".join(w for w, _ in t) for t in tw]}
                     ).to_csv(TABLES / f"lda_topics_{lang}.csv", index=False)

        for i in range(a.topics):
            sub[f"t{i}"] = W[:, i]
        sub["main_topic"] = W.argmax(axis=1) + 1
        all_assign.append(sub[["url", "source_country", "published", "main_topic"]
                              ].assign(lang=lang))

        wordclouds(tw, lang, a.topics)
        chart_by_country(sub, lang, a.topics)
        chart_over_time(sub, lang, a.topics)
        print(f"  wrote lda_wordcloud_{lang}.png, lda_by_country_{lang}.png, "
              f"lda_over_time_{lang}.png")

    pd.concat(all_assign, ignore_index=True).to_csv(PROCESSED / "lda_article_topics.csv", index=False)
    open(REPORTS / "lda_report.txt", "w").write("\n".join(L))
    print("\nwrote lda_article_topics.csv and lda_report.txt")
    print("\nNEXT: read the topic word lists and give each topic a name.")
    print("LDA does not name topics. That part is yours.")


if __name__ == "__main__":
    main()
