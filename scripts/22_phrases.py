#!/usr/bin/env python3
"""
22_phrases.py
-------------
Common PHRASES, not common words.

WHY THIS EXISTS

LDA gave us word lists. Some of those words are useless on their own. An English
topic contained "people". People doing what? "People came forward", "young
people", "people in power" and "people are tired of it" are four different
stories, and the topic model collapses them into one token.

This script does three things about that.

  1. N-GRAMS. The most frequent word sequences per language, from two words up
     to --ngram-max (default 4), with a syndication guard so one wire story
     republished 300 times cannot manufacture a phrase.

  2. COLLOCATIONS. For a chosen word, which other words appear near it far more
     often than chance would allow. This is the standard method in corpus-based
     discourse analysis, and it is what actually answers the "people" question.

  3. REPEATED SENTENCES. Whole sentences that appear in several articles. These
     are usually quotations: the same line from Milano, Deneuve, a court
     ruling or a wire agency, reproduced across outlets and countries.

     N-grams cannot find these. A 25-word quote is not a 4-gram, and raising
     --ngram-max far enough to reach it would be enormously expensive for a
     result that whole-sentence matching gets almost for free.

     This runs on the DEDUPLICATED corpus, which matters: duplicate articles
     were already removed, so a sentence repeating here is a line genuinely
     re-quoted by a different article, not the same article counted twice.
     Each repeated sentence is reported with how many outlets and countries
     carried it and the span of dates over which it travelled.

Plus KWIC (keyword in context) concordance lines, so every number can be traced
back to sentences you can read.

WHICH SCORE, AND WHY IT MATTERS

Three scores are reported for every collocate pair.

  log-Dice   Use this one for ranking. It is frequency-stable, which means the
             English subcorpus (38k articles) and the Swedish one (2k) produce
             numbers on the same scale and can be compared directly. Range is
             roughly 0-14; anything above ~7 is a strong collocation.

  PMI        Mutual information. Ranks rare pairs far too highly: a typo that
             happens to sit next to the node word twice can outrank a genuine
             collocation. Reported for reference, not for ranking.

  LL         Log-likelihood. Good significance test, but it scales with corpus
             size, so a Swedish LL of 40 and an English LL of 40 do not mean
             the same thing.

HOW A COLLOCATION IS COUNTED

For each occurrence of the node word, every non-stopword token within a window
of +/- N tokens (default 5) counts as one co-occurrence. Sentence boundaries are
not respected, which is standard practice and slightly generous.

INPUT
    data/processed/deduped_articles.csv   preferred: unique articles only
    data/processed/labeled_articles.csv   fallback if the deduped file is absent

    The deduped file is preferred on purpose. Topic modelling currently reads
    the pre-deduplication file, which counts syndicated copies; phrase counts
    are far more sensitive to that than topic models are, because 300 copies of
    one story make its phrasing look like a national style.

OUTPUT
    outputs/tables/ngrams_<lang>.csv              phrases with frequency + dispersion
    outputs/tables/collocations_<lang>.csv        collocates of each target word
    outputs/tables/kwic_<lang>.csv                concordance lines to read
    outputs/tables/repeated_sentences_<lang>.csv  quotations that travelled
    outputs/reports/phrases_report.txt            the summary

USAGE
    pip install pandas ftfy langdetect
    python scripts/22_phrases.py
    python scripts/22_phrases.py --targets people,women,victim,movement
    python scripts/22_phrases.py --langs en,fr --max-per-lang 40000
    python scripts/22_phrases.py --ngram-max 6      # longer fixed-length phrases
"""

import argparse
import hashlib
import math
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# Repository paths.
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

# Stopword lists, boilerplate filters and language detection are shared with
# 17_lda_topics.py. Sharing them is deliberate: if the two scripts filtered
# different words, the phrase tables and the topic tables would stop being
# comparable, which is the main reason to run both.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _textutils import (stop_for, detect_langs, repair_rot47,   # noqa: E402
                        _alpha_share)

MIN_CHARS = 800                 # same floor as the topic model
STALE = re.compile(r"\b(?:202[3-6])\b")   # a 2023-2026 date means the page
                                          # served today's content, not the article

try:
    import ftfy
    HAVE_FTFY = True
except ImportError:
    HAVE_FTFY = False

TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)

# Default node words. Deliberately a mix: generic words that made LDA hard to
# read, and words whose framing is the actual research question.
DEFAULT_TARGETS = {
    "en": ["people", "women", "victim", "movement", "allegations", "harassment"],
    "es": ["mujeres", "victima", "movimiento", "denuncia", "violencia", "acoso"],
    "fr": ["femmes", "victime", "mouvement", "plainte", "violences", "harcelement"],
    "pt": ["mulheres", "vitima", "movimento", "denuncia", "violencia", "assedio"],
    "de": ["frauen", "opfer", "bewegung", "vorwuerfe", "gewalt", "belaestigung"],
    "sv": ["kvinnor", "offer", "roerelsen", "anklagelser", "vald", "trakasserier"],
    "it": ["donne", "vittima", "movimento", "denuncia", "violenza", "molestie"],
}


# ── loading ────────────────────────────────────────────────────────────────
def pick_infile(explicit):
    if explicit:
        return _Path(explicit)
    dd = PROCESSED / "deduped_articles.csv"
    if dd.exists():
        return dd
    return PROCESSED / "labeled_articles.csv"


def load_and_clean(path, extra_cols=()):
    """Same cleaning as 17_lda_topics.py, but tolerant about which columns the
    input happens to carry — the deduped file and the labelled file differ."""
    want = ["url", "source_country", "outlet", "published", "article_text",
            "fetch_status", "text_chars"] + [c for c in extra_cols if c]
    head = pd.read_csv(path, nrows=1)
    cols = [c for c in want if c in head.columns]
    if "article_text" not in cols:
        sys.exit(f"{path} has no article_text column")

    parts = []
    for ch in pd.read_csv(path, usecols=cols, chunksize=100_000, low_memory=False):
        if "fetch_status" in ch.columns:
            ch = ch[ch["fetch_status"].isin(["OK", "OK_WAYBACK"])]
        ch = ch[ch["article_text"].notna()]
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df)

    if "text_chars" in df.columns:
        df = df[df["text_chars"] >= MIN_CHARS]
    else:
        df = df[df["article_text"].astype(str).str.len() >= MIN_CHARS]
    n1 = len(df)

    stale = df["article_text"].astype(str).str.slice(0, 6000).str.contains(STALE, na=False)
    df = df[~stale]

    if HAVE_FTFY:
        bad = df["article_text"].astype(str).str.contains("Ã|Â|ð|ï¼", regex=True, na=False)
        if bad.any():
            df.loc[bad, "article_text"] = (df.loc[bad, "article_text"]
                                           .astype(str).apply(ftfy.fix_text))
            print(f"  repaired broken encoding in {int(bad.sum()):,} articles")
    else:
        print("  NOTE: ftfy not installed — mojibake will pollute the phrases.")
        print("        pip install ftfy")

    # Some US local-news platforms serve the article body ROT47-encoded as a
    # scraper deterrent. Those articles pass every length check and then
    # contribute nothing but noise. Decode them; see _textutils.repair_rot47.
    fixed = df["article_text"].astype(str).apply(repair_rot47)
    n_rot = int(sum(c for _, c in fixed))
    if n_rot:
        df["article_text"] = [t for t, _ in fixed]
        print(f"  decoded ROT47-obfuscated body text in {n_rot:,} articles")

    if "outlet" not in df.columns:
        df["outlet"] = "UNKNOWN"
    if "source_country" not in df.columns:
        df["source_country"] = "UNKNOWN"

    print(f"  with text     : {n0:,}")
    print(f"  long enough   : {n1:,}")
    print(f"  after cleaning: {len(df):,}")
    return df.reset_index(drop=True)


def tokenize(text):
    return [t.lower() for t in TOKEN.findall(str(text)) if len(t) > 1]


# A sentence ends at . ! ? … or their full-width forms, followed by whitespace.
# Deliberately simple: it over-splits on abbreviations ("Mr. Weinstein"), which
# costs us a few sentences but never invents a repeat that is not there.
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
QUOTE_CHARS = '"\u201c\u201d\u00ab\u00bb\u201e\u2018\u2019\u300c\u300d'


def sentences(text, min_words, max_words):
    """Yield (normalised_key, original_sentence, is_quote).

    The key is lowercased and stripped of punctuation so that a sentence
    reproduced with different quote marks or a trailing attribution comma still
    matches. The original is kept for display."""
    for raw in SENT_SPLIT.split(str(text)):
        raw = raw.strip()
        if not raw:
            continue
        toks = tokenize(raw)
        if not (min_words <= len(toks) <= max_words):
            continue
        # Backstop against any obfuscated or mojibake text that survived
        # cleaning: real prose is overwhelmingly letters once spaces are
        # ignored, encoded text is not.
        if _alpha_share(raw) < 0.55:
            continue
        yield " ".join(toks), raw, any(c in raw for c in QUOTE_CHARS)


def skey(norm):
    """64-bit stable digest. Storing digests rather than the sentences keeps the
    counter small on a corpus with roughly half a million sentences; the text
    itself is recovered in the attribution pass, only for repeats."""
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=8).digest()


# ── statistics ─────────────────────────────────────────────────────────────
def log_dice(f_ab, f_a, f_b):
    """Frequency-stable association score. Comparable across subcorpora of very
    different sizes, which is exactly our situation: 38k English articles
    against 2k Swedish ones."""
    return 14 + math.log2(2 * f_ab / (f_a + f_b)) if f_ab else float("-inf")


def pmi(f_ab, f_a, f_b, n_tokens, span):
    exp = f_a * f_b * span / n_tokens
    return math.log2(f_ab / exp) if f_ab and exp else float("-inf")


def loglik(f_ab, f_a, f_b, n_tokens):
    """Dunning's log-likelihood on the 2x2 contingency table."""
    def _x(o, e):
        return o * math.log(o / e) if o > 0 and e > 0 else 0.0
    a = f_ab
    b = f_a - f_ab
    c = f_b - f_ab
    d = n_tokens - f_a - f_b + f_ab
    if min(a, b, c, d) < 0:
        return 0.0
    tot = a + b + c + d
    ea = (a + b) * (a + c) / tot
    eb = (a + b) * (b + d) / tot
    ec = (c + d) * (a + c) / tot
    ed = (c + d) * (b + d) / tot
    return 2 * (_x(a, ea) + _x(b, eb) + _x(c, ec) + _x(d, ed))


# ── the two passes ─────────────────────────────────────────────────────────
def pass1_vocab(texts):
    """Count unigrams. Pass 2 only builds n-grams from tokens that survive a
    frequency floor here — that is what keeps the n-gram counter from growing
    to several gigabytes on 25 million tokens."""
    uni = Counter()
    for t in texts:
        uni.update(tokenize(t))
    return uni


def pass2(df, uni, stops, targets, window, min_tok_freq, kwic_per_target,
          ngram_max, sent_min_words, sent_max_words):
    """Counting pass: n-gram frequencies, document frequencies, sentence
    digests, collocations for the target words, and concordance lines.

    Outlet and country attribution is deliberately NOT done here. Holding a
    per-phrase outlet counter meant one Counter object for every candidate
    phrase, which on a full 25,000-article language reaches several gigabytes.
    Pass 3 does attribution instead, for the few thousand phrases that survive."""
    keep = {w for w, c in uni.items() if c >= min_tok_freq}
    stopset = set(stops)
    tset = set(targets)

    ngram_f = Counter()                       # phrase -> occurrences
    ngram_docs = Counter()                    # phrase -> documents
    sent_docs = Counter()                     # sentence digest -> documents
    colloc = {t: Counter() for t in targets}  # node -> collocate -> co-occurrences
    node_hits = Counter()
    kwic = []
    kwic_n = Counter()                        # per-target quota
    n_tokens = 0
    n_sents = 0

    for row in df.itertuples(index=False):
        text = row.article_text
        toks = tokenize(text)
        n_tokens += len(toks)
        outlet = getattr(row, "outlet", "UNKNOWN")

        # --- n-grams: content-bounded, so "of the" and "said that" cannot win
        seen = set()
        for n in range(2, ngram_max + 1):
            for i in range(len(toks) - n + 1):
                w = toks[i:i + n]
                if w[0] in stopset or w[-1] in stopset:
                    continue
                if any(x not in keep for x in w):
                    continue
                p = " ".join(w)
                ngram_f[p] += 1
                seen.add(p)
        for p in seen:
            ngram_docs[p] += 1

        # --- whole sentences, counted once per article. Counting per article
        #     rather than per occurrence stops a single article that repeats its
        #     own strapline from looking like a travelling quotation.
        sseen = set()
        for norm, _raw, _q in sentences(text, sent_min_words, sent_max_words):
            n_sents += 1
            sseen.add(skey(norm))
        for k in sseen:
            sent_docs[k] += 1

        # --- collocations + KWIC around each target word
        if tset:
            for i, tok in enumerate(toks):
                if tok not in tset:
                    continue
                node_hits[tok] += 1
                lo, hi = max(0, i - window), min(len(toks), i + window + 1)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    c = toks[j]
                    if c in stopset or c not in keep or c == tok:
                        continue
                    colloc[tok][c] += 1
                if kwic_n[tok] < kwic_per_target:
                    kwic_n[tok] += 1
                    kwic.append({
                        "target": tok,
                        "left": " ".join(toks[max(0, i - 8):i]),
                        "node": tok,
                        "right": " ".join(toks[i + 1:i + 9]),
                        "outlet": outlet,
                        "country": getattr(row, "source_country", "UNKNOWN"),
                        "url": getattr(row, "url", ""),
                    })

    return dict(ngram_f=ngram_f, ngram_docs=ngram_docs, sent_docs=sent_docs,
                colloc=colloc, node_hits=node_hits, kwic=kwic,
                n_tokens=n_tokens, n_sents=n_sents, uni=uni)


def pass3_attribution(df, phrases, sent_keys, sent_min_words, sent_max_words):
    """Second read of the text, restricted to the phrases and sentences that
    already proved they repeat. Records which outlets and countries carried
    each one and over what span of dates — that is what separates a quotation
    that travelled from one outlet's house phrasing."""
    ph_outlet = defaultdict(Counter)
    ph_country = defaultdict(set)
    st_outlet = defaultdict(Counter)
    st_country = defaultdict(set)
    st_text = {}
    st_quote = {}
    st_dates = defaultdict(list)

    maxn = max((p.count(" ") + 1) for p in phrases) if phrases else 0

    for row in df.itertuples(index=False):
        outlet = getattr(row, "outlet", "UNKNOWN")
        country = getattr(row, "source_country", "UNKNOWN")
        published = getattr(row, "published", None)
        text = row.article_text

        if phrases:
            toks = tokenize(text)
            for n in range(2, maxn + 1):
                for i in range(len(toks) - n + 1):
                    p = " ".join(toks[i:i + n])
                    if p in phrases:
                        ph_outlet[p][outlet] += 1
                        ph_country[p].add(country)

        for norm, raw, is_q in sentences(text, sent_min_words, sent_max_words):
            k = skey(norm)
            if k in sent_keys:
                st_outlet[k][outlet] += 1
                st_country[k].add(country)
                st_dates[k].append(published)
                if k not in st_text:
                    st_text[k] = raw
                    st_quote[k] = is_q
                elif is_q and not st_quote[k]:
                    st_text[k] = raw          # prefer a version with quote marks
                    st_quote[k] = True

    return dict(ph_outlet=ph_outlet, ph_country=ph_country,
                st_outlet=st_outlet, st_country=st_country,
                st_text=st_text, st_quote=st_quote, st_dates=st_dates)


# ── output ─────────────────────────────────────────────────────────────────
def ngram_table(res, att, min_docs, max_outlet_share, top):
    rows = []
    for p, f in res["ngram_f"].most_common():
        d = res["ngram_docs"][p]
        if d < min_docs:
            continue
        oc = att["ph_outlet"].get(p)
        if not oc:
            continue
        top_out, top_n = oc.most_common(1)[0]
        share = top_n / sum(oc.values())
        rows.append({
            "phrase": p,
            "words": p.count(" ") + 1,
            "occurrences": f,
            "documents": d,
            "outlets": len(oc),
            "countries": len(att["ph_country"].get(p, ())),
            "top_outlet": top_out,
            "top_outlet_share": round(share, 3),
            # A phrase concentrated in one outlet is that outlet's house style
            # or an un-caught syndicated copy, not a feature of the language.
            "syndication_flag": share > max_outlet_share,
        })
        if len(rows) >= top * 4:
            break
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    return t.sort_values(["syndication_flag", "occurrences"],
                         ascending=[True, False]).head(top)


def sentence_table(res, att, min_docs, min_outlets, top):
    """Sentences that appear in several different articles. Ranked by how many
    distinct outlets carried them, not by raw frequency: a line printed twenty
    times by one newspaper is that newspaper's boilerplate, while a line
    printed once each by twenty newspapers is a quotation that travelled."""
    rows = []
    for k, d in res["sent_docs"].most_common():
        if d < min_docs:
            continue
        oc = att["st_outlet"].get(k)
        if not oc:
            continue
        outlets = len(oc)
        dates = [x for x in att["st_dates"].get(k, []) if pd.notna(x)]
        dates = pd.to_datetime(pd.Series(dates), errors="coerce", utc=True).dropna()
        rows.append({
            "sentence": att["st_text"].get(k, ""),
            "words": len(att["st_text"].get(k, "").split()),
            "articles": d,
            "outlets": outlets,
            "countries": len(att["st_country"].get(k, ())),
            "looks_like_quote": bool(att["st_quote"].get(k)),
            "first_seen": dates.min().date().isoformat() if len(dates) else "",
            "last_seen": dates.max().date().isoformat() if len(dates) else "",
            "days_spanned": int((dates.max() - dates.min()).days) if len(dates) > 1 else 0,
            "single_outlet": outlets < min_outlets,
        })
        if len(rows) >= top * 4:
            break
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    return t.sort_values(["single_outlet", "outlets", "articles"],
                         ascending=[True, False, False]).head(top)


def colloc_table(res, window, top):
    rows = []
    N = res["n_tokens"]
    span = 2 * window
    for node, counter in res["colloc"].items():
        f_a = res["uni"].get(node, 0)
        if not f_a:
            continue
        for c, f_ab in counter.most_common(400):
            f_b = res["uni"].get(c, 0)
            if f_b < 5 or f_ab < 3:
                continue
            rows.append({
                "node": node,
                "collocate": c,
                "co_occurrences": f_ab,
                "node_freq": f_a,
                "collocate_freq": f_b,
                "log_dice": round(log_dice(f_ab, f_a, f_b), 3),
                "pmi": round(pmi(f_ab, f_a, f_b, N, span), 3),
                "log_likelihood": round(loglik(f_ab, f_a, f_b, N), 1),
            })
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    return (t.sort_values(["node", "log_dice"], ascending=[True, False])
             .groupby("node", group_keys=False).head(top))


def phrase_frames(res, targets, top=6):
    """The most frequent n-grams that actually contain each target word. This is
    the direct answer to 'what does people mean here'."""
    out = {}
    for t in targets:
        hits = [(p, f) for p, f in res["ngram_f"].most_common(40_000)
                if t in p.split()]
        out[t] = hits[:top]
    return out


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default=None)
    ap.add_argument("--langs", default="en,es,fr,pt,de,sv")
    ap.add_argument("--targets", default=None,
                    help="comma-separated node words; default is a per-language set")
    ap.add_argument("--max-per-lang", type=int, default=25_000)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-token-freq", type=int, default=5)
    ap.add_argument("--min-docs", type=int, default=10)
    ap.add_argument("--max-outlet-share", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=400)
    ap.add_argument("--kwic", type=int, default=40, help="lines kept per target")
    ap.add_argument("--ngram-max", type=int, default=4,
                    help="longest fixed-length phrase to count (default 4). "
                         "Long quotations are found by sentence matching "
                         "instead, which is far cheaper than raising this.")
    ap.add_argument("--sent-min-words", type=int, default=6)
    ap.add_argument("--sent-max-words", type=int, default=60)
    ap.add_argument("--sent-min-articles", type=int, default=3,
                    help="a sentence must appear in this many articles to count "
                         "as repeated")
    ap.add_argument("--sent-min-outlets", type=int, default=2,
                    help="below this, a repeated sentence is treated as one "
                         "outlet's boilerplate rather than a travelling quote")
    ap.add_argument("--lang-col", default=None,
                    help="use an existing column instead of running language "
                         "detection (detection on 100k articles is slow)")
    a = ap.parse_args()

    path = pick_infile(a.infile)
    if not path.exists():
        sys.exit(f"{path} not found. Run 08_dedupe.py or 07_fetch_and_label.py first.")
    print(f"reading {path.name}")
    df = load_and_clean(path, extra_cols=(a.lang_col,))

    langs = [l.strip() for l in a.langs.split(",") if l.strip()]
    budget = min(len(df), max(a.max_per_lang * 6, 30_000))
    work = df.sample(n=budget, random_state=42) if len(df) > budget else df
    work = work.copy()
    if a.lang_col and a.lang_col in work.columns and work[a.lang_col].notna().any():
        print(f"using existing '{a.lang_col}' column for language")
        work["lang"] = work[a.lang_col].astype(str).str.lower().str[:2]
    else:
        print(f"detecting language on {len(work):,} articles (slow, one pass)...")
        work["lang"] = detect_langs(work["article_text"])
    work = work[work["lang"].isin(langs)]
    print(work["lang"].value_counts().to_string())

    L = ["PHRASES, COLLOCATIONS AND REPEATED SENTENCES", "=" * 62,
         f"source file : {path.name}",
         f"n-grams     : 2 to {a.ngram_max} words",
         f"sentences   : {a.sent_min_words}-{a.sent_max_words} words, repeated in "
         f">= {a.sent_min_articles} articles",
         f"window      : +/-{a.window} tokens",
         f"ranking     : log-Dice (frequency-stable across subcorpora)", ""]

    for lang in langs:
        sub = work[work["lang"] == lang]
        if len(sub) < 300:
            print(f"\n{lang}: only {len(sub)} articles, skipped")
            continue
        if len(sub) > a.max_per_lang:
            sub = sub.sample(n=a.max_per_lang, random_state=42)
        print(f"\n=== {lang.upper()}  ({len(sub):,} articles) ===")

        targets = ([t.strip().lower() for t in a.targets.split(",")]
                   if a.targets else DEFAULT_TARGETS.get(lang, []))
        stops = stop_for(lang)

        print("  pass 1: vocabulary")
        uni = pass1_vocab(sub["article_text"])
        print(f"    {len(uni):,} distinct tokens")
        print("  pass 2: n-grams, sentences, collocations, concordance")
        res = pass2(sub, uni, stops, targets, a.window, a.min_token_freq, a.kwic,
                    a.ngram_max, a.sent_min_words, a.sent_max_words)
        print(f"    {res['n_tokens']:,} tokens, {res['n_sents']:,} sentences, "
              f"{len(res['ngram_f']):,} candidate phrases")

        # Only phrases and sentences that already repeat are worth attributing.
        survivors = {p for p, d in res["ngram_docs"].items() if d >= a.min_docs}
        rep_sents = {k for k, d in res["sent_docs"].items()
                     if d >= a.sent_min_articles}
        print(f"  pass 3: attribution for {len(survivors):,} phrases and "
              f"{len(rep_sents):,} repeated sentences")
        att = pass3_attribution(sub, survivors, rep_sents,
                                a.sent_min_words, a.sent_max_words)

        ng = ngram_table(res, att, a.min_docs, a.max_outlet_share, a.top)
        co = colloc_table(res, a.window, 40)
        kw = pd.DataFrame(res["kwic"])
        st = sentence_table(res, att, a.sent_min_articles, a.sent_min_outlets, a.top)

        ng.to_csv(TABLES / f"ngrams_{lang}.csv", index=False)
        co.to_csv(TABLES / f"collocations_{lang}.csv", index=False)
        kw.to_csv(TABLES / f"kwic_{lang}.csv", index=False)
        st.to_csv(TABLES / f"repeated_sentences_{lang}.csv", index=False)

        L.append(f"--- {lang.upper()}  n={len(sub):,}  tokens={res['n_tokens']:,} ---")
        if not ng.empty:
            clean = ng[~ng["syndication_flag"]].head(15)
            L.append("  most frequent phrases (syndication-filtered):")
            for _, r in clean.iterrows():
                L.append(f"    {r['occurrences']:>6,}  {r['phrase']}"
                         f"   [{r['documents']:,} docs, {r['outlets']} outlets]")
            dropped = int(ng["syndication_flag"].sum())
            if dropped:
                L.append(f"    ({dropped} phrases flagged as single-outlet and set aside)")
        for t, frames in phrase_frames(res, targets).items():
            if frames:
                L.append(f"  phrases containing '{t}':")
                for p, f in frames:
                    L.append(f"    {f:>6,}  {p}")
        if not co.empty:
            for node, g in co.groupby("node"):
                words = ", ".join(g.head(10)["collocate"])
                L.append(f"  strongest collocates of '{node}': {words}")
        if not st.empty:
            travelled = st[~st["single_outlet"]]
            L.append(f"  repeated sentences: {len(st):,} found, "
                     f"{len(travelled):,} carried by more than one outlet")
            for _, r in travelled.head(8).iterrows():
                span = f", {r['days_spanned']}d apart" if r["days_spanned"] else ""
                L.append(f"    [{r['outlets']} outlets, {r['countries']} countries"
                         f"{span}]{' QUOTED' if r['looks_like_quote'] else ''}")
                L.append(f"      \u201c{r['sentence'][:200]}\u201d")
        L.append("")

    (REPORTS / "phrases_report.txt").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {REPORTS / 'phrases_report.txt'}")
    print("tables in outputs/tables/: ngrams_<lang>.csv, collocations_<lang>.csv,")
    print("                           kwic_<lang>.csv, repeated_sentences_<lang>.csv")
    print("\nRead the KWIC file before quoting any phrase. A collocation is a")
    print("statistical claim; the concordance is the evidence for it.")


if __name__ == "__main__":
    main()
