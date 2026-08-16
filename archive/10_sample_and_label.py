#!/usr/bin/env python3
"""
10_sample_and_label.py
----------------------
Draws a stratified sample from the fetched articles, labels it, and reports the
one number that settles the composition question:

    Of the articles that matched only general gender-violence vocabulary,
    what share are actually about the movement?

If that share is high, the generic-matched articles belong in the study and the
corpus stands. If it is low, they are a different population and we either split
the study or filter them out. Either way it becomes a measured fact instead of
an argument.

WHY A SAMPLE
  Labeling all 107,481 fetched articles takes roughly 11 hours. A stratified
  sample of 2,000 answers the composition question in about 15 minutes, with a
  margin of error near +/-3 points per stratum. The full run can happen after.

STRATIFICATION
  Cells are (matched a hashtag?) x (language group of the publishing country).
  Both dimensions are sampled because they are confounded: Romance-language
  countries are also the generic-vocabulary countries, so a sample stratified on
  only one of them cannot separate the two effects.

SETUP
    pip install google-genai pandas tqdm python-dotenv
    # .env in this folder must contain:  GOOGLE_API_KEY=your-key

RUN
    python 10_sample_and_label.py                 # 2,000 articles, ~15 min
    python 10_sample_and_label.py --n 1000        # faster, wider error bars
    python 10_sample_and_label.py --dry-run       # draw sample only, no API

OUTPUT
    sample_labeled.csv        every sampled article, its label and reasoning
    sample_report.txt         the summary tables, ready to paste into the brief
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

IN_FILE = "labeled_articles.csv"
OUT_CSV = "sample_labeled.csv"
OUT_TXT = "sample_report.txt"
MODEL = "gemini-2.5-flash"
MAX_SEND = 5000
API_WORKERS = 4   # MacBook Air; raise only if the preflight is fast

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
    "Other": [],
}
REV = {c: l for l, cs in LANG_GROUPS.items() for c in cs}

# ── the rubric. Same as the full run, so sample labels are comparable. ──
RUBRIC = """You are coding a news article for an academic study of how global news
media covered the #MeToo movement and its backlash (2017-2019).

Read the article, then answer ONLY with a JSON object with these keys IN THIS
ORDER. The order matters: work through the evidence before committing.

1. "language": ISO 639-1 code of the article's language.

2. "is_about_movement": true/false. TRUE if the article concerns the #MeToo
   movement, a comparable feminist movement, or a specific case explicitly
   framed as part of one. FALSE for routine crime reporting, court reports, or
   statistics about gender violence that make no reference to a movement.
   This distinction is the point of the study - be strict. An article about a
   single assault case with no movement framing is FALSE even though its
   subject matter is gender violence.

3. "movement_reasoning": one or two sentences. What in the text made this
   true or false? Name the specific cue.

4. "article_type": one of "straight_news", "opinion_editorial",
   "feature_analysis", "meme_roundup", "interview", "other".

5. "primary_focus": one short phrase - what the article is mainly about.

6. "voice_balance": one of "movement_voices_dominate",
   "backlash_voices_dominate", "balanced", "no_quoted_voices".

7. "evaluative_language": list of words or phrases where the JOURNALIST (not a
   quoted source) makes an evaluative choice. Empty list if none.

8. "stance_reasoning": TWO TO FOUR sentences. Apply the dominant-effect test:
   what is the reader's takeaway, and what in the text produces it? Do NOT
   state the stance label here - reason toward it.

9. "stance": one of "support", "backlash", "neutral", "satire",
   "not_applicable". Use "not_applicable" when is_about_movement is false.
   Tone of subject matter is NOT stance: an article describing an assault
   sympathetically is support, not backlash.

10. "stance_confidence": "H", "M", or "L".

11. "evidence": ONE short quote (under 15 words) from the article, verbatim.

Return ONLY the JSON object. No preamble, no markdown fences."""

SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "is_about_movement": {"type": "boolean"},
        "movement_reasoning": {"type": "string"},
        "article_type": {"type": "string", "enum": [
            "straight_news", "opinion_editorial", "feature_analysis",
            "meme_roundup", "interview", "other"]},
        "primary_focus": {"type": "string"},
        "voice_balance": {"type": "string", "enum": [
            "movement_voices_dominate", "backlash_voices_dominate",
            "balanced", "no_quoted_voices"]},
        "evaluative_language": {"type": "array", "items": {"type": "string"}},
        "stance_reasoning": {"type": "string"},
        "stance": {"type": "string", "enum": [
            "support", "backlash", "neutral", "satire", "not_applicable"]},
        "stance_confidence": {"type": "string", "enum": ["H", "M", "L"]},
        "evidence": {"type": "string"},
    },
    "required": ["language", "is_about_movement", "movement_reasoning",
                 "article_type", "primary_focus", "voice_balance",
                 "evaluative_language", "stance_reasoning", "stance",
                 "stance_confidence", "evidence"],
    "propertyOrdering": ["language", "is_about_movement", "movement_reasoning",
                         "article_type", "primary_focus", "voice_balance",
                         "evaluative_language", "stance_reasoning", "stance",
                         "stance_confidence", "evidence"],
}
FIELDS = SCHEMA["propertyOrdering"]


def wilson(k, n, z=1.96):
    """95% CI for a proportion. Wilson interval - behaves at small n and
    near 0/1, unlike the normal approximation."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_fetched(path):
    """Read only what sampling needs, in chunks, keeping only usable rows."""
    keep = ["url", "source_country", "has_hashtag_term", "fetch_status",
            "keywords_matched", "text_chars"]
    parts = []
    for ch in pd.read_csv(path, usecols=keep, chunksize=200_000, low_memory=False):
        parts.append(ch[ch["fetch_status"].isin(["OK", "OK_WAYBACK"])])
    return pd.concat(parts, ignore_index=True)


def attach_text(path, urls):
    """Second pass: pull article_text for the sampled URLs only."""
    want = set(urls)
    keep = ["url", "fetched_title", "article_text", "source_country",
            "keywords_matched", "has_hashtag_term", "published", "outlet"]
    parts = []
    for ch in pd.read_csv(path, usecols=keep, chunksize=100_000, low_memory=False):
        hit = ch[ch["url"].isin(want)]
        if len(hit):
            parts.append(hit)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def label_one(client, types, row):
    prompt = (
        f"SOURCE COUNTRY: {row.get('source_country','')}\n"
        f"KEYWORDS THAT MATCHED THIS ARTICLE: {row.get('keywords_matched','')}\n"
        f"(The keywords are why this article entered the corpus. They may be "
        f"misleading - judge the article on its text, not on the keyword.)\n\n"
        f"HEADLINE: {row.get('fetched_title','')}\n\n"
        f"ARTICLE TEXT:\n{str(row.get('article_text',''))[:MAX_SEND]}")
    resp = client.models.generate_content(
        model=MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=RUBRIC,
            response_mime_type="application/json",
            response_schema=SCHEMA,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ))
    return json.loads(resp.text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--infile", default=IN_FILE)
    ap.add_argument("--dry-run", action="store_true",
                    help="draw and describe the sample, make no API calls")
    ap.add_argument("--workers", type=int, default=API_WORKERS)
    a = ap.parse_args()

    if not os.path.exists(a.infile):
        sys.exit(f"{a.infile} not found - run this in the folder with your fetch output.")

    print("reading fetched articles (large file, ~1 min)...")
    df = load_fetched(a.infile)
    print(f"  usable articles available: {len(df):,}")

    df["lang_group"] = df["source_country"].map(REV).fillna("Other")
    df["match_type"] = df["has_hashtag_term"].map({True: "hashtag", False: "generic"})
    df["stratum"] = df["match_type"] + " / " + df["lang_group"]

    # proportional allocation with a floor, so small cells are still estimable
    counts = df["stratum"].value_counts()
    FLOOR = 60
    alloc = {}
    for s, n in counts.items():
        prop = int(round(a.n * n / len(df)))
        alloc[s] = min(n, max(FLOOR if n >= FLOOR else n, prop))

    parts = []
    for s, k in alloc.items():
        parts.append(df[df["stratum"] == s].sample(n=k, random_state=20260731))
    samp = pd.concat(parts, ignore_index=True)

    print(f"\nstratified sample: {len(samp):,} articles across {len(alloc)} cells")
    tbl = samp.groupby(["match_type", "lang_group"]).size().unstack(fill_value=0)
    print(tbl.to_string())

    if a.dry_run:
        samp.to_csv("sample_frame_only.csv", index=False)
        print("\ndry run - wrote sample_frame_only.csv, no API calls made.")
        return

    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.environ.get("GOOGLE_API_KEY")
        except ImportError:
            pass
    if not key:
        sys.exit("No GOOGLE_API_KEY found. Put it in a .env file in this folder.")

    print("\npulling article text for the sample...")
    txt = attach_text(a.infile, samp["url"].tolist())
    samp = samp[["url", "stratum", "match_type", "lang_group"]].merge(
        txt, on="url", how="left")
    samp = samp[samp["article_text"].notna()]
    print(f"  text attached for {len(samp):,}")

    from google import genai
    from google.genai import types

    # Hard timeout. Without this a single hung connection blocks a worker
    # forever and the progress bar sits at 0% with no error.
    try:
        client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=90_000))   # milliseconds
    except Exception:
        client = genai.Client(api_key=key)   # older SDK: no timeout support

    # ── preflight: one call, so failure is visible in seconds not never ──
    print("preflight: sending one test request...")
    _t0 = time.time()
    try:
        _probe = samp.iloc[0].to_dict()
        label_one(client, types, _probe)
        _dt = time.time() - _t0
        print(f"  OK in {_dt:.1f}s. Estimated total: "
              f"~{len(samp)*_dt/a.workers/60:.0f} min with {a.workers} workers.")
    except Exception as e:
        print(f"  FAILED after {time.time()-_t0:.1f}s: {type(e).__name__}: {str(e)[:200]}")
        print("\nStopping. Fix this before running the full sample.")
        print("Run  python api_test.py  for a focused diagnosis.")
        sys.exit(1)

    for f in FIELDS:
        samp[f] = ""
    samp["llm_error"] = ""

    def do(pair):
        i, row = pair
        for attempt in range(5):
            try:
                return i, label_one(client, types, row), ""
            except Exception as e:
                msg = str(e)
                if "429" in msg or "quota" in msg.lower():
                    time.sleep(5 * (2 ** attempt))
                    continue
                return i, None, f"{type(e).__name__}"
        return i, None, "retries_exhausted"

    print(f"\nlabeling {len(samp):,} articles with {a.workers} workers...")
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(do, (i, r)) for i, r in samp.to_dict("index").items()]
        bar = tqdm(as_completed(futs), total=len(futs), desc="label",
                   mininterval=1.0, smoothing=0.1)
        for fu in bar:
            i, out, err = fu.result()
            if out:
                n_ok += 1
            else:
                n_err += 1
            bar.set_postfix(ok=n_ok, err=n_err, refresh=False)
            if out:
                for f in FIELDS:
                    v = out.get(f, "")
                    samp.at[i, f] = (json.dumps(v, ensure_ascii=False)
                                     if isinstance(v, (list, dict)) else v)
            else:
                samp.at[i, "llm_error"] = err

    samp.to_csv(OUT_CSV, index=False)
    lab = samp[samp["stance"].astype(str).str.len() > 0].copy()
    lab["about"] = lab["is_about_movement"].astype(str).str.lower().isin(["true", "1"])

    # ── report ──
    out = []
    def w(s=""):
        print(s)
        out.append(s)

    w("=" * 68)
    w("DOES THE GENERIC-VOCABULARY MATERIAL BELONG IN THE STUDY?")
    w("=" * 68)
    w(f"labeled: {len(lab):,} of {len(samp):,} sampled "
      f"({samp['llm_error'].astype(str).str.len().gt(0).sum()} errors)")
    w("")
    w("Share of articles that are ACTUALLY ABOUT THE MOVEMENT:")
    w("")
    w(f"  {'stratum':28s} {'n':>5s} {'about':>6s} {'rate':>7s}   95% CI")
    for mt in ["hashtag", "generic"]:
        sub = lab[lab["match_type"] == mt]
        if not len(sub):
            continue
        k, n = sub["about"].sum(), len(sub)
        lo, hi = wilson(k, n)
        w(f"  {'ALL ' + mt.upper():28s} {n:5d} {k:6d} {k/n*100:6.1f}%   "
          f"[{lo*100:.1f}, {hi*100:.1f}]")
        for lg in sorted(sub["lang_group"].unique()):
            s2 = sub[sub["lang_group"] == lg]
            k2, n2 = s2["about"].sum(), len(s2)
            if n2 < 15:
                continue
            lo2, hi2 = wilson(k2, n2)
            w(f"    {mt + ' / ' + lg:26s} {n2:5d} {k2:6d} {k2/n2*100:6.1f}%   "
              f"[{lo2*100:.1f}, {hi2*100:.1f}]")
        w("")

    kh = lab[lab["match_type"] == "hashtag"]["about"]
    kg = lab[lab["match_type"] == "generic"]["about"]
    if len(kh) and len(kg):
        gap = (kh.mean() - kg.mean()) * 100
        w(f"GAP: hashtag-matched are {gap:+.1f} points more likely to be about")
        w("     the movement than generic-matched.")
        w("")
        w("HOW TO READ THIS")
        w("  generic rate above ~70%  -> the material belongs; corpus stands.")
        w("  generic rate 40-70%      -> mixed; filter on is_about_movement and")
        w("                              report both populations.")
        w("  generic rate below ~40%  -> it is a different population; either")
        w("                              split the study or restrict the corpus.")
        w("")
        est = kg.mean()
        pool = 50668   # generic-matched articles successfully fetched
        w(f"  At the observed generic rate, roughly {int(pool*est):,} of the")
        w(f"  {pool:,} fetched generic-matched articles are movement coverage,")
        w(f"  and about {int(pool*(1-est)):,} are not.")

    w("")
    w("=" * 68)
    w("STANCE, AMONG ARTICLES ACTUALLY ABOUT THE MOVEMENT")
    w("=" * 68)
    am = lab[lab["about"]]
    if len(am):
        vc = am["stance"].value_counts()
        for k, v in vc.items():
            w(f"  {k:16s} {v:5d}  {v/len(am)*100:5.1f}%")
        w("")
        w("  by language group:")
        ct = pd.crosstab(am["lang_group"], am["stance"], normalize="index") * 100
        w(ct.round(1).to_string())

    w("")
    w("=" * 68)
    w("KEYWORD-BASED STANCE vs MODEL STANCE  (how wrong was the proxy?)")
    w("=" * 68)
    if "article_stance" in samp.columns:
        merged = lab.merge(
            pd.read_csv(a.infile, usecols=["url", "article_stance"],
                        low_memory=False), on="url", how="left")
        w(pd.crosstab(merged["article_stance"], merged["stance"]).to_string())

    w("")
    w("SPOT-CHECK THESE BEFORE TRUSTING ANY OF IT:")
    w("  open sample_labeled.csv and read `movement_reasoning` for 10 rows where")
    w("  is_about_movement is False and match_type is generic. If the model is")
    w("  calling real movement coverage 'not about the movement', the whole")
    w("  table above is wrong and you should say so rather than present it.")

    with open(OUT_TXT, "w") as fh:
        fh.write("\n".join(out))
    print(f"\nwrote {OUT_CSV} and {OUT_TXT}")


if __name__ == "__main__":
    main()
