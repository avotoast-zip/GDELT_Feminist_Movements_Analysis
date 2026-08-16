#!/usr/bin/env python3
"""
20_events_and_propagation.py
----------------------------
Two things.

ONE. An event timeline. Daily article counts across the whole corpus with the
known events marked.

TWO. A propagation analysis. For each event, this measures how long coverage
took to reach each country, and whether it reached them at all.

THE QUESTION IT ANSWERS

The Kavanaugh hearing happened in the United States on one day. American
coverage spikes that day. Does coverage in France, India or Brazil spike on the
same day, a few days later, or never?

If there is a lag, we can ask what predicts it. Distance? Shared language?
Political alliance? This script produces the lag numbers. It does not claim to
explain them.

HOW AN ARTICLE IS LINKED TO AN EVENT

Each event has its own keyword set, searched in the article text. An article
counts as covering that event if it contains one of those words and was
published in the window around the event date.

This is deliberately narrow. "Kavanaugh" is a very specific word. That makes
false positives rare and means a country's count is a real signal.

HOW LAG IS MEASURED

For each country we take its daily count of articles about the event, then
report three things:

  first_day   the first day the country published anything about it
  peak_day    the day the country published the most about it
  lag         peak_day minus the event date, in days

Counts are also shown as a share of that country's total output that week, so a
small country with 20 articles is comparable to the United States with 4,000.

RUN
    python 20_events_and_propagation.py
    python 20_events_and_propagation.py --event kavanaugh
    python 20_events_and_propagation.py --window 45 --min-articles 5

OUTPUT
    event_timeline.png            whole corpus, daily, events marked
    propagation_<event>.png       heatmap, countries by days since the event
    propagation_summary.csv       every event, every country, its lag
    events_report.txt             the numbers in text
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ARTICLES = PROCESSED / "labeled_articles.csv"

# ── The events. Edit this list freely. ──
# origin  = the country where the event happened
# words   = searched in the article text, lowercase, no accents needed
# Keep the words specific. A vague word will pull in unrelated coverage.
EVENTS = [
    # ─────────────────────────────────────────────────────────────────────
    # IMPORTANT: our corpus is built from #MeToo-era keywords. The hashtag was
    # created on 15 October 2017. Coverage from BEFORE that date rarely contains
    # any of our keywords, so it was never collected.
    #
    # Daily article counts prove it:  14 Oct = 50 articles, 15 Oct = 95,
    # 16 Oct = 360, 17 Oct = 528. The world did not start writing on the 16th.
    # Our collection did.
    #
    # Any event before 15 Oct 2017 therefore cannot be used to measure how fast
    # news travelled. Those events carry blind=True and are excluded from the
    # lag statistics.
    # ─────────────────────────────────────────────────────────────────────

    dict(key="weinstein", date="2017-10-05", origin="United States",
         label="NYT publishes Weinstein story",
         words=["weinstein"], blind=True,
         note="Happened 10 days before the #MeToo hashtag existed. Our corpus "
              "cannot see the first 10 days of this story, so every country "
              "appears to peak around day +12. That is when our collection "
              "starts, not when the world reacted. Do not read a lag here."),

    dict(key="metoo_tweet", date="2017-10-15", origin="United States",
         label="Alyssa Milano tweet starts #MeToo",
         words=["metoo", "me too movement", "alyssa milano"], blind=False,
         note="This is the first event our corpus can see properly, because "
              "the hashtag it searches for was created on this day."),

    # The Nordic countries peaked 36 to 45 days after the Milano tweet. That is
    # not a delayed reaction to an American tweet. It is their own national
    # wave, with its own trigger and its own hashtags. Treated as its own event.
    dict(key="nordic_wave", date="2017-11-21", origin="Sweden",
         label="Swedish wave, Kulturprofilen expose",
         words=["kulturprofil", "arnault", "tystnadtagning", "visjungerut",
                "svenska akademien"], blind=False,
         note="Svenska Dagbladet published the Kulturprofilen allegations on "
              "21 November 2017. Swedish coverage peaks on 21 and 22 November. "
              "Swedish hashtags such as #tystnadtagning and #visjungerut belong "
              "to this wave, not to the American one."),

    # Found while testing: a global peak 16 days after the Swedish wave, on
    # 6 December 2017. That is Time naming "The Silence Breakers" Person of the
    # Year. It is a global media event with no single origin country.
    dict(key="time_poty", date="2017-12-06", origin=None,
         label="Time Person of the Year, The Silence Breakers",
         words=["silence breakers", "person of the year", "persona del ano",
                "personnalite de l annee"], blind=False,
         note="A global magazine cover, not a national event. No origin "
              "country, so no lag is measured. Included because it produces a "
              "worldwide simultaneous spike."),

    dict(key="deneuve", date="2018-01-09", origin="France",
         label="Deneuve open letter in Le Monde",
         words=["deneuve", "droit d importuner", "importuner"], blind=False,
         note="The clearest case of near-instant spread in this corpus."),

    dict(key="la_manada", date="2018-04-26", origin="Spain",
         label="La Manada verdict",
         words=["manada"], blind=False,
         note="Only 21 Spanish articles with text survive from this window, so "
              "the picture is thin. Spanish coverage of 'manada' in our corpus "
              "is dominated by the 2019 Manresa case instead."),

    dict(key="elenao", date="2018-09-29", origin="Brazil",
         label="#EleNao protests",
         words=["elenao", "ele nao", "elenão"], blind=False,
         note="Only 19 Brazilian articles. Brazil entered our corpus mostly "
              "through the word 'feminicidio', not through its hashtags, so "
              "this movement is badly undercounted."),

    dict(key="kavanaugh", date="2018-09-27", origin="United States",
         label="Kavanaugh and Ford hearing",
         words=["kavanaugh", "blasey"], blind=False,
         note="The Senate confirmed Kavanaugh on 6 October, which is day +9. "
              "Countries peaking at +7 to +9 are probably covering the "
              "confirmation vote, not arriving late to the hearing."),

    # India's wave began with the Tanushree Dutta interview in late September,
    # but daily volume shows the real surge starting 11 October: 255 articles on
    # the 11th, 353 on the 12th. Dated to the surge, not to the first allegation.
    dict(key="india_metoo", date="2018-10-11", origin="India",
         label="India #MeToo surge, Akbar and Ramani",
         words=["akbar", "tanushree", "ramani", "nana patekar"], blind=False,
         note="Indian daily volume: 255 articles on 11 Oct, 353 on 12 Oct. "
              "MJ Akbar resigned on 17 October. Earlier allegations began in "
              "late September, so this date marks the surge, not the start."),

    dict(key="burning_sun", date="2019-03-11", origin="South Korea",
         label="Burning Sun scandal",
         words=["burning sun", "seungri", "jung joon"], blind=True,
         note="CANNOT BE STUDIED with this corpus. Only one South Korean "
              "article with text falls in the whole of 2019. Four articles "
              "worldwide mention Burning Sun. Kept in the list so the gap is "
              "visible instead of silently missing."),

    dict(key="manresa", date="2019-11-19", origin="Spain",
         label="Manresa verdict, second 'manada' case",
         words=["manada", "manresa"], blind=False,
         note="Spanish coverage peaks 18 to 20 November 2019. This drives much "
              "of the November 2019 spike, alongside the UN day on the 25th."),

    dict(key="un_day_2018", date="2018-11-25", origin=None,
         label="UN day on violence against women 2018",
         words=["violencia de genero", "violence against women",
                "violences faites aux femmes", "25 de noviembre"], blind=False,
         note="An annual observance, not a news event. There is no origin "
              "country, so no lag can be measured. Included to show which "
              "countries mark the day."),
]


def _wrap(t, w):
    import textwrap
    return textwrap.wrap(t, w) or [""]


def norm(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^\w\s#]", " ", s)


def load():
    cols = ["url", "source_country", "published", "article_text",
            "fetch_status", "text_chars"]
    parts = []
    for ch in pd.read_csv(ARTICLES, usecols=cols, chunksize=100_000, low_memory=False):
        ch = ch[ch["fetch_status"].isin(["OK", "OK_WAYBACK"])]
        ch = ch[ch["article_text"].notna()]
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    df["dt"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    df = df[df["dt"].notna()]
    df["day"] = df["dt"].dt.tz_localize(None).dt.normalize()
    return df.reset_index(drop=True)


def timeline_chart(df):
    daily = df.groupby("day").size()
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.fill_between(daily.index, daily.values, color="#3E6B73", alpha=.55, linewidth=0)
    ax.plot(daily.index, daily.values, color="#2A4F55", linewidth=.8)

    ymax = daily.max()
    for i, e in enumerate(EVENTS):
        d = pd.Timestamp(e["date"])
        if d < daily.index.min() or d > daily.index.max():
            continue
        ax.axvline(d, color="#B5432A", linewidth=1, alpha=.75, linestyle="--")
        y = ymax * (0.97 - 0.075 * (i % 5))
        ax.annotate(e["label"], xy=(d, y), fontsize=7.5, rotation=0,
                    ha="left", va="top", color="#7A1420",
                    xytext=(4, 0), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.22", fc="#FFF3F0",
                              ec="#E0C4BD", lw=.5))
    ax.set_ylabel("articles per day")
    ax.set_title("Coverage over time, with known events",
                 fontsize=13, fontweight="bold", loc="left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.grid(alpha=.22)
    ax.set_xlim(daily.index.min(), daily.index.max())
    fig.tight_layout()
    fig.savefig(FIGURES / "event_timeline.png", dpi=135)
    plt.close(fig)
    print("wrote event_timeline.png")


def event_subset(df, ev, window):
    d0 = pd.Timestamp(ev["date"])
    lo, hi = d0 - pd.Timedelta(days=7), d0 + pd.Timedelta(days=window)
    win = df[(df["day"] >= lo) & (df["day"] <= hi)].copy()
    if not len(win):
        return None, d0
    pat = "|".join(re.escape(w) for w in ev["words"])
    hit = win["article_text"].astype(str).str.slice(0, 6000).apply(norm) \
        .str.contains(pat, regex=True, na=False)
    return win[hit].copy(), d0


def propagation(df, ev, window, min_articles):
    sub, d0 = event_subset(df, ev, window)
    if sub is None or not len(sub):
        return None, None
    sub["offset"] = (sub["day"] - d0).dt.days

    rows = []
    for c, g in sub.groupby("source_country"):
        if c == "UNKNOWN" or len(g) < min_articles:
            continue
        daily = g.groupby("offset").size()
        # A country with 6 articles has a "peak" of 2 on a random day. That is
        # noise, not a peak. Smooth over 3 days before taking the maximum, and
        # mark anything under 15 articles as unreliable.
        full = daily.reindex(range(-7, window + 1), fill_value=0)
        sm = full.rolling(3, center=True, min_periods=1).mean()
        peak = int(sm.idxmax())
        thresh = max(2, daily.max() * 0.2)
        onset_days = daily[daily >= thresh].index
        onset = int(min(onset_days)) if len(onset_days) else peak
        rows.append(dict(event=ev["key"], country=c, articles=int(len(g)),
                         onset_day=onset, peak_day=peak,
                         peak_articles=int(daily.max()),
                         reliable=bool(len(g) >= 15)))
    if not rows:
        return None, None
    res = pd.DataFrame(rows).sort_values(
        ["reliable", "peak_day", "onset_day"], ascending=[False, True, True])

    # heatmap matrix: country x offset, each row scaled to its own max
    countries = res["country"].tolist()
    offs = list(range(-7, window + 1))
    M = np.zeros((len(countries), len(offs)))
    for i, c in enumerate(countries):
        daily = sub[sub["source_country"] == c].groupby("offset").size()
        for j, o in enumerate(offs):
            M[i, j] = daily.get(o, 0)
        if M[i].max() > 0:
            M[i] = M[i] / M[i].max()
    return res, (countries, offs, M)


def prop_chart(ev, res, mat, window):
    countries, offs, M = mat
    h = max(3.0, 0.30 * len(countries))
    fig, ax = plt.subplots(figsize=(11.5, h))
    im = ax.imshow(M, aspect="auto", cmap="rocket_r" if False else "magma_r",
                   extent=[offs[0] - .5, offs[-1] + .5, len(countries) - .5, -.5],
                   interpolation="nearest")
    ax.set_yticks(range(len(countries)))
    lab = []
    for c in countries:
        r = res[res["country"] == c].iloc[0]
        star = " *" if (ev["origin"] and c == ev["origin"]) else ""
        mark = "" if r["reliable"] else "  [low n]"
        lab.append(f"{c}{star}  (n={r['articles']}, peak {r['peak_day']:+d}d){mark}")
    ax.set_yticklabels(lab, fontsize=8)
    ax.axvline(0, color="#B5432A", linewidth=1.6)
    ax.set_xlabel("days since the event")
    title = (f"{ev['label']}   ({ev['date']}"
             + (f", origin {ev['origin']}" if ev["origin"] else "") + ")")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left",
                 color=("#B5432A" if ev.get("blind") else "#17171E"))
    if ev.get("blind"):
        ax.text(0.0, 1.02, "CORPUS CANNOT SEE THIS EVENT PROPERLY - "
                           "do not read a lag from this chart",
                transform=ax.transAxes, fontsize=8.5, color="#B5432A",
                va="bottom", ha="left", style="italic")
    fig.colorbar(im, ax=ax, shrink=.75, label="share of that country's peak day")
    fig.tight_layout()
    fig.savefig(FIGURES / f"propagation_{ev['key']}.png", dpi=135)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=45,
                    help="days after the event to follow")
    ap.add_argument("--min-articles", type=int, default=6,
                    help="minimum articles for a country to appear")
    ap.add_argument("--event", default=None, help="run one event by key")
    a = ap.parse_args()

    print("loading (about a minute)...")
    df = load()
    print(f"  {len(df):,} articles with text and a date")

    timeline_chart(df)

    L = ["EVENT PROPAGATION", "=" * 70, ""]
    allres = []
    events = [e for e in EVENTS if (a.event is None or e["key"] == a.event)]
    for ev in events:
        print(f"\n{ev['key']}...")
        res, mat = propagation(df, ev, a.window, a.min_articles)
        if res is None:
            print("  no coverage found")
            L += [f"{ev['label']}: no coverage found", ""]
            continue
        prop_chart(ev, res, mat, a.window)
        allres.append(res)

        origin_peak = None
        if ev["origin"] is not None and (res["country"] == ev["origin"]).any():
            origin_peak = int(res[res["country"] == ev["origin"]]["peak_day"].iloc[0])

        L.append(f"--- {ev['label']}  ({ev['date']})")
        L.append(f"    origin: {ev['origin'] or 'none, annual observance'}")
        if ev.get("note") and not ev.get("blind"):
            for ln in _wrap(ev["note"], 66):
                L.append(f"    note: {ln}" if ln == _wrap(ev["note"], 66)[0]
                         else f"          {ln}")
        L.append(f"    countries covering it: {len(res)}   "
                 f"articles: {int(res['articles'].sum()):,}")
        rel = res[res["reliable"]]
        L.append(f"    countries with enough articles to trust (n>=15): {len(rel)}")
        if ev.get("blind"):
            L.append("    *** LAG NOT MEASURABLE FOR THIS EVENT ***")
            L.append("    " + ev.get("note", ""))
            L.append("")
            continue
        if origin_peak is not None:
            L.append(f"    origin country peaked at day {origin_peak:+d}")
            others = rel[rel["country"] != ev["origin"]]
            if len(others):
                lag = others["peak_day"] - origin_peak
                L.append(f"    other countries peaked a median of "
                         f"{lag.median():+.0f} days after the origin")
                L.append(f"    same day or next day : "
                         f"{int((lag <= 1).sum())} of {len(others)}")
                L.append(f"    2 to 6 days later    : "
                         f"{int(((lag >= 2) & (lag <= 6)).sum())}")
                L.append(f"    a week or more later : {int((lag >= 7).sum())}")
        if len(rel):
            L.append("    order of peaks: " + ", ".join(
                f"{r.country} {r.peak_day:+d}d" for r in rel.itertuples()))
        L.append("")
        print(f"  {len(res)} countries, {int(res['articles'].sum()):,} articles")

    if allres:
        out = pd.concat(allres, ignore_index=True)
        out.to_csv(TABLES / "propagation_summary.csv", index=False)
        print(f"\nwrote propagation_summary.csv ({len(out)} rows)")

        L += ["", "=" * 70, "REACH: how many countries covered each event", "=" * 70]
        for ev in events:
            r = out[out["event"] == ev["key"]]
            if len(r):
                L.append(f"  {ev['label'][:44]:44s} {len(r):3d} countries")

    L += ["", "=" * 70, "READ THIS BEFORE USING THE NUMBERS", "=" * 70,
          "",
          "1. A country marked [low n] has too few articles for its peak day to",
          "   mean anything. With 6 articles the peak is wherever 2 of them",
          "   happened to land. Only countries with 15 or more are used in the",
          "   lag statistics above.",
          "",
          "2. A late peak may be a different event, not a delayed reaction.",
          "   Kavanaugh is the clearest case. The hearing was 27 September. The",
          "   Senate confirmed him on 6 October, which is day +9. Many countries",
          "   peak at +7 to +9. That is most likely them covering the",
          "   confirmation, not arriving late to the hearing.",
          "   Before calling something a lag, check what else happened that week.",
          "",
          "3. Download success varied by country, from 26% to 80%. A country",
          "   with poor retrieval will look quieter than it was.",
          "",
          "4. Absence of coverage here means absence in our corpus. It does not",
          "   prove the country ignored the event.",
          ""]
    open(REPORTS / "events_report.txt", "w").write("\n".join(L))
    print("wrote events_report.txt")
    print("\nRead events_report.txt first. The heatmaps show the shape,")
    print("the report gives the lag numbers.")


if __name__ == "__main__":
    main()
