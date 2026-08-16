# Global #MeToo Media Coverage, 2017–2019

A computational study of how #MeToo was covered by news media worldwide, built
on [GDELT](https://www.gdeltproject.org/) via BigQuery.

Research project at FADS, IU MSDS, supervised by Prof. Julia Dessauer and
Prof. Nilesh Shinde.

---

## The corpus

| Stage | Count |
|---|---|
| Articles collected (v3 corpus), Oct 2017 – Dec 2019 | **212,957** |
| Publishing countries / territories | 190 |
| Distinct outlets | 11,744 |
| Links attempted after pre-fetch deduplication | 205,528 |
| Downloaded with usable text | 107,481 |
| Unique after text-based deduplication (80% cutoff) | 76,925 |
| Used for topic modelling | 80,923 |

Matching uses a hand-built multilingual codebook of ~1,700 terms covering all
156 DEED countries plus a GLOBAL block, in native scripts, with stance, scope
and confidence columns (`codebook/metoo_keywords_v3.csv`).

---

## Headline findings

**The hashtag / general-vocabulary split is real, not a searching artifact.**
Re-matching against full article text rather than URL and headline moved
hashtag share only 52.8% → 54.7%. France moved most, 16.0% → 22.9%, and remains
77% general vocabulary.

**Backlash language lives in article bodies, not headlines.** "false accusations"
went from 31 hits to 837 once full text was searched; "witch hunt" 337 → 1,106;
"due process" 291 → 1,076. Articles carrying both support and backlash vocabulary
went from 71 to 2,899. This corrects an earlier claim that the two vocabularies
were effectively separate.

**American dominance was partly syndication.** Text-based deduplication removed
33,940 copies; the US fell from 24.2% to 19.3% of the corpus. iheart.com alone
produced 2,598 copies of 40 distinct stories.

**Event reach varies enormously, and not by country size.** The Deneuve open
letter reached 25 countries with nine of eleven measurable ones peaking within a
day. Sweden's Kulturprofilen exposé — which triggered a national reckoning —
reached 2 countries. #EleNão reached 1.

**Topic models reproduce the same split independently.** Spanish and Portuguese
topics are dominated by individual femicide cases and court vocabulary; English
and French by the movement and named figures. The Swedish Academy appears as an
isolated topic, matching the propagation result by a different method.

**The open methodological question.** A 398-article labelled sample found 90.5%
of hashtag-matched articles are about the movement, versus 17.1% of
general-vocabulary ones — roughly 42,000 of 50,700 general-vocabulary articles
are ordinary crime and court reporting. Whether they belong in the study is
unresolved. See `docs/findings.md`.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .env.example              copy to .env; never commit the real one
├── scripts/                  the pipeline, in run order
├── codebook/                 keyword codebook, country lookup, topic names
├── assets/geo/               map geometry for the dashboards
├── docs/                     pipeline, findings, known issues, progress updates
├── outputs/
│   ├── dashboards/           three self-contained HTML dashboards
│   ├── figures/              topic charts, word clouds, propagation plots
│   ├── reports/              text reports written by the scripts
│   └── tables/               profile tables, topic tables, summaries
├── data/                     GITIGNORED — 7.6 GB, contains article full text
└── archive/                  superseded scripts and figures, kept for provenance
```

## Start here

- `outputs/dashboards/metoo_grand_dashboard.html` — the combined dashboard, opens offline
- `docs/findings.md` — what the project has established
- `docs/pipeline.md` — what each script does and what it reads and writes
- `docs/known_issues.md` — the caveats that must travel with any result

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add your Gemini key
```

Scripts resolve all paths from the repository root, so run them from anywhere:

```bash
python scripts/17_lda_topics.py
```

They will not run end-to-end from a clean clone, because `data/` is not in the
repository. Stages 01–02 need the BigQuery exports; stages 05a onward need
`data/processed/labeled_articles.csv`, which represents about six hours of
fetching. See `data/README.md`.

---

## Notes on this repository

Two files were renamed during the August 2026 reorganisation:

| Before | Now |
|---|---|
| `metoo_keyword_corpus.sql` | `scripts/01_bigquery_corpus.sql` |
| `local_match_v3.py` | `scripts/02_local_match_v3.py` |

Script numbers 11, 12 and 18 do not exist. 10 → 13 is explained in
`docs/known_issues.md`; 18 was never recovered.

`data/` and `.env` are excluded from version control. If the API key in `.env`
was ever shared or committed anywhere, rotate it.
