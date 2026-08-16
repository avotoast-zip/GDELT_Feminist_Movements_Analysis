# Folder audit — "Resampling - GDELT"

**120 files, 7.6 GB, entirely flat, no git, no README, no requirements file.**
Audited 16 Aug 2026. Nothing has been moved or deleted yet.

---

## 1. What this project is

A computational study of global #MeToo media coverage, Oct 2017 – Dec 2019, built on
GDELT via BigQuery, for FADS under Profs. Dessauer and Shinde.

The v3 corpus is the live one: **212,957 articles, 190 countries, 11,744 outlets**,
matched against a ~1,700-term multilingual codebook. Matching was moved off BigQuery
to a local Python script because the sandbox caps CPU per byte scanned.

Where the numbers stand:

| Stage | Count |
|---|---|
| Articles collected (v3 corpus) | 212,957 |
| Links attempted after pre-fetch dedup | 205,528 |
| Downloaded with usable text | 107,481 |
| Unique after text-based dedup (80% cutoff) | 76,925 |
| Used for topic modelling | 80,923 |

---

## 2. The pipeline, reconstructed from the code

Every script hardcodes bare filenames and expects to run from this folder. That is the
single constraint that governs how the folder can be restructured.

```
metoo_keywords_v3.csv  (codebook)
metoo_keyword_corpus.sql ──► BigQuery ──► cand_2017q4 … cand_2019q4.csv  (9 files, 266 MB)
        │
        ▼
local_match_v3.py ──► keyword_hits_v3.csv (6.7 GB, one row per article×keyword)
                 └──► articles_v3.csv (46 MB, one row per article)
        │
        ▼
03_enrich_v3.py ──► articles_v3_enriched.csv (53 MB)
        │              ▲
        │              └── add_country.py (one-time patch, already applied)
        ├──► 04_build_dashboard.py ──► metoo_dashboard_v3.html
        ├──► 09_corpus_profile.py  ──► profile_1 … profile_6.csv
        ▼
05a_dedupe_prefetch.py ──► prefetch_dedupe_map.csv, fetch_list.csv (51 MB)
        │
        ├──► 06_pilot_fetch_rate.py ──► pilot_fetch_results.csv
        │         └──► 06b_diagnose_failures.py ──► failure_diagnosis.csv
        ▼
07_fetch_and_label.py ──► labeled_articles.csv (422 MB)  ◄── THE MASTER FILE
        └── 07b_wayback_retry.py patches it in place
        │
        ├──► 08_dedupe.py ──► dedupe_map.csv, deduped_articles.csv (291 MB)
        ├──► 13_label_simple.py ──► sample_frame_simple.csv, sample_labeled.csv, sample_report.txt
        │         └──► 14_spotcheck.py (prints only)
        ├──► 15_export_samples.py ──► article_samples.csv / .html
        ├──► 16_rematch_fulltext.py ──► rematched_articles.csv, rematch_report.txt
        ├──► 17_lda_topics.py ──► lda_topics_*.csv, lda_article_topics.csv, lda_*.png, lda_report.txt
        │         └──► 21_name_topics.py ──► topic_names.csv → topic_names_final.csv, named_*.png
        ├──► 19_build_fetch_dashboard.py ──► metoo_dashboard_fetch.html
        └──► 20_events_and_propagation.py ──► event_timeline.png, propagation_*.png,
                                              propagation_summary.csv, events_report.txt

metoo_grand_dashboard.html ── no generator script in the folder (hand-built)
```

### Gaps and inconsistencies worth recording

- **Missing script numbers: 01, 02, 11, 12, 18.** 01/02 are the BigQuery steps, which
  survive only as `metoo_keyword_corpus.sql`. 11, 12 and 18 are gone — 10 → 13 is
  explained (the threaded labeler hung and was rewritten sequentially), 18 is not.
- **`08_dedupe.py` still calls itself `05_dedupe.py` in its own docstring.**
- **Two dead-end branches.** `16_rematch_fulltext.py` and `08_dedupe.py` produce
  `rematched_articles.csv` and `deduped_articles.csv`, but *nothing reads them*.
  Scripts 17, 19 and 20 all read `labeled_articles.csv` directly. So the LDA, the
  event propagation and the fetch dashboard were all computed on the **pre-dedup,
  pre-rematch** corpus. That is why topic modelling reports 80,923 articles rather
  than the 76,925 unique ones. Not necessarily wrong — but undocumented, and it needs
  to be stated plainly in the repo rather than discovered later.
- **The editorial-overlay dedup rule never ran properly** — it needs stance labels that
  do not exist yet. Only 130 articles were split on text difference alone.

---

## 3. What the project found

Four results carry the study.

**Full-text re-matching changed less than expected, and that is the finding.** Matching
on article body rather than URL and headline moved hashtag share only 52.8% → 54.7%.
France moved most, 16.0% → 22.9%, and is still 77% general vocabulary. The
hashtag/general-vocabulary split is real, not a searching artifact.

**Backlash vocabulary was nearly invisible in headlines.** "false accusations" 31 → 837,
"witch hunt" 337 → 1,106, "due process" 291 → 1,076. Articles carrying both support and
backlash language went from 71 to 2,899. This corrects the earlier claim that the two
vocabularies were effectively separate.

**American dominance was partly syndication.** Text-based dedup removed 33,940 copies.
The US fell from 24.2% to 19.3% of the corpus. iheart.com alone produced 2,598 copies of
40 distinct stories.

**Event propagation splits sharply by event, not by country.** The Deneuve letter reached
25 countries, nine of eleven measurable ones peaking within a day. Sweden's Kulturprofilen
exposé — which triggered a national reckoning — reached 2. #EleNão reached 1. The Nordic
countries' 7–45 day "lags" after the Milano tweet are their own national waves, not slow
reactions. Weinstein and Burning Sun are formally unusable and marked as such.

**LDA independently reproduces the same split.** Spanish and Portuguese topics are
dominated by individual femicide cases and court vocabulary; English and French by the
movement and named figures. The Swedish Academy appears as an isolated topic — the same
domestic-containment result the propagation analysis found by a different method.

**The open question.** A 398-article labelled sample found 90.5% of hashtag articles are
about the movement versus 17.1% of general-vocabulary ones — roughly 42,000 of 50,700
general-vocabulary articles are ordinary crime and court reporting. Deciding whether they
are in or out of the study is the unresolved methodological question. Full-corpus
labelling is blocked on ~$100–150 of API credit.

---

## 4. File inventory and verdicts

### Must not reach GitHub

| File | Why |
|---|---|
| `.env` | Contains a live `GOOGLE_API_KEY`. Never commit. Replace with `.env.example`. |
| `article_samples.csv` / `.html` | Full text of ~120 copyrighted news articles. |
| `labeled_articles.csv`, `deduped_articles.csv`, `rematched_articles.csv` | Full article text at scale. |
| `.DS_Store` | macOS junk. |

### Too large for GitHub regardless (GitHub blocks >100 MB per file)

| File | Size | Regenerable? |
|---|---|---|
| `keyword_hits_v3.csv` | **6.7 GB** | Yes, from `local_match_v3.py` |
| `labeled_articles.csv` | 422 MB | No — 6+ hours of fetching, irreplaceable |
| `deduped_articles.csv` | 291 MB | Yes, from `08_dedupe.py` |

### Large but under the cap — still belongs in data/, not git

`articles_v3_enriched.csv` (53 MB), `cand_*.csv` (266 MB across 9 files),
`fetch_list.csv` (51 MB), `keyword_hits.csv` (48 MB), `articles_v3.csv` (46 MB),
`prefetch_dedupe_map.csv` (36 MB), `rematched_articles.csv` (34 MB),
`dedupe_map.csv` (14 MB), `lda_article_topics*.csv` (23 MB).

### Superseded or dead

| File | Status |
|---|---|
| `keyword_hits.csv` (48 MB) | v2 output. Referenced by nothing. Superseded by v3. |
| `metoo_keywords_bq.csv` | v2 codebook, superseded by `metoo_keywords_v3.csv`. Keep only as provenance for the SQL. |
| `10_sample_and_label.py` | The threaded labeler that hung. Superseded by `13_label_simple.py`. Worth archiving, not deleting — it documents why. |
| `add_country.py` | One-time patch, already applied. Archive. |
| `lda_by_country_*.png`, `lda_over_time_*.png` (12 files) | "Topic 1…8" versions, superseded by the `named_*` set. |
| `api_test.py` | Trivial, but harmless and documents API setup. |

### Keep and publish — these are the deliverables

Dashboards (`metoo_grand_dashboard.html`, `metoo_dashboard_v3.html`,
`metoo_dashboard_fetch.html`), all `named_*.png` and `lda_wordcloud_*.png`,
`event_timeline.png`, 11 `propagation_*.png`, the four `.txt` reports,
`profile_1…6.csv`, `propagation_summary.csv`, `lda_topics_*.csv`,
`topic_names.csv` / `topic_names_final.csv`, `dedupe_thresholds.json`,
`update_summary.md`, the codebook `metoo_keywords_v3.csv`, `sourcesbycountry.csv`,
the map geometry (`world_paths.json`, `microstate_dots.json`), and all 22 scripts.

---

## 5. The one structural constraint

Every script uses bare relative filenames (`IN_FILE = "articles_v3.csv"`) and assumes the
working directory is this folder. Moving scripts into `src/` and data into `data/` breaks
all 22 of them unless the path constants are patched at the same time.

That is a decision, not a detail: a proper repo layout requires editing every script's
path constants, which is safe but touches every file.

---

## 6. Proposed structure

```
metoo-gdelt/
├── README.md                  ← project, pipeline, findings, how to run
├── requirements.txt
├── .gitignore                 ← data/, .env, *.csv over size, .DS_Store
├── .env.example
├── docs/
│   ├── pipeline.md            ← the DAG above
│   ├── findings.md            ← from update_summary.md
│   ├── known_issues.md        ← dead-end branches, outlet collapsing, stale pages
│   └── methodology_notes.md   ← dedup rules, sampling, the Weinstein caveat
├── scripts/                   ← all 22, numbers preserved
├── codebook/                  ← metoo_keywords_v3.csv, sourcesbycountry.csv, topic_names.csv
├── outputs/
│   ├── dashboards/  figures/  reports/  tables/
├── data/                      ← gitignored, stays on your disk
│   ├── raw/  interim/  processed/
└── archive/                   ← superseded scripts, kept for provenance
```
