# Pipeline

Every script resolves its paths from the repository root via a header block, so
they can be run from any working directory:

```bash
python scripts/09_corpus_profile.py
```

## Flow

```
codebook/metoo_keywords_v3.csv
        │
scripts/01_bigquery_corpus.sql ──► BigQuery ──► data/raw/cand_2017q4 … cand_2019q4.csv
        │
        ▼
scripts/02_local_match_v3.py ──► data/interim/keyword_hits_v3.csv   (6.7 GB, article × keyword)
                             └──► data/interim/articles_v3.csv       (one row per article)
        │
        ▼
scripts/03_enrich_v3.py ──► data/interim/articles_v3_enriched.csv
        │
        ├──► scripts/04_build_dashboard.py ──► outputs/dashboards/metoo_dashboard_v3.html
        ├──► scripts/09_corpus_profile.py   ──► outputs/tables/profile_1 … profile_6.csv
        ▼
scripts/05a_dedupe_prefetch.py ──► data/interim/prefetch_dedupe_map.csv
                               └──► data/interim/fetch_list.csv
        │
        ├──► scripts/06_pilot_fetch_rate.py  ──► outputs/tables/pilot_fetch_results.csv
        │         └──► scripts/06b_diagnose_failures.py ──► outputs/tables/failure_diagnosis.csv
        ▼
scripts/07_fetch_and_label.py ──► data/processed/labeled_articles.csv   ◄── THE MASTER FILE
        └── scripts/07b_wayback_retry.py patches it in place
        │
        ├──► scripts/08_dedupe.py ──► data/interim/dedupe_map.csv
        │                         └──► data/processed/deduped_articles.csv
        ├──► scripts/13_label_simple.py ──► data/processed/sample_labeled.csv
        │                              └──► outputs/reports/sample_report.txt
        │         └──► scripts/14_spotcheck.py (prints to screen)
        ├──► scripts/15_export_samples.py ──► data/processed/article_samples.csv / .html
        ├──► scripts/16_rematch_fulltext.py ──► data/processed/rematched_articles.csv
        │                                   └──► outputs/reports/rematch_report.txt
        ├──► scripts/17_lda_topics.py ──► outputs/tables/lda_topics_<lang>.csv
        │                             ├──► data/processed/lda_article_topics.csv
        │                             ├──► outputs/figures/lda_*.png
        │                             └──► outputs/reports/lda_report.txt
        │         └──► scripts/21_name_topics.py ──► codebook/topic_names.csv (you edit this)
        │                                       ├──► outputs/tables/topic_names_final.csv
        │                                       └──► outputs/figures/named_*.png
        ├──► scripts/22_phrases.py ──► outputs/tables/ngrams_<lang>.csv
        │                          ├──► outputs/tables/collocations_<lang>.csv
        │                          ├──► outputs/tables/kwic_<lang>.csv
        │                          ├──► outputs/tables/repeated_sentences_<lang>.csv
        │                          └──► outputs/reports/phrases_report.txt
        ├──► scripts/19_build_fetch_dashboard.py ──► outputs/dashboards/metoo_dashboard_fetch.html
        └──► scripts/20_events_and_propagation.py ──► outputs/figures/event_timeline.png
                                                  ├──► outputs/figures/propagation_*.png
                                                  ├──► outputs/tables/propagation_summary.csv
                                                  └──► outputs/reports/events_report.txt
```

`scripts/22_phrases.py` is the one script that reads `deduped_articles.csv` by
preference rather than `labeled_articles.csv`, because phrase counts are far more
sensitive to syndicated duplicates than topic models are.

`outputs/dashboards/metoo_grand_dashboard.html` has no generator script — it was
hand-assembled from the outputs of the scripts above.

## What each script does

| Script | Purpose |
|---|---|
| `01_bigquery_corpus.sql` | Scans 27 months of the GDELT GKG table down to candidate articles. ~5 hours. Partition filters are required — the project runs in BigQuery Sandbox mode. |
| `02_local_match_v3.py` | Matches the ~1,700-term codebook against candidates locally. Runs on a laptop because the sandbox caps CPU per byte scanned, a ratio that splitting the job cannot beat. Prints regression checks 4a–4e. |
| `03_enrich_v3.py` | Derives `article_stance`, `article_type`, distinct-keyword counts. |
| `04_build_dashboard.py` | Corpus dashboard: choropleth, stance timeline, keyword and outlet bars. Self-contained HTML, no CDN. |
| `05a_dedupe_prefetch.py` | Cheap pre-fetch deduplication on URL variants and identical titles. Collapses certain duplicates; flags wire syndication rather than collapsing it. |
| `06_pilot_fetch_rate.py` | Fetches 1,000 random URLs to measure link rot before committing to the full run. |
| `06b_diagnose_failures.py` | Separates genuinely dead links from scraper blocking — the distinction that decides whether a country's silence is real. |
| `07_fetch_and_label.py` | Parallel fetch of all URLs; optional Gemini labelling. Article text and model reasoning are stored with every label; rows without text are never sent to the API. |
| `07b_wayback_retry.py` | Patient second pass at the Wayback CDX index for rows archive.org refused rather than answered. |
| `08_dedupe.py` | MinHash/LSH near-duplicate detection over article body shingles, with an editorial-overlay guard. |
| `09_corpus_profile.py` | The six descriptive tables requested by the supervisors: hashtag vs generic by country, match classes, language, outlets and concentration, top keywords, suspect TLD assignments. |
| `13_label_simple.py` | Sequential LLM labeller. Replaced the threaded version, which hung on network calls. Checkpoints every 20 calls and resumes. |
| `14_spotcheck.py` | Prints the model's own reasoning for the calls that decide the composition question. |
| `15_export_samples.py` | Readable article samples for supervisors, spread across countries and match types. |
| `16_rematch_fulltext.py` | Re-runs the codebook against downloaded article bodies rather than URL and headline. |
| `17_lda_topics.py` | LDA, one model per language — a pooled model would map languages, not themes. Repairs mojibake, filters site furniture and stale pages. |
| `19_build_fetch_dashboard.py` | Dashboard for the fetch and deduplication stages. |
| `20_events_and_propagation.py` | Event timeline and per-country coverage lag for eleven events. |
| `21_name_topics.py` | Two-step human naming of LDA topics, then re-renders every chart with the names. |
| `22_phrases.py` | Common phrases rather than common words: n-grams with a syndication guard, log-Dice collocations for chosen node words, KWIC concordance lines, and repeated whole sentences — the quotations that travelled between outlets and countries. Answers what a generic topic word like *people* actually means in context. |
| `_textutils.py` | Shared stopword, boilerplate and accented-form lists plus language detection, imported by both `17` and `22` so the topic tables and the phrase tables filter identically. |
| `api_test.py` | Twenty-second check that the Gemini key works, before starting a long run. |
