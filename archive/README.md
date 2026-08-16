# archive/

Superseded files, kept because they document decisions rather than because they
are useful. **Nothing here has been path-patched for the reorganised layout — do
not run these scripts as-is.**

| File | Why it is here |
|---|---|
| `10_sample_and_label.py` | The threaded LLM labeller. It blocked on network calls that never returned: the SDK timeout sat inside a `try/except` that silently fell back to a client with no timeout, and four threads shared one HTTP connection pool. Rewritten sequentially as `scripts/13_label_simple.py`. Kept as the record of a real debugging finding. |
| `add_country.py` | One-time patch that added `source_country` to `articles_v3_enriched.csv` after the matcher omitted it. Already applied; the logic now lives inside `scripts/02_local_match_v3.py`. |
| `metoo_keywords_bq.csv` | The v2 codebook, superseded by `codebook/metoo_keywords_v3.csv`. Retained because `scripts/01_bigquery_corpus.sql` was written against it. |
| `superseded_figures/` | The twelve `lda_by_country_*` and `lda_over_time_*` charts labelled "Topic 1…8", superseded by the `named_*` charts in `outputs/figures/`. |
