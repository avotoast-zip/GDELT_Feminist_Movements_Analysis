# Known issues and caveats

These must travel with any result taken from this repository.

## Corpus construction

**The corpus cannot see the first ten days of the Weinstein story.** The
codebook is built on #MeToo terms and the hashtag was created on 15 October
2017; the NYT story broke on 5 October. Daily counts run 50 articles on 14
October, 95 on the 15th, 360 on the 16th, 528 on the 17th. Every country
therefore appears to peak around day +12 for Weinstein. That is the collection's
starting line, not a delay. The chart is marked unusable.

**Burning Sun cannot be studied.** The corpus contains one South Korean article
with text in the whole of 2019.

**Brazil entered the corpus mostly through `feminicidio`, not its hashtags**, so
#EleNão and the Brazilian movement are badly undercounted.

**Outlet names are collapsed by domain.** `g1.globo.com`, `oglobo.globo.com` and
`extra.globo.com` all appear as `globo.com`. Outlet counts and concentration
figures are affected.

**Two earlier v2 defects, fixed at source in v3.** A `generic` codebook artifact
matched 17,352 generic-drug articles, 85% of Poland's volume. And GDELT stores
titles HTML-entity-encoded, so `violência` arrives as `viol&#xEA;ncia`, where the
`&` creates a word boundary that defeated the `\bviol\b` guard — about 6,000
false positives. Regression checks 4a–4d in `scripts/02_local_match_v3.py` assert
both are gone.

## Fetching

**7% of downloaded pages served current content instead of the original
article.** 7,485 such articles are excluded from topic modelling; they are still
present in `labeled_articles.csv`.

**Some article bodies arrive ROT47-encoded.** US local-news platforms built on
the Lee Enterprises / BLOX stack serve the lead paragraph as plain text and the
rest of the body ROT47-encoded as a scraper deterrent: `kAmp?5 r2C=D@?` is
`<p>And Carlson`. These articles pass every length and quality filter and then
contribute nothing but noise. Roughly 0.6% of the corpus in a 4,000-article
check, concentrated in US local outlets — which matters for a study comparing
countries, because it thins exactly the American local coverage. Found in August
2026 when repeated-sentence detection surfaced encoded strings as "quotations".
`_textutils.repair_rot47()` now decodes them, and both `17_lda_topics.py` and
`22_phrases.py` call it. On the check sample it repaired 23 of 23 affected
articles with no false positives across 3,977 unaffected ones.

**Download success varies by country, from 26% to 80%.** A country with poor
retrieval looks quieter than it was. Absence of coverage in this corpus is not
evidence that a country ignored an event.

## Two dead-end branches

`16_rematch_fulltext.py` writes `rematched_articles.csv` and `08_dedupe.py`
writes `deduped_articles.csv`, but **nothing downstream reads either file**.
Scripts 17, 19 and 20 all read `labeled_articles.csv` directly.

Consequences:

- Topic modelling, event propagation and the fetch dashboard were computed on
  the **pre-deduplication, pre-rematch** corpus. This is why LDA reports 80,923
  articles rather than the 76,925 unique ones.
- Any result stated in terms of "unique articles" and any result from scripts
  17/19/20 are counting different populations.

This is not necessarily wrong — deduplication removes syndicated copies that
carry real topical signal — but it is a choice that was never made explicitly.
Re-running 17 and 20 against `deduped_articles.csv` would settle how much the
duplicates move the results.

**The editorial-overlay deduplication rule never ran properly.** It requires
stance labels that do not exist yet, so only 130 articles were split out on text
difference alone. It should be re-run after full labelling.

## Labelling

**Full-corpus LLM labelling is not done.** It is blocked on roughly $100–150 of
API credit. Everything stance-related currently rests on a 398-article sample.

**The free Gemini tier silently produced `API_ERROR` rows under load.** Use a
paid tier. `scripts/api_test.py` checks the key in twenty seconds before a long
run starts.

**Script numbers 11, 12 and 18 do not exist.** 10 → 13 is documented: the
threaded labeller (`archive/10_sample_and_label.py`) burned five seconds of CPU
across five minutes of wall clock, blocked on network calls that never returned,
and was rewritten sequentially as `13_label_simple.py`. What 18 was is not
recorded.

## Method

**Tone is not stance.** An early keyword-based stance grouping put Pieter Hanson
at the top of the backlash figures; he was a viral satirical meme, not backlash.
That finding is the empirical justification for LLM classification over keyword
stance assignment.

**A late peak may be a different event.** The Kavanaugh hearing was 27 September
2018; the Senate confirmed him on 6 October, day +9. Countries peaking at +7 to
+9 are probably covering the confirmation vote, not arriving late to the hearing.

**Countries with fewer than 15 articles are excluded from lag statistics.** With
six articles, the peak is wherever two of them happened to land.

## Legal

**Nexis Uni is accessible through IU SSO, but its licence prohibits automated or
AI use of fetched text.** It is not used in this pipeline.

**`data/` holds the full text of copyrighted news articles** and is excluded
from version control for that reason as well as size.
