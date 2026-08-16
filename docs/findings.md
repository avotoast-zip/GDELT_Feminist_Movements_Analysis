# Findings

What the project has established, and how confident each result is.
Dated progress notes to supervisors are in `progress_update_2026-08.md`.

---

## 1. Corpus composition: the central problem

Only **56.3%** of the corpus matched a hashtag term. **36.8%** matched only
generic gender-violence vocabulary. Hashtag-only retention varies enormously:

| High | | Low | |
|---|---|---|---|
| United States | 79% | France | 19% |
| India | 68% | Mexico | 15% |
| | | Argentina | 11% |
| | | Brazil | 9% |
| | | Peru | 7% |

Volume ranking across countries is therefore **partly an artifact of codebook
morphology**, not of editorial attention.

A 398-article LLM-labelled sample settles what the generic-vocabulary material
actually is:

| Stratum | n | About the movement | 95% CI |
|---|---|---|---|
| Hashtag-matched | 199 | 90.5% | 86–94 |
| Generic-matched | 199 | 17.1% | 12–23 |

Of roughly 50,700 fetched generic-matched articles, about **8,662 are movement
coverage and 42,037 are not** — they are ordinary crime and court reporting.

Stance among articles that *are* about the movement: 44.9% support, 44.4%
neutral, 10.7% backlash.

**This is the open decision.** Either the generic-vocabulary articles are
filtered out, or the study is explicitly about two populations. It cannot stay
implicit.

---

## 2. Full-text re-matching: a small change, and that is the result

Until August 2026, matching used only the URL and the headline. 80% of articles
have no stored headline, so most were matched on the URL alone — one or two
words against an article's several hundred.

After searching the downloaded body text:

| Measure | Before | After |
|---|---|---|
| Keywords found per article | 1.02 | 1.53 |
| Hashtag share | 52.8% | 54.7% |
| Articles that gained a hashtag | — | 2,004 |
| France, hashtag share | 16.0% | 22.9% |

The change is small, and that is what makes it informative: the
hashtag/general-vocabulary split is not an artifact of weak searching. France is
still 77% general vocabulary after reading whole articles. The two groups are
genuinely different populations.

### Backlash vocabulary was almost invisible before

| Term | Before | After |
|---|---|---|
| false accusations | 31 | 837 |
| witch hunt | 337 | 1,106 |
| due process | 291 | 1,076 |

Articles containing backlash vocabulary: 2.9% → 5.3%. Articles containing
**both** support and backlash vocabulary: 71 → 2,899.

No editor writes "false accusations" into a URL. It appears in the body, in a
quoted defence lawyer, in a paragraph about due process.

**This corrects an earlier claim.** The two vocabulary groups were reported as
effectively separate at 0.2% overlap. On full text the overlap is 2.7% and will
rise further. Earlier country-level backlash figures were built on the half of
the vocabulary that was visible.

---

## 3. Deduplication: American dominance was partly syndication

Pre-fetch deduplication on URLs and titles removed 7,429. Text-based
deduplication over five-word shingles removed far more:

| Overlap required | Copies found | Unique remaining |
|---|---|---|
| 100%, identical text | 24,165 | 86,700 |
| ≥ 90% | 30,259 | 80,606 |
| **≥ 80% (used)** | **33,940** | **76,925** |

Validated on 99 sampled pairs: median true word overlap 100%, no pair below 50%.
These are literally identical texts.

| Site | Country | Copies | Distinct stories |
|---|---|---|---|
| iheart.com | US | 2,598 | 40 |
| maville.com | France | 504 | 143 |
| indiatimes.com | India | 452 | 116 |
| biobiochile.cl | Chile | 260 | 1 |

iHeartMedia owns hundreds of US radio stations; one article publishes to all of
them, and one story appeared 308 times.

**The United States fell from 24.2% of the corpus to 19.3%.** Every other major
country rose.

---

## 4. Event propagation: reach is the variable, not speed

| Event | Origin | Countries reached |
|---|---|---|
| Milano tweet | United States | 35 |
| Deneuve open letter | France | 25 |
| Kavanaugh hearing | United States | 24 |
| India surge (Akbar / Ramani) | India | 16 |
| Kulturprofilen exposé | Sweden | 2 |
| #EleNão | Brazil | 1 |

A French cultural controversy went worldwide within a day — nine of eleven
measurable countries peaked same-day or next-day. A Swedish one that produced a
full national reckoning barely crossed the border.

**The Nordic countries are a separate story.** After the Milano tweet most
countries peaked within four days; Denmark peaked at +7, Finland +36, Sweden +37,
Norway +45. Those are not slow reactions. They are separate national waves with
their own triggers — Sweden's being the Kulturprofilen exposé.

Two events are formally unusable: Weinstein (predates the hashtag the codebook
searches for) and Burning Sun (one South Korean article with text in all of
2019). See `known_issues.md`.

---

## 5. Topic modelling: the same split, found independently

LDA is run separately per language — a pooled model maps languages, not themes.
Seven languages had enough articles: English 38,122, Spanish 18,675, French
15,023, Portuguese 3,354, German 2,734, Swedish 2,079, Italian 936.

English topics: legal and political investigations; music industry; workplace;
film industry; Indian criminal cases; US politics and Weinstein.

French produced the cleanest set: Weinstein, criminal trials, police
investigations, #BalanceTonPorc and the Muller case, Haenel and Polanski, Tariq
Ramadan, Darmanin.

**Two results worth keeping.**

The Swedish Academy appears as its own isolated topic — the propagation analysis
independently showed that story reached two countries. Two methods agreeing that
Sweden's wave was domestic.

Spanish and Portuguese topics are dominated by individual femicide cases and
court vocabulary; English and French by the movement and named figures. **This is
the hashtag/generic split from section 1, reappearing without anyone labelling
anything.**

### Four data problems the first LDA run exposed, since fixed

- Mojibake in 4,314 articles (`prisión` stored as `prisiÃ³n`) built whole topics
  out of broken fragments.
- Site furniture: a Swedish topic built entirely around *annons* (advertisement),
  a German one around a privacy policy, a French one around cookie notices.
- Junk pages: an English topic that was a country dropdown from a registration form.
- Stale pages: 7,485 articles served today's content instead of the original.

---

## 6. Where the numbers stand

| | Count |
|---|---|
| Articles collected | 212,957 |
| Links tried | 205,528 |
| Downloaded with text | 107,481 |
| Unique after duplicate removal | 76,925 |
| Usable for topic modelling | 80,923 |

---

## 7. Open

1. **The scope decision** — are generic-vocabulary articles in the study or not.
2. **Full-corpus labelling** — blocked on ~$100–150 of API credit.
3. **Re-running 17 and 20 on the deduplicated corpus** — see `known_issues.md`.
4. **Outlet name collapsing** — subdomains merge into one domain.
