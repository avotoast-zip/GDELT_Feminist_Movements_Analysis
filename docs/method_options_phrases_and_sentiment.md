# Method options: phrases, sentiment, and stance at scale

Written 19 Aug 2026, before implementing anything, so the choice is made on the
merits rather than on whatever was easiest to code.

Three questions are on the table:

1. How do we find common **phrases** rather than common words, so that a generic
   LDA term like *people* resolves into what it actually means in context?
2. Is **sentiment analysis** worth adding, and what would it tell us?
3. Can the **full-corpus stance labelling** be done without the $100–150 of API
   credit the project ran out of?

The short answers: yes to phrases and there are two complementary methods worth
running; sentiment is cheap but answers a different question than the one we
care about, and this project has already been burned once by exactly that
confusion; and yes, full-corpus stance labelling can almost certainly be done
locally for free.

---

## 1. Phrases instead of words

The corpus is in seven languages, so anything requiring hand-built language
resources multiplies by seven. That constraint rules out more than it looks like
it does. Four families of method, weakest to strongest for our case.

### a. Raw n-gram counting

Count every 2-to-4 word sequence, rank by frequency.
`CountVectorizer(ngram_range=(2,4))` — we already have scikit-learn.

Honest assessment: this is the obvious thing and it mostly produces boilerplate.
The top bigrams of any news corpus are *said the*, *of the*, *last year*. It
becomes useful only with a dispersion filter (a phrase must appear across many
outlets, not 300 times in one syndicated story) — and we have the duplicate map
needed to build that filter. Worth running as a one-hour baseline, not as the
answer.

### b. Statistical collocations — the corpus-linguistics standard

Instead of "which phrases are frequent", ask **"which words co-occur far more
often than chance"**. Scores: mutual information (PMI), log-likelihood, t-score,
and log-Dice.

This is the established method in corpus-based discourse analysis, which is the
tradition this project sits in — the same toolkit used in published
corpus-assisted studies of media coverage, including of #MeToo itself. It is
purely statistical, so it works identically in Swedish and Portuguese with no
per-language resources.

**log-Dice is the score to use.** PMI over-rewards rare pairs (a typo appearing
twice next to one word scores enormously), and raw log-likelihood scales with
corpus size so English and Swedish results can't be compared. log-Dice is
frequency-stable and comparable across our very unequal language subcorpora.

This directly answers the *people* problem. We ask for the collocates of
*people* in the English subcorpus, and it comes back as *young people*, *people
came forward*, *people in power*, *ordinary people* — with a KWIC concordance
(keyword in context) showing the actual sentences behind each. That is both an
analysis and a validation tool: it lets us read what a topic word is doing
rather than guessing.

### c. Automatic phrase detection fed back into LDA

`gensim`'s `Phrases` model learns which adjacent word pairs behave as a single
unit and rewrites the text so *sexual_harassment* and *due_process* become
single tokens. Run it twice and you get trigrams.

The attraction is that it plugs into the pipeline we already have: re-run
`17_lda_topics.py` on phrase-joined text and the topics come out labelled with
phrases instead of bare words, with no change to the modelling logic. Cheapest
real improvement available.

### d. Keyphrase extraction per document

- **YAKE** — unsupervised, statistical, explicitly multilingual, no training,
  no models to download. Extracts the keyphrases *of a single document*.
- **KeyBERT + KeyphraseVectorizers** — embeds candidate phrases and the document
  with a multilingual sentence transformer and keeps the phrases closest to the
  document meaning. KeyphraseVectorizers restricts candidates to grammatical
  noun phrases using spaCy part-of-speech patterns (the PatternRank approach),
  which is what stops it returning fragments like *of the movement*. spaCy has
  models for all seven of our languages.

These answer "what is *this article* about", which is a different and also
useful question — it would give every article a phrase-level descriptor we could
aggregate by country or by month.

### e. Worth flagging: BERTopic

Not strictly phrase extraction, but it is the modern replacement for LDA and it
solves this problem as a side effect: it clusters multilingual sentence
embeddings and labels each cluster with c-TF-IDF over n-grams, so topics arrive
already described by phrases. Comparative studies generally find it produces
more coherent and more interpretable topics than LDA on news text.

The catch for us: it needs embeddings for ~77k articles (feasible on a laptop,
a few hours), and it would be a second topic model rather than a fix to the
existing one. My view is that it belongs on the list as a validation run — if
BERTopic and LDA agree on the Spanish/Portuguese versus English/French split,
that finding gets considerably stronger.

### Recommendation

Two layers, in this order:

1. **log-Dice collocations plus KWIC** for a chosen set of target words —
   *people*, *women*, *victim*, *movement*, *allegations*, and the generic terms
   that currently make topics hard to read. Free, fast, no models, and it is the
   method a discourse-analysis reviewer would expect to see.
2. **gensim `Phrases` feeding the existing LDA**, so the topic tables and word
   clouds are rebuilt on phrases. Small change, visible improvement.

Then BERTopic as an optional cross-check, and YAKE/KeyBERT only if we decide we
want per-article descriptors.

---

## 2. Sentiment analysis

### What is available, and it is free

| Model | Languages | Trained on | Note |
|---|---|---|---|
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | 8 incl. en, es, fr, pt, de, it | tweets | The most-used multilingual option |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` | same | tweets | Reports 0.693 F1 / accuracy on its own test set |
| `nlptown/bert-base-multilingual-uncased-sentiment` | 6 incl. en, fr, de, es | product reviews | 1–5 star output |
| `tabularisai/multilingual-sentiment-analysis`, `clapAI/roberta-base-multilingual-sentiment` | broader | mixed | Newer, wider language coverage |
| `pysentimiento` | es, en, it, pt | social media | Convenient wrapper, strong on Spanish |

All run locally on CPU. No API cost.

### Two real problems before we use any of them

**Domain mismatch.** Every one of these is trained on tweets or product reviews.
Our corpus is 700-word news articles. A model that learned sentiment from "this
phone is terrible" is being asked to score court reporting. The 0.693 figure
above is on its own tweet test set; on news it will be lower, and we would not
know by how much without checking.

**Sentiment is not stance, and this project already knows that.** An article
reporting a femicide has strongly negative sentiment and is unambiguously
sympathetic to the movement. An article approvingly quoting a lawyer on due
process may read as calm and neutral while being backlash coverage. This is
precisely the failure that put Pieter Hanson — a satirical meme — at the top of
the backlash rankings when the project used GDELT tone. Running a sentiment
model and interpreting it as support-versus-backlash would repeat that mistake
with a better-dressed tool.

### Where sentiment is legitimately useful

Not as a stance proxy, but as its own variable:

- **Emotional register by country and language.** Do Spanish-language femicide
  reports read as more negative than English-language movement commentary? That
  is a real, publishable comparison.
- **Change over time within a country.** Does register shift after a national
  trigger event like Kulturprofilen?
- **As a contrast with stance.** Articles that are negative in sentiment but
  supportive in stance are exactly the population that keyword-based methods
  misclassify. Quantifying that overlap is a methods contribution in itself.

Practical notes: these models cap at 512 tokens, so articles need chunking with
scores aggregated per article; and whatever we run should be validated against
the 398 hand-labelled articles we already have before any number is reported.

---

## 3. The one I would do first: stance at full-corpus scale, for free

This was not in the question, but the research turned it up and it matters more
than either of the above, because it unblocks the project's largest open item —
the $100–150 of API credit that stopped full-corpus labelling.

**Zero-shot classification via natural language inference** does this without an
API. You give the model the article and a hypothesis — "This text is about the
#MeToo movement" — and it returns whether the text entails it.

- `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` — 27 languages,
  ~300M parameters, 74–87% accuracy across XNLI languages. Covers our seven.
  Caveat the model card states plainly: its multilingual training data was
  machine-translated, which costs quality.
- **Political DEBATE** (published in *Political Analysis*) — purpose-built for
  political text classification, 150–435M parameters. Zero-shot it reaches 95.8%
  F1 on its own benchmark, beating Claude 3.5 Sonnet on that test, and with only
  **10–25 labelled documents** in few-shot it outperforms supervised classifiers
  trained on thousands. It runs on a MacBook. The limitation: **English only**,
  and the authors are explicit that non-English behaviour is unknown.

So the practical shape is a split: Political DEBATE for the English subcorpus,
which is the largest single slice at 38,122 articles, and mDeBERTa zero-shot for
the other six languages — or, better, a small multilingual classifier fine-tuned
on the 398 articles we have already labelled.

**That existing 398-article labelled sample is the asset here.** It is ground
truth. It lets us measure any of these approaches against Gemini's labels rather
than trusting a benchmark number, and 398 is comfortably above the 10–25 that
few-shot methods need. The cost is a few hours of laptop compute and zero
dollars, and the output settles the composition question for all 76,925 unique
articles instead of 398 of them.

---

## Suggested order

1. **Phrases** — log-Dice collocations plus KWIC, then `Phrases` into the
   existing LDA. This is what was asked for and it is the fastest visible win.
2. **Free stance labelling** — validate a zero-shot and a few-shot approach
   against the 398 labels, then run whichever wins over the full corpus.
3. **Sentiment** — only after stance exists, and framed as emotional register,
   never as support versus backlash.

---

## Sources

- [KeyBERT](https://github.com/MaartenGr/KeyBERT) · [KeyphraseVectorizers documentation](https://keyphrase-vectorizers.readthedocs.io/en/latest/KeyphraseVectorizers.html) · [Unsupervised keyphrase extraction with PatternRank](https://towardsdatascience.com/unsupervised-keyphrase-extraction-with-patternrank-28ec3ca737f0/)
- [How BERTopic differs from LDA](https://bertopic.com/how-is-bertopic-different-from-lda/) · [Comparative analysis of BERTopic versus LDA (2026)](https://journals.sagepub.com/doi/10.1177/14413582251399667) · [Multilingual transformer and BERTopic for short text](https://arxiv.org/pdf/2402.03067)
- [Statistics in Corpus Linguistics: Semantics and Discourse](https://www.cambridge.org/core/books/statistics-in-corpus-linguistics/semantics-and-discourse/3CC9D42A719A484A565BC139E9353A2C) · [Corpus-based critical discourse analysis of news reports on the #MeToo movement](https://core.ac.uk/works/46562913) · [Integrating corpus linguistics and text mining for European media coverage](https://www.mdpi.com/2673-5172/6/4/196)
- [cardiffnlp/twitter-xlm-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) · [cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual) · [tabularisai/multilingual-sentiment-analysis](https://huggingface.co/tabularisai/multilingual-sentiment-analysis) · [XLM-T framework](https://github.com/cardiffnlp/xlm-t)
- [MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) · [Political DEBATE: efficient zero-shot and few-shot classifiers for political text](https://www.cambridge.org/core/journals/political-analysis/article/political-debate-efficient-zeroshot-and-fewshot-classifiers-for-political-text/8D0B3E2AAF711F4812E42466DE503A13) · [Stance detection: a practical guide](https://www.cambridge.org/core/journals/political-science-research-and-methods/article/stance-detection-a-practical-guide-to-classifying-political-beliefs-in-text/E227E746BD7D9751526DA0EC2C378787) · [Automated stance detection in complex topics and small languages](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0302380)
