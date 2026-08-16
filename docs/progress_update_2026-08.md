# Update

Everything done since the last email.

---

# 1. Searched the article text for keywords

Before this we only searched the web address and the headline. 80% of articles
have no headline saved, so most were matched on the web address alone. A web
address holds one or two words. An article holds hundreds.

Now we search the full downloaded text.

**What changed**

Keywords found per article went from 1.02 to 1.53.

Hashtag share went from 52.8% to 54.7%. 2,004 articles gained a hashtag they
always had.

France moved most, from 16.0% to 22.9%.

**What it means**

The change is small. That matters. It means the split between hashtag articles
and general-vocabulary articles is not caused by weak searching. France is still
77% general vocabulary after we read the whole article. The two groups are
genuinely different.

**Backlash vocabulary was almost invisible before**

| Term | Found before | Found now |
|---|---|---|
| false accusations | 31 | 837 |
| witch hunt | 337 | 1,106 |
| due process | 291 | 1,076 |

Articles containing backlash vocabulary went from 2.9% to 5.3%.

Articles containing both support and backlash vocabulary went from 71 to 2,899.

This is because backlash language does not appear in headlines or web addresses.
No editor writes "false accusations" into a web address. It appears in the body,
in a quoted defence lawyer, in a paragraph about due process.

**This corrects something I reported earlier.** I said the two vocabulary groups
were effectively separate, at 0.2% overlap. On full text the overlap is 2.7% and
will rise further. The earlier backlash figures by country were built on the
half of the vocabulary that was visible.

---

# 2. Removed duplicate articles using the text

The first pass compared web addresses and headlines. It removed 7,429.

This pass compares the article text itself.

| Overlap required | Duplicate copies found | Unique left |
|---|---|---|
| 100%, identical text | 24,165 | 86,700 |
| 90% or more | 30,259 | 80,606 |
| 80% or more | 33,197 | 77,668 |

We use the 80% cutoff. **33,940 copies removed, leaving 76,925 unique articles.**

**How duplicates are found.** Each article's text is cut into overlapping
five-word chunks. Two articles are compared by how many chunks they share. 80%
or more means the same story.

**I verified this.** On 99 sampled pairs the true word overlap had a median of
100%. Zero pairs fell below 50%. These are literally identical texts.

**Which sites produced the most copies**

| Site | Country | Copies | Distinct stories |
|---|---|---|---|
| iheart.com | US | 2,598 | 40 |
| maville.com | France | 504 | 143 |
| indiatimes.com | India | 452 | 116 |
| biobiochile.cl | Chile | 260 | 1 |

iHeartMedia owns hundreds of US radio stations. Each gets its own web address.
One article publishes to all of them. One story appeared 308 times.

**This changed the country picture.** The United States dropped from 24.2% of
the corpus to 19.3%. Every other major country rose. Part of American dominance
was syndication, not editorial attention.

**One limitation.** The rule that keeps a wire story separate when a newspaper
adds its own editorial comment could not run properly. It needs the stance
labels, which we do not have yet. Only 130 articles were split out on text
difference alone. This should be re-run after labelling.

---

# 3. Added a second dashboard page

The first dashboard covered the collected corpus. The new page covers what
happened after: downloading and duplicate removal.

It shows what happened to every link, download success by country and year, how
the corpus shape changed, the duplicate table above, and which sites produced
the most copies.

---

# 4. Event timeline and how coverage spread

Two things. A timeline of daily article counts with events marked. And, for each
event, how long coverage took to reach each country.

**Events that spread almost instantly**

The Deneuve open letter in Le Monde, 9 January 2018. 25 countries covered it.
Nine of eleven measurable countries peaked the same day or the next.

**Events that stayed home**

The Kulturprofilen exposé in Sweden, 21 November 2017. This triggered Sweden's
national reckoning. It reached 2 countries.

#EleNão in Brazil, September 2018. It reached 1 country.

| Event | Origin | Countries reached |
|---|---|---|
| Deneuve letter | France | 25 |
| Milano tweet | United States | 35 |
| Kavanaugh hearing | United States | 24 |
| India wave | India | 16 |
| Kulturprofilen | Sweden | 2 |
| #EleNão | Brazil | 1 |

A French cultural controversy went worldwide in a day. A Swedish one that
produced a national movement barely crossed the border.

**The Nordic countries are a separate story.** After the Milano tweet most
countries peaked within four days. Denmark peaked at 7 days, Finland 36, Sweden
37, Norway 45. That is not slow reaction. Those are their own national waves
with their own triggers. Sweden's was the Kulturprofilen exposé.

**One event cannot be studied and I want to be clear about why.** The Weinstein
story broke on 5 October 2017. Our keyword list is built on #MeToo terms, and
the hashtag was created on 15 October. Articles from those first ten days
contain none of our keywords, so they were never collected.

Daily counts: 14 October, 50 articles. 15 October, 95. 16 October, 360. 17
October, 528. The world did not start writing on the 16th. Our collection did.

So every country appears to peak around day 12 for Weinstein. That is our
starting line, not a delay. The chart is marked as unusable.

Burning Sun in South Korea also cannot be studied. Our corpus contains one
South Korean article with text in the whole of 2019.

---

# 5. LDA topic modelling

**What LDA does.** It finds groups of words that appear together. Each group is
a topic. Each article gets a score for how much of each topic it contains. It
does not name the topics. We do that.

**The language problem.** LDA counts words. Run it across all languages at once
and Spanish groups with Spanish, French with French. You get a map of languages,
not themes. So we run a separate model per language.

Seven languages had enough articles: English 38,122, Spanish 18,675, French
15,023, Portuguese 3,354, German 2,734, Swedish 2,079, Italian 936.

**Four data problems the first run exposed**

Broken character encoding in 4,314 articles. Spanish "prisión" was stored as
"prisiÃ³n". Whole topics were built from the broken fragments. Now repaired.

Website furniture. A Swedish topic was built entirely around "annons", meaning
advertisement. A German one around the privacy policy. A French one around
cookie notices. Now filtered.

Junk pages. An English topic was a country dropdown list from a registration
form. Now filtered.

Stale pages. 7,485 articles served today's content instead of the old article.
Excluded.

**Sample of the results, English**

| Topic | Words |
|---|---|
| Legal and political investigations | allegations, court, investigation, akbar, kavanaugh |
| Music industry | music, kelly, song, awards, singer |
| Workplace | harassment, workplace, company, employees |
| Film industry | film, actor, india, director, actress |
| Indian criminal cases | police, rape, delhi, nirbhaya |
| US politics and Weinstein | trump, weinstein, hollywood, clinton, burke |

**French produced the cleanest set:** Weinstein, criminal trials, police
investigations, #BalanceTonPorc and the Muller case, Haenel and Polanski, the
Tariq Ramadan case, the Darmanin case.

**Two results worth noting.**

The Swedish Academy appears as its own isolated topic. The propagation analysis
independently showed that story reached two countries. Two different methods
agreeing that Sweden's wave was domestic.

Spanish and Portuguese topics are dominated by individual femicide cases and
court vocabulary. English and French topics are dominated by the movement and
named figures. This is the same split we found by other means, appearing again
without anyone labelling anything.

---

# 6. Where the numbers stand

| | Count |
|---|---|
| Articles collected | 212,957 |
| Links tried | 205,528 |
| Downloaded with text | 107,481 |
| Unique after duplicate removal | 76,925 |
| Usable for topic modelling | 92,096 |

---

# 7. Still open

**The decision about what we are studying.** A sample of 398 downloaded articles
found that 17% of general-vocabulary articles are actually about the movement,
against 90% of hashtag articles. That means roughly 42,000 of the 50,700
general-vocabulary articles are ordinary crime and court reporting.

**Labelling the full corpus.** Needs about $100 to $150 in AI credits. The
project ran out.

**Two known data problems.** Outlet names are collapsed, so g1.globo.com,
oglobo.globo.com and extra.globo.com all appear as globo.com. And 7% of
downloaded pages served current content instead of the original article.
