#!/usr/bin/env python3
"""
17_lda_topics.py
----------------
Finds topics in the downloaded article text, makes word clouds, and charts how
topics vary by country and over time.

WHAT LDA DOES

You give it a pile of articles. It finds groups of words that tend to appear
together. Each group is a topic. Each article gets a score for how much of each
topic it contains.

You choose how many topics to look for. It gives back word groups like
{trial, court, testimony, verdict, judge}. You name the topics yourself. LDA
does not name them.

THE LANGUAGE PROBLEM, AND HOW THIS SCRIPT HANDLES IT

LDA counts words. Our corpus is in dozens of languages. Run it on everything at
once and Spanish articles group with Spanish articles, French with French. You
get a map of languages, not a map of themes.

So this script runs LDA SEPARATELY FOR EACH LANGUAGE by default. You get English
topics, Spanish topics, French topics, and so on. Those are comparable to each
other by theme even though the words differ.

Use --pooled if you want the single-model version anyway, to show what it looks
like.

TWO FILTERS APPLIED FIRST

1. Articles whose text mentions a year from 2023 to 2026 are dropped. Every
   article was published between 2017 and 2019. A later year means the website
   served today's content instead of the old article. About 7% of downloads.
   Leaving them in would put current news into the topics.

2. Articles under 800 characters are dropped. Too short to carry a topic.

RUN
    pip install scikit-learn wordcloud matplotlib langdetect pandas
    python 17_lda_topics.py                    # per language, 8 topics each
    python 17_lda_topics.py --topics 12
    python 17_lda_topics.py --languages en es fr
    python 17_lda_topics.py --pooled           # one model over everything

OUTPUT
    lda_topics_<lang>.csv        top words per topic
    lda_wordcloud_<lang>.png     word clouds, one panel per topic
    lda_by_country_<lang>.png    which topics dominate which countries
    lda_over_time_<lang>.png     topics rising and falling by month
    lda_article_topics.csv       every article and its strongest topic
    lda_report.txt               the summary
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
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
try:
    import ftfy
    HAVE_FTFY = True
except ImportError:
    HAVE_FTFY = False
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

IN_FILE = PROCESSED / "labeled_articles.csv"
MIN_CHARS = 800
MIN_ARTICLES_PER_LANG = 300
STALE = re.compile(r"\b(202[3-6])\b")

# Stopwords per language. LDA is very sensitive to these: leave them out and
# every topic is dominated by "the", "and", "of".
STOP = {
"en": """a about after against all also an and any are as at be because been before being
between both but by can could did do does doing down during each few for from further had
has have having he her here hers him his how i if in into is it its itself just me more most
my no nor not of off on once only or other our out over own same she should so some such
than that the their them then there these they this those through to too under until up very
was we were what when where which while who whom why will with would you your said says say
told also one two new first last year years time day says will can may might""",
"es": """a al algo algunas algunos ante antes como con contra cual cuando de del desde donde dos
el ella ellas ellos en entre era erais eran es esa esas ese eso esos esta estas este esto
estos ha habia han hasta hay la las le les lo los mas me mi mis mucho muy no nos o os otra
otras otro otros para pero poco por porque que quien se sea ser si sin sobre solo son su sus
te tiene tienen todo todos tras un una uno unos y ya fue fueron dijo segun ano anos dos tres""",
"fr": """a au aux avec ce ces dans de des du elle en et eux il je la le les leur lui ma mais me
meme mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes
toi ton tu un une vos votre vous c d j l m n s t y ete etee etees etes etant suis es est
sommes etes sont plus tout tous cette apres avoir faire dit selon ans deux trois ete""",
"pt": """a ao aos aquela aquelas aquele aqueles aquilo as ate com como da das de dela delas dele
deles depois do dos e ela elas ele eles em entre era eram essa essas esse esses esta estas
este estes eu foi foram ha isso isto ja la lhe lhes mais mas me mesmo meu meus minha na nas
nao nem no nos nossa nosso num numa o os ou para pela pelas pelo pelos por qual quando que
quem se sem ser seu seus so sua suas tambem te tem tu um uma voce anos dois disse""",
"de": """aber alle als also am an auch auf aus bei bin bis bist da damit dann das dass dein deine
dem den der des dem die dies diese doch dort du durch ein eine einem einen einer eines er es
euer eure fur hab habe haben hat hatte hier hin ich ihr ihre im in ist ja jede jedem jeden
jeder jenes kann kein keine machen mein meine mit muss nach nicht noch nun nur ob oder ohne
sehr sein seine sich sie sind so soll ueber um und uns unser vom von vor war waren was weg
weil wenn werden wie wieder wir wird wirst wo zu zum zur ueber jahre sagte""",
"it": """a ad al alla alle allo anche che chi ci come con cui da dal dei del della delle di do
dopo e ed egli essere fa fare gli ha hanno ho i il in io la le lei li lo loro ma me mi ne
nel nella no noi non nostro o per piu poi qua quale quando quel quella questo qui se senza
si sia siamo sono su sul sulla suo te ti tra tu tuo un una uno voi anni ha detto""",
"sv": """och det att i en jag hon som han pa den med var sig for sa till ar men ett om hade de
av icke mig du henne da sin nu har inte hans honom skulle hennes dar min man ej vid kunde
nagot fran ut nar efter upp vi dem vara vad over an dig kan sina har eller vill blir mot
ni bland detta ocksa efter blivit dess inom mellan sadan sagt""",
}
# Website furniture that LDA will otherwise turn into its own topic. Found by
# reading the first run: an entire Swedish topic was built around "annons"
# (advertisement), a German one around "datenschutzerklaerung" (privacy policy),
# and a French one around "cookies".
BOILERPLATE = {
"en": """cookies cookie privacy policy terms newsletter subscribe advertisement
advertise sponsored share tweet facebook twitter instagram whatsapp email print
comments comment login register sign copyright rights reserved read more click
here photo image getty reuters associated press file update updated published
tags related stories follow us app download menu search home news sport
business world video podcast newsletter signup account settings""",
"es": """cookies cookie politica privacidad terminos boletin suscribete publicidad
patrocinado compartir comentarios comentario iniciar sesion registrate derechos
reservados leer mas foto imagen efe reuters archivo actualizado publicado
etiquetas relacionadas siguenos aplicacion descargar menu buscar inicio noticias
deportes negocios mundo video suscripcion cuenta""",
"fr": """cookies cookie politique confidentialite conditions newsletter abonnez
publicite sponsorise partager commentaires commentaire connexion inscrivez
droits reserves lire plus photo image afp reuters archive mis jour publie
etiquettes articles suivez application telecharger menu recherche accueil
actualites sport economie monde video abonnement compte contenu""",
"pt": """cookies cookie politica privacidade termos newsletter assine publicidade
patrocinado compartilhar comentarios comentario entrar cadastre direitos
reservados leia mais foto imagem reuters arquivo atualizado publicado
tags relacionadas siga aplicativo baixar menu buscar inicio noticias esportes
negocios mundo video assinatura conta""",
"de": """cookies cookie datenschutz datenschutzerklaerung nutzungsbedingungen
newsletter abonnieren werbung anzeige gesponsert teilen kommentare kommentar
anmelden registrieren rechte vorbehalten mehr lesen foto bild getty images
reuters archiv aktualisiert veroeffentlicht schlagworte folgen app herunterladen
menue suche startseite nachrichten sport wirtschaft welt video abo konto uhr""",
"it": """cookies cookie privacy termini newsletter abbonati pubblicita
sponsorizzato condividi commenti commento accedi registrati diritti riservati
leggi altro foto immagine ansa reuters archivio aggiornato pubblicato tag
correlati seguici app scarica menu cerca home notizie sport economia mondo""",
"sv": """annons annonser cookies cookie integritetspolicy villkor nyhetsbrev
prenumerera reklam sponsrad dela kommentarer kommentar logga registrera
rattigheter forbehallna las mer foto bild tt reuters arkiv uppdaterad publicerad
taggar relaterade folj app ladda meny sok hem nyheter sport ekonomi varlden""",
}


# The lists above were originally written without accents, while the articles
# keep theirs. So "nar" was filtered but "när" was not, and Swedish produced two
# topics made entirely of function words. These are the accented forms.
ACCENTED = {
"es": """más está también qué cómo día año años había están sí después
según aún así méxico españa mujeres""",
"fr": """était été où déjà très même après plutôt là ça c'est qu'il qu'elle
années année aujourd'hui être fait faits""",
"pt": """são não também está só até já português mês ano anos você
então além porque""",
"de": """für über während müssen können hätte wäre größer später jahre jahr
möchte natürlich zurück""",
"sv": """när här där något över många även måste får går fått sedan andra
bara mycket finns kommer vet blir sitt sina såg än""",
"it": """più però così già perché città può può essere anni anno""",
}


def _strip_accents(t):
    import unicodedata
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c))


def stop_for(lang):
    """Return stopwords in BOTH accented and unaccented form, so a list written
    either way still filters correctly."""
    base = " ".join([STOP.get(lang, ""), STOP["en"],
                     BOILERPLATE.get(lang, ""), BOILERPLATE["en"],
                     ACCENTED.get(lang, "")])
    words = set(base.split())
    words |= {_strip_accents(w) for w in words}
    return sorted(words)


DEFAULT_STOP = STOP["en"]





def load_and_clean():
    cols = ["url", "source_country", "outlet", "published", "article_text",
            "fetch_status", "text_chars", "keywords_matched", "has_hashtag_term"]
    parts = []
    for ch in pd.read_csv(IN_FILE, usecols=cols, chunksize=100_000, low_memory=False):
        ch = ch[ch["fetch_status"].isin(["OK", "OK_WAYBACK"])]
        ch = ch[ch["article_text"].notna()]
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df)
    df = df[df["text_chars"] >= MIN_CHARS]
    n1 = len(df)
    stale = df["article_text"].astype(str).str.slice(0, 6000).str.contains(STALE, na=False)
    df = df[~stale]
    n2 = len(df)

    # Some pages were saved with broken character encoding. Spanish "prisión"
    # became "prisiÃ³n", Portuguese "violência" became "violãªncia". Left alone,
    # LDA builds entire topics out of the broken fragments. 5% of articles are
    # affected. ftfy repairs them.
    if HAVE_FTFY:
        bad = df["article_text"].astype(str).str.contains("Ã|Â|ð|ï¼", regex=True, na=False)
        if bad.any():
            df.loc[bad, "article_text"] = df.loc[bad, "article_text"].astype(str).apply(ftfy.fix_text)
            print(f"  repaired broken encoding in {int(bad.sum()):,} articles")
    else:
        print("  NOTE: ftfy not installed. Broken-encoding articles will create")
        print("        junk topics. Run: pip install ftfy")

    # Some mojibake cannot be repaired, usually Chinese or Japanese text that
    # was double-encoded. ftfy leaves it as strings like "ï¼å". If a lot of the
    # text still looks like that after repair, the article is unreadable and
    # would otherwise become its own topic. Drop it.
    def broken_ratio(t):
        t = str(t)[:2000]
        if not t:
            return 1.0
        bad = sum(1 for c in t if c in "ÃÂðï¼å¼è¾ºæºå½ç")
        return bad / len(t)
    br = df["article_text"].apply(broken_ratio)
    unreadable = br > 0.06
    if unreadable.any():
        print(f"  dropped {int(unreadable.sum()):,} articles with unreadable text")
    df = df[~unreadable]

    # Pages that were never articles: registration forms, site homepages,
    # error pages. They survived the date filter because they carry no date.
    JUNK = ["united states of america us virgin islands",
            "this site could be risky", "enable javascript",
            "your browser is out of date", "select your country",
            "subscribe to continue reading"]
    jl = df["article_text"].astype(str).str.slice(0, 1500).str.lower()
    junk = jl.apply(lambda t: any(j in t for j in JUNK))
    df = df[~junk]

    print(f"  with text            : {n0:,}")
    print(f"  long enough          : {n1:,}  (dropped {n0-n1:,} short)")
    print(f"  not stale content    : {n2:,}  (dropped {stale.sum():,} wrong-article)")
    print(f"  not a junk page      : {len(df):,}  (dropped {int(junk.sum()):,})")
    return df.reset_index(drop=True)


def detect_langs(series):
    """Detect on a Series of texts. Only ever called on a subsample, because
    langdetect on 100k articles takes far too long."""
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    def safe(t):
        try:
            return detect(str(t)[:700])
        except Exception:
            return None
    return series.apply(safe)


def run_lda(texts, n_topics, lang, max_features=8000):
    vec = CountVectorizer(max_df=0.55, min_df=8, max_features=max_features,
                          stop_words=stop_for(lang),
                          token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
    X = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                    learning_method="online", max_iter=25,
                                    n_jobs=-1)
    W = lda.fit_transform(X)
    vocab = vec.get_feature_names_out()
    return lda, W, vocab


def top_words(lda, vocab, k=15):
    out = []
    for i, comp in enumerate(lda.components_):
        idx = comp.argsort()[::-1][:k]
        out.append([(vocab[j], float(comp[j])) for j in idx])
    return out


def wordclouds(tw, lang, n_topics):
    ncol = 3
    nrow = (n_topics + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.2, nrow * 2.8))
    axes = axes.ravel() if n_topics > 1 else [axes]
    for i, words in enumerate(tw):
        freq = {w: v for w, v in words}
        wc = WordCloud(width=520, height=330, background_color="white",
                       colormap="RdGy", prefer_horizontal=0.95).generate_from_frequencies(freq)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Topic {i+1}", fontsize=11, fontweight="bold")
        axes[i].axis("off")
    for j in range(len(tw), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Topics in {lang.upper()} articles", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / f"lda_wordcloud_{lang}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def chart_by_country(sub, lang, n_topics):
    top = sub["source_country"].value_counts()
    top = top[top >= 40].head(12).index
    if len(top) < 2:
        return
    m = (sub[sub["source_country"].isin(top)]
         .groupby("source_country")[[f"t{i}" for i in range(n_topics)]].mean())
    m = m.loc[top]
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.45 * len(m))))
    im = ax.imshow(m.values, aspect="auto", cmap="RdPu")
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m.index, fontsize=9)
    ax.set_xticks(range(n_topics))
    ax.set_xticklabels([f"T{i+1}" for i in range(n_topics)], fontsize=9)
    ax.set_title(f"Topic mix by country, {lang.upper()}", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="average share")
    fig.tight_layout()
    fig.savefig(FIGURES / f"lda_by_country_{lang}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def chart_over_time(sub, lang, n_topics):
    s = sub.copy()
    s["month"] = pd.to_datetime(s["published"], errors="coerce", utc=True).dt.strftime("%Y-%m")
    m = s.groupby("month")[[f"t{i}" for i in range(n_topics)]].mean().sort_index()
    if len(m) < 3:
        return
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i in range(n_topics):
        ax.plot(m.index, m[f"t{i}"], label=f"Topic {i+1}", linewidth=1.7)
    ax.set_xticks(range(0, len(m), max(1, len(m) // 12)))
    ax.set_xticklabels([m.index[i] for i in range(0, len(m), max(1, len(m) // 12))],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("average share of articles")
    ax.set_title(f"Topics over time, {lang.upper()}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / f"lda_over_time_{lang}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=int, default=8)
    ap.add_argument("--languages", nargs="*", default=None,
                    help="e.g. --languages en es fr. Default: all with enough articles")
    ap.add_argument("--pooled", action="store_true",
                    help="one model over all languages (shows the language problem)")
    ap.add_argument("--max-per-lang", type=int, default=25_000,
                    help="cap per language to keep it fast")
    a = ap.parse_args()

    print("loading and filtering...")
    df = load_and_clean()

    # Detecting language on every article is far too slow. Subsample first.
    # We only need enough per language to fit a topic model.
    budget = min(len(df), max(a.max_per_lang * 6, 30_000))
    work = df.sample(n=budget, random_state=42).copy()
    print(f"\ndetecting language on a {budget:,}-article subsample "
          f"(a few minutes)...")
    work["lang"] = detect_langs(work["article_text"])
    work = work[work["lang"].notna()]
    # A Russian article misdetected as Portuguese put a whole junk topic into
    # the PT model. Keep only languages we have stopwords for, so every model
    # is properly cleaned.
    known = set(STOP) | set(BOILERPLATE)
    unknown = (~work["lang"].isin(known)).sum()
    if unknown:
        print(f"  {unknown:,} articles are in languages we have no stopword list "
              f"for; they are excluded")
    work = work[work["lang"].isin(known)]
    df = work

    counts = df["lang"].value_counts()
    print("\narticles per language:")
    print(counts.head(12).to_string())

    L = ["LDA TOPIC MODELLING", "=" * 60,
         f"articles used: {len(df):,}", ""]

    if a.pooled:
        langs = ["pooled"]
        groups = {"pooled": df.sample(n=min(a.max_per_lang, len(df)), random_state=42)}
        L.append("MODE: pooled (all languages in one model)")
        L.append("Expect topics that separate by language, not by theme.")
    else:
        langs = a.languages or [l for l, c in counts.items()
                                if c >= MIN_ARTICLES_PER_LANG][:6]
        groups = {}
        for l in langs:
            g = df[df["lang"] == l]
            if len(g) < MIN_ARTICLES_PER_LANG:
                print(f"  skipping {l}: only {len(g)} articles")
                continue
            groups[l] = g.sample(n=min(a.max_per_lang, len(g)), random_state=42)
        L.append(f"MODE: separate model per language ({', '.join(groups)})")
    L.append("")

    all_assign = []
    for lang, sub in groups.items():
        print(f"\n=== {lang.upper()}  ({len(sub):,} articles) ===")
        lda, W, vocab = run_lda(sub["article_text"].astype(str).str.slice(0, 6000),
                                a.topics, lang if lang != "pooled" else "en")
        tw = top_words(lda, vocab)
        L.append(f"--- {lang.upper()}  n={len(sub):,} ---")
        for i, words in enumerate(tw):
            ws = ", ".join(w for w, _ in words[:12])
            print(f"  Topic {i+1}: {ws}")
            L.append(f"  Topic {i+1}: {ws}")
        L.append("")

        pd.DataFrame({"topic": [f"Topic {i+1}" for i in range(a.topics)],
                      "top_words": [", ".join(w for w, _ in t) for t in tw]}
                     ).to_csv(TABLES / f"lda_topics_{lang}.csv", index=False)

        for i in range(a.topics):
            sub[f"t{i}"] = W[:, i]
        sub["main_topic"] = W.argmax(axis=1) + 1
        all_assign.append(sub[["url", "source_country", "published", "main_topic"]
                              ].assign(lang=lang))

        wordclouds(tw, lang, a.topics)
        chart_by_country(sub, lang, a.topics)
        chart_over_time(sub, lang, a.topics)
        print(f"  wrote lda_wordcloud_{lang}.png, lda_by_country_{lang}.png, "
              f"lda_over_time_{lang}.png")

    pd.concat(all_assign, ignore_index=True).to_csv(PROCESSED / "lda_article_topics.csv", index=False)
    open(REPORTS / "lda_report.txt", "w").write("\n".join(L))
    print("\nwrote lda_article_topics.csv and lda_report.txt")
    print("\nNEXT: read the topic word lists and give each topic a name.")
    print("LDA does not name topics. That part is yours.")


if __name__ == "__main__":
    main()
