#!/usr/bin/env python3
"""
_textutils.py
-------------
Shared text resources for the language-aware scripts: stopword lists,
website-boilerplate lists, accented-form lists, and language detection.

These lived inside 17_lda_topics.py until 22_phrases.py needed exactly the same
lists. Two copies would have drifted apart, and the moment they drift the topic
results and the phrase results stop being comparable — which is the whole point
of running both. So they live here and both scripts import them.

Add a stopword because a topic is polluted by it, and the phrase tables get the
same fix automatically.
"""

import re



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


# ── ROT47-obfuscated article bodies ────────────────────────────────────────
# Some US local-news platforms (Lee Enterprises / BLOX titles, among others)
# serve the article body ROT47-encoded as a scraper deterrent: the first
# paragraph arrives as plain text and the rest as strings like
# "kAmp?5 r2C=D@?" — which is "<p>And Carlson". Left alone these articles look
# like text, pass every length filter, and then contribute nothing but noise to
# topic models and phrase counts.
#
# ROT47 is its own inverse, so decoding is the same operation as encoding. The
# encoding is applied per paragraph, and only the encoded paragraphs are
# converted, because running it over the plain-text lead would destroy it.

ROT47_HINT = re.compile(r"kAm|k\^Am|E96 |2\?5 ")     # "<p>", "</p>", "the ", "and "
_HTML_TAG = re.compile(r"<[^>]{1,400}>")
_ROT47_SEG = re.compile(r"(?=kAm)|(?<=k\^Am)")


def rot47(s):
    return "".join(chr(33 + ((ord(c) - 33 + 47) % 94)) if 33 <= ord(c) <= 126 else c
                   for c in s)


def _alpha_share(s):
    ns = [c for c in s[:1200] if not c.isspace()]
    return sum(c.isalpha() for c in ns) / len(ns) if ns else 0.0


def _decode_windows(seg, win, offset, min_gain):
    """Decode qualifying fixed-size windows of a string. ROT47 maps character by
    character, so a window boundary can fall anywhere without harm."""
    changed = False
    parts, i = [], 0
    if offset:
        parts.append(seg[:offset])
        i = offset
    while i < len(seg):
        chunk = seg[i:i + win]
        dec = rot47(chunk)
        if len(chunk) >= 40 and _alpha_share(dec) - _alpha_share(chunk) > min_gain:
            chunk = dec
            changed = True
        parts.append(chunk)
        i += win
    return "".join(parts), changed


def repair_rot47(text, min_gain=0.20, win=200):
    """Decode ROT47 paragraphs in place. Returns (text, changed).

    Decoding is decided in windows, not whole articles. These pages interleave
    plain and encoded text — the lead paragraph plain, the body encoded, often
    on a single line — so a whole-article test averages the two out and finds
    nothing. A window is decoded only when doing so clearly raises its share of
    alphabetic characters; ordinary punctuation-heavy prose never clears that
    bar, and already-decoded text cannot be re-encoded by a later pass because
    re-encoding would lower the share, not raise it.

    Two passes run, the second offset by half a window, so a plain/encoded
    transition falling inside a window is caught on the second look."""
    text = str(text)
    if not ROT47_HINT.search(text):
        return text, False
    out, changed = [], False
    for line in text.split("\n"):
        pieces = []
        for seg in _ROT47_SEG.split(line):
            # A segment that opens with the encoded "<p>" marker is an encoded
            # paragraph by construction, so it is decoded whatever its length —
            # several are short one-liners ("kAmxE H2D?U8217^E 7F? @C 7F??J]" is
            # "<p>It wasn't fun or funny.") that a length floor would skip.
            if seg.startswith("kAm"):
                dec = rot47(seg)
                if _alpha_share(dec) > _alpha_share(seg):
                    pieces.append(dec)
                    changed = True
                    continue
            if len(seg) < 40:
                pieces.append(seg)
                continue
            for offset in (0, win // 2):
                seg, ch = _decode_windows(seg, win, offset, min_gain)
                changed |= ch
            pieces.append(seg)
        joined = "".join(pieces)
        out.append(_HTML_TAG.sub(" ", joined) if changed else joined)
    return "\n".join(out), changed
