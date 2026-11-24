"""Shared text cleaning for the sentiment project (used by the notebook and the Streamlit app)."""
import re
import simplemma

_STOP = """
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself
it its itself they them their theirs themselves what which who whom this that these those am is are was were be
been being have has had having do does did doing a an the and but if or because as until while of at by for with
about between into through during before after above below to from up down in out on off over under again further
then once here there when where why how all any both each few more most other some such own same so than too very
s t can will just should now would could also get got one even really
"""
NEGATIONS = {"not", "no", "nor", "never", "none", "nothing", "nobody", "nowhere", "neither", "cannot", "without", "hardly"}
STOP_WORDS = set(_STOP.split()) - NEGATIONS

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_HTML = re.compile(r"<[^>]+>")
_CONTRACTIONS = [
    (r"\bwon't\b", "will not"), (r"\bcan't\b", "can not"), (r"\bcannot\b", "can not"), (r"\bain't\b", "is not"),
    (r"n't\b", " not"), (r"'re\b", " are"), (r"'ll\b", " will"), (r"'ve\b", " have"), (r"'m\b", " am"), (r"'d\b", " would"),
]

# negations that are glued to the next word:  "not good" -> "not_good"
_FUSE = {"not", "no", "never", "without", "hardly", "neither", "nor"}

def _fuse_negations(tokens):
    out, i = [], 0
    while i < len(tokens):
        w = tokens[i]
        if w in _FUSE and i + 1 < len(tokens) and tokens[i + 1] not in NEGATIONS:
            out.append(w + "_" + tokens[i + 1])
            i += 2
        else:
            out.append(w)
            i += 1
    return out

def clean_text(text, lemmatize=True):
    """lowercase -> remove links/html/symbols -> expand contractions -> tokenise -> drop stop-words (keep negations) -> lemmatise."""
    if not isinstance(text, str):
        return ""
    t = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    t = _URL.sub(" ", t)
    t = _HTML.sub(" ", t)
    t = t.replace("*", " ")                      # masked words such as "**"
    for pat, rep in _CONTRACTIONS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"'s\b", " ", t)                  # possessives
    t = re.sub(r"[^a-z\s]", " ", t)              # digits, punctuation, emojis, symbols
    tokens = t.split()                           # tokenise
    tokens = [w for w in tokens if (w not in STOP_WORDS and len(w) > 1) or w in NEGATIONS]
    if lemmatize:
        tokens = [simplemma.lemmatize(w, lang="en") for w in tokens]
        tokens = [w for w in tokens if w not in STOP_WORDS]
    tokens = _fuse_negations(tokens)
    return " ".join(tokens)
