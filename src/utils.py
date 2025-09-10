import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

_DOWNLOADED = False
def _ensure_nltk():
    global _DOWNLOADED
    if _DOWNLOADED:
        return
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    _DOWNLOADED = True

_lemmatizer = WordNetLemmatizer()
_stop = set()
def preprocess(text: str) -> str:
    _ensure_nltk()
    global _stop
    if not _stop:
        _stop = set(stopwords.words("english"))
    t = text.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    words = [w for w in t.split() if w not in _stop and len(w) > 2]
    lemmas = [_lemmatizer.lemmatize(w) for w in words]
    return " ".join(lemmas)
