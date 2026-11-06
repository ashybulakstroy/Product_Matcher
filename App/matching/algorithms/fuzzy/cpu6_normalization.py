# Adapted from R:/Torgsoft_1C_CPU_Matching_6/matching_cpu6/normalize.py.
# Kept separate so existing algorithms retain their original normalization.
"""Safe, local, deterministic normalizations: no dictionaries or ML models.

Original product name and code are never changed, only auxiliary representations.
"""

from dataclasses import dataclass
import re
import unicodedata

_SPACES_RE = re.compile(r"\s+")
_NONWORDS_RE = re.compile(r"[^\w.]+", flags=re.UNICODE)
_NONDECIMAL_DOT_RE = re.compile(r"(?<!\d)\.|\.(?!\d)")
_DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
_SIZE_MULT_RE = re.compile(r"(?<=\d)\s*[×хx*]\s*(?=\d)")
# Conservative unit spacing; ordinary model numbers (e.g. PN20, M12) remain intact.
_NUMERIC_UNIT_RE = re.compile(
    r"(?<!\w)(\d+(?:\.\d+)?)(мм|см|вт|квт|кг|г|мл|л)(?!\w)"
)


def normalize_name(raw: str) -> str:
    """Normalize two names consistently; no external reference data needed."""
    if not isinstance(raw, str):
        raise TypeError("Товарное наименование должно быть строкой")
    text = unicodedata.normalize("NFKC", raw).casefold().replace("ё", "е")
    text = _DECIMAL_COMMA_RE.sub(".", text)
    text = _SIZE_MULT_RE.sub("x", text)
    text = _NUMERIC_UNIT_RE.sub(r"\1 \2", text)
    text = _NONWORDS_RE.sub(" ", text)
    text = _NONDECIMAL_DOT_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", text).strip()


@dataclass(frozen=True, slots=True)
class PreparedName:
    original: str
    normalized: str
    sorted_words: str
    word_set: frozenset[str]


def prepare_name(name: str) -> PreparedName:
    normalized = normalize_name(name)
    words = normalized.split()
    return PreparedName(
        original=name,
        normalized=normalized,
        sorted_words=" ".join(sorted(words)),
        word_set=frozenset(words),
    )
