import re
import unicodedata


def normalize_name(value: str) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = text.replace("%", " ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"[^0-9а-яa-zё\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
