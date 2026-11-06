"""Shared attribute extraction and penalties; scores remain algorithm-specific."""
import re
import unicodedata
from decimal import Decimal


_SQL3_UNITS = {
    "мм": ("length", 1), "см": ("length", 10), "м": ("length", 1000),
    "г": ("mass", 1), "кг": ("mass", 1000),
    "мл": ("volume", 1), "л": ("volume", 1000),
}


def _sql3_features(name):
    from decimal import Decimal
    text = unicodedata.normalize("NFKC", str(name or "")).lower().replace("ё", "е")
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    attributes = {}

    def measure(match):
        dimension, factor = _SQL3_UNITS[match[2]]
        attributes.setdefault(dimension, set()).add(Decimal(match[1]) * factor)
        return " "

    text = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(мм|см|мл|кг|г|л|м)(?![а-яa-z])", measure, text)
    # Preserve model identifiers such as МР-3, МР 3, МР3 as one token.
    text = re.sub(r"([а-яa-z]+)[-\s]+(\d)", r"\1\2", text)
    tokens = re.findall(r"[а-яa-z0-9]+(?:\.[0-9]+)?", text)
    stop = {"и", "или", "для", "с", "со", "по", "из", "на", "в", "во", "шт", "товар", "изделие"}
    words = tuple(dict.fromkeys(t for t in tokens if t not in stop and not any(c.isdigit() for c in t) and len(t) > 2))
    codes = set(t for t in tokens if any(c.isdigit() for c in t))
    return words, attributes, codes


BRAND_ALIASES = {
    'зубр': 'zubr', 'zubr': 'zubr', 'вихрь': 'vihr', 'vihr': 'vihr',
    'сибртех': 'sibrtech', 'sibrtech': 'sibrtech', 'эксперт': 'expert', 'expert': 'expert',
    'tytan': 'tytan', 'титан': 'tytan', 'jakko': 'jakko', 'vetonit': 'vetonit',
    'makita': 'makita', 'bosch': 'bosch', 'stayer': 'stayer', 'matrix': 'matrix',
    'kraftool': 'kraftool', 'stanley': 'stanley', 'hammer': 'hammer', 'huter': 'huter',
    'dewalt': 'dewalt', 'metabo': 'metabo', 'patriot': 'patriot', 'grinda': 'grinda',
}


COLOR_ROOTS = {
    'бел': 'белый', 'черн': 'черный', 'красн': 'красный', 'желт': 'желтый',
    'зелен': 'зеленый', 'син': 'синий', 'голуб': 'голубой', 'сер': 'серый',
    'коричнев': 'коричневый', 'бежев': 'бежевый', 'оранжев': 'оранжевый',
    'розов': 'розовый', 'фиолетов': 'фиолетовый', 'прозрачн': 'прозрачный',
}


COLOR_ENDINGS = {'ый', 'ий', 'ой', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие', 'ого', 'ой', 'ую'}


TYPE_FAMILIES = {
    'краска': ('краска', 'краски'), 'краскопульт': ('краскопульт', 'краскопульты'),
    'маркер': ('маркер', 'маркеры'), 'кисть': ('кисть', 'кисти'),
    'щетка': ('щетка', 'щетки'), 'клей': ('клей', 'клеи'),
    'герметик': ('герметик', 'герметики'), 'пена': ('пена', 'пены'),
    'болт': ('болт', 'болты'), 'гайка': ('гайка', 'гайки'),
    'саморез': ('саморез', 'саморезы'), 'шуруп': ('шуруп', 'шурупы'),
    'электрод': ('электрод', 'электроды'), 'тачка': ('тачка', 'тачки'),
    'труба': ('труба', 'трубы'), 'муфта': ('муфта', 'муфты'),
    'сверло': ('сверло', 'сверла'), 'бур': ('бур', 'буры'),
}


TYPE_LOOKUP = {word: kind for kind, words in TYPE_FAMILIES.items() for word in words}


def _features(name):
    text = unicodedata.normalize('NFKC', str(name or '')).lower().replace('ё', 'е')
    text = re.sub(r'(?<=\d),(?=\d)', '.', text)
    tokens = re.findall(r'[а-яa-z]+', text)
    colors = set()
    for token in tokens:
        for root, color in COLOR_ROOTS.items():
            if token.startswith(root) and token[len(root):] in COLOR_ENDINGS:
                colors.add(color)
    brands = {BRAND_ALIASES[t] for t in tokens if t in BRAND_ALIASES}
    kind = next((TYPE_LOOKUP[t] for t in tokens if t in TYPE_LOOKUP), None)
    # Preserve the complete identifier, including suffixes: 06325-2 != 06325-3.
    articles = set(re.findall(r'(?<!\w)\d{3,}(?:-\d+)+(?!\w)', text))
    without_articles = text
    for article in articles:
        without_articles = without_articles.replace(article, ' ')
    models = set()
    model_text = re.sub(r'(?<![\w.])\d+(?:\.\d+)?\s*(?:мм|см|мл|кг|г|л|м)(?![а-яa-z])', ' ', without_articles)
    for token in re.findall(r'(?<!\w)(?:[а-яa-z]+-?\d+|[а-яa-z]{1,2} \d+)(?:[.-]\d+)*(?!\w)', model_text):
        models.add(re.sub(r'[- ]', '', token))
    # Latin model series (HYDRO FIX / CLASSIC FIX) excluding known brands and units.
    models.update(t for t in re.findall(r'\b[a-z]{2,}\b', text)
                  if t not in BRAND_ALIASES and t not in {'mm', 'cm', 'kg', 'ml', 'pcs', 'pc', 'set', 'unit'})
    # Retain order and the complete tuple of dimensions, not just the final number.
    dimensions = set()
    dimension_pattern = r'(?<![\w.])(\d+(?:\.\d+)?(?:\s*[xх×]\s*\d+(?:\.\d+)?)+)\s*(мм|см|м)?(?![а-яa-z])'

    def dimension(match):
        factor = {'мм': 1, 'см': 10, 'м': 1000, None: 1}[match[2]]
        values = tuple(Decimal(v.strip()) * factor for v in re.split(r'[xх×]', match[1]))
        dimensions.add((bool(match[2]), values))
        return ' '

    without_articles = re.sub(dimension_pattern, dimension, without_articles)
    lexical, measures, numbers = _sql3_features(without_articles)
    # Color and brand are scored separately, so grammatical forms do not hurt lexical recall.
    excluded = set(BRAND_ALIASES) | {t for t in tokens if any(
        t.startswith(root) and t[len(root):] in COLOR_ENDINGS for root in COLOR_ROOTS)}
    words = tuple(dict.fromkeys(TYPE_LOOKUP.get(t, t) for t in lexical if t not in excluded))
    return dict(words=words, measures=measures, numbers=numbers, colors=colors,
                brands=brands, kind={kind} if kind else set(), articles=articles,
                models=models, dimensions=dimensions)


def apply_attribute_penalties(query, target, score, reasons):
    """Shared characteristic checks for lexical and semantic relevance."""
    for key, label, conflict in (
        ('kind', 'Вид товара', .2), ('articles', 'Артикул', .2),
        ('models', 'Модель/серия', .4), ('dimensions', 'Размеры', .35),
        ('colors', 'Цвет', .55), ('brands', 'Марка', .7),
    ):
        expected, actual = query[key], target[key]
        if not expected:
            continue
        if not actual:
            score *= .9
            reasons.append(f'{label}: не указан')
        elif expected <= actual:
            reasons.append(f'{label}: совпадает')
        else:
            score *= conflict
            reasons.append(f'{label}: отличается')
    for dimension, values in query['measures'].items():
        actual = target['measures'].get(dimension)
        label = {'mass': 'Масса', 'volume': 'Объём', 'length': 'Размер'}[dimension]
        if actual is None:
            score *= .9
            reasons.append(f'{label}: не указан')
        elif values == actual:
            reasons.append(f'{label}: совпадает')
        else:
            score *= .35
            reasons.append(f'{label}: отличается')
    # Bare numbers still matter, but are weaker than a recognized size or article.
    if query['numbers'] and not query['numbers'] <= target['numbers']:
        score *= .75 if not target['numbers'] else .6
        reasons.append('Числовые признаки: отличаются или отсутствуют')
    return round(score, 2), reasons
