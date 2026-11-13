"""Single registry for the engine, HTTP API and test-page controls."""
from App.matching.contracts import AlgorithmSpec

ALGORITHMS = {
    'copilot_1': AlgorithmSpec(
        id='copilot_1',
        name='copilot_1',
        title='Copilot 1',
        description='Стандартный алгоритм сопоставления по нормализации и сходству RapidFuzz.',
        module='App.matching.algorithms.fuzzy.copilot_1',
        function='match_products',
        execution='cpu',
    ),
    'sql_str_2': AlgorithmSpec(
        id='sql_str_2',
        name='sql_str_2',
        title='SQL STR 2',
        description='Строгий SQL-подобный поиск по смысловым словам: вхождение ключевых слов, без артикулов и цифрового шума.',
        module='App.matching.algorithms.lexical.sql_str_2',
        function='match_products_sql_str_2',
        execution='cpu',
    ),
    'sql_str_3': AlgorithmSpec(
        id='sql_str_3',
        name='sql_str_3',
        title='SQL STR 3',
        description='Взвешенные слова, строгие совпадения, размеры, масса, объём и модели.',
        module='App.matching.algorithms.lexical.sql_str_3',
        function='match_products_sql_str_3',
        execution='cpu',
    ),
    'opencode_4': AlgorithmSpec(
        id='opencode_4',
        name='opencode_4',
        title='OpenCode 4',
        description='Вид товара, артикул, модель, размеры, цвет и марка по идеям анализа ошибок OpenCode.',
        module='App.matching.algorithms.hybrid.opencode_4',
        function='match_products_opencode_4',
        execution='cpu',
    ),
    'semantic_5': AlgorithmSpec(
        id='semantic_5',
        name='semantic_5',
        title='Semantic 5',
        description='Локальный смысловой поиск multilingual-e5-small с проверкой характеристик. Без SQL.',
        module='App.matching.algorithms.semantic.e5',
        function='match_products_semantic_5',
        execution='shared_model',
    ),
    'exact_normalized': AlgorithmSpec(
        id='exact_normalized', name='exact_normalized', title='Exact / Normalized Match',
        description='Только точное равенство после нормализации CPU Matching-6; без совпадения кандидатов нет.',
        module='App.matching.algorithms.fuzzy.exact_normalized',
        function='match_products_exact_normalized',
    ),
    'token_sort': AlgorithmSpec(
        id='token_sort', name='token_sort', title='Token Sort Ratio',
        description='Самостоятельный Token Sort Ratio с нормализацией CPU Matching-6; порядок слов не важен.',
        module='App.matching.algorithms.fuzzy.token_sort',
        function='match_products_token_sort',
    ),
    'token_set': AlgorithmSpec(
        id='token_set', name='token_set', title='Token Set Ratio',
        description='Сравнение множеств слов RapidFuzz; допускает дополнительные слова в названии.',
        module='App.matching.algorithms.fuzzy.token_set',
        function='match_products_token_set',
    ),
    'levenshtein': AlgorithmSpec(
        id='levenshtein', name='levenshtein', title='Levenshtein',
        description='Нормированное расстояние Левенштейна: вставки, удаления и замены символов.',
        module='App.matching.algorithms.fuzzy.levenshtein',
        function='match_products_levenshtein',
    ),
    'jaro_winkler': AlgorithmSpec(
        id='jaro_winkler', name='jaro_winkler', title='Jaro-Winkler',
        description='Символьная близость Jaro-Winkler с учётом общего начала строки.',
        module='App.matching.algorithms.fuzzy.jaro_winkler',
        function='match_products_jaro_winkler',
    ),
    'jaccard': AlgorithmSpec(
        id='jaccard', name='jaccard', title='Jaccard',
        description='Доля пересечения множеств слов относительно их объединения; без общих слов кандидатов нет.',
        module='App.matching.algorithms.fuzzy.jaccard',
        function='match_products_jaccard',
    ),
    'semantic_rubert_12': AlgorithmSpec(
        id='semantic_rubert_12', name='semantic_rubert_12', title='RuBERT Tiny2',
        description='Русскоязычный смысловой поиск RuBERT Tiny2: CLS, 312 измерений, локальный ONNX CPU.',
        module='App.matching.algorithms.semantic.rubert_12',
        function='match_products_semantic_rubert_12', execution='shared_model',
    ),
    'semantic_minilm_13': AlgorithmSpec(
        id='semantic_minilm_13', name='semantic_minilm_13', title='Multilingual MiniLM L12',
        description='Многоязычный смысловой поиск MiniLM L12: masked mean, 384 измерения, локальный ONNX CPU.',
        module='App.matching.algorithms.semantic.minilm_13',
        function='match_products_semantic_minilm_13', execution='shared_model',
    ),
    'bm25_14': AlgorithmSpec(
        id='bm25_14', name='bm25_14', title='BM25 14',
        description='Okapi BM25: редкие термины, длина названия, проверка характеристик. CPU без API.',
        module='App.matching.algorithms.lexical.bm25_14',
        function='match_products_bm25_14', execution='cpu',
    ),
    'tfidf_15': AlgorithmSpec(
        id='tfidf_15', name='tfidf_15', title='TF-IDF 15 (char n-grams)',
        description='TF-IDF + cosine: слова и символьные 3-5 граммы, устойчивость к опечаткам. CPU.',
        module='App.matching.algorithms.lexical.tfidf_15',
        function='match_products_tfidf_15', execution='cpu',
    ),
}

ALGORITHM_REGISTRY = {key: spec.metadata() for key, spec in ALGORITHMS.items()}


def get_algorithm(algorithm_id):
    try:
        return ALGORITHMS[algorithm_id]
    except KeyError as error:
        raise ValueError(f'Неизвестный алгоритм: {algorithm_id}') from error


def select_algorithms(algorithm_id='copilot_1', algorithms=None):
    selected = list(algorithms) if algorithms is not None else (
        list(ALGORITHMS) if algorithm_id in (None, '', 'all') else [algorithm_id])
    if not selected:
        selected = list(ALGORITHMS)
    return [get_algorithm(key) for key in dict.fromkeys(selected)]
