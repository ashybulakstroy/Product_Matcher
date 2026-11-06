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
