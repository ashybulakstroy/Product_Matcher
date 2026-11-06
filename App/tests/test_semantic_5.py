import numpy as np
import pytest

from App.services import semantic_matching as semantic
from App.services.semantic_model import SemanticUnavailable
from App.web_debug import match_debug_payload


class FakeEncoder:
    """Controlled vectors exercise ranking/cache independently of model quality."""
    def __init__(self):
        self.calls = []

    def encode(self, texts, prefix, progress=None):
        self.calls.append((list(texts), prefix))
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for index, text in enumerate(texts):
            # Synonyms have identical controlled vectors; all others are orthogonal.
            vectors[index, 0 if any(t in text.lower() for t in ('болгарка', 'шлифовальная', 'сверло')) else 1] = 1
        return vectors


@pytest.fixture
def encoder(monkeypatch, tmp_path):
    fake = FakeEncoder()
    monkeypatch.setattr(semantic, '_encoder', fake)
    monkeypatch.setattr(semantic, '_catalogs', semantic.OrderedDict())
    original = semantic.prepare_catalog
    monkeypatch.setattr(semantic, 'prepare_catalog', lambda stock, progress=None:
                        original(stock, progress=progress, cache_dir=tmp_path))
    return fake, tmp_path


def test_semantic_candidate_does_not_require_common_words(encoder):
    result = semantic.match_products_semantic_5([{'name': 'Болгарка'}],
        [{'id': 'a', 'name': 'Краска'}, {'id': 'b', 'name': 'Угловая шлифовальная машина'}])
    assert result[0]['candidates'][0]['stock_id'] == 'b'
    assert result[0]['candidates'][0]['semantic_similarity_pct'] == 100


def test_size_conflict_penalizes_semantic_match(encoder):
    candidates = semantic.match_products_semantic_5([{'name': 'Сверло 6 мм'}],
        [{'id': 'a', 'name': 'Сверло 10 мм'}, {'id': 'b', 'name': 'Сверло 6 мм'}])[0]['candidates']
    assert candidates[0]['stock_id'] == 'b'
    assert candidates[0]['similarity_pct'] > candidates[1]['similarity_pct']
    assert any('отличается' in reason for reason in candidates[1]['match_reasons'])


def test_cache_reuses_disk_and_invalidates_on_name_or_order_change(encoder):
    fake, directory = encoder
    stock = [{'name': 'Краска'}, {'name': 'Болгарка'}]
    semantic.prepare_catalog(stock)
    assert len(fake.calls) == 1
    semantic.prepare_catalog(stock)
    assert len(fake.calls) == 1
    semantic._catalogs.clear()
    semantic.prepare_catalog(stock)
    assert len(fake.calls) == 1
    semantic.prepare_catalog(stock[::-1])
    assert len(fake.calls) == 2
    semantic.prepare_catalog([{'name': 'Новое название'}])
    assert len(fake.calls) == 3
    assert len(list(directory.glob('catalog-*.npz'))) == 3


def test_corrupt_cache_is_rebuilt(encoder):
    fake, directory = encoder
    signature = semantic._signature(['Болгарка'])
    (directory / f'catalog-{signature}.npz').write_bytes(b'broken')
    semantic.prepare_catalog([{'name': 'Болгарка'}])
    assert len(fake.calls) == 1


def test_price_breaks_semantic_ties_in_shared_api(encoder):
    result = match_debug_payload([{'name': 'Болгарка', 'price': 100}],
        [{'id': 'a', 'name': 'Угловая шлифовальная машина', 'price': 200},
         {'id': 'b', 'name': 'Угловая шлифовальная машина', 'price': 101}],
        algorithm_id='semantic_5')
    assert result['best_candidate']['stock_id'] == 'b'


def test_empty_query_or_stock_needs_no_model(monkeypatch):
    def fail():
        raise AssertionError('Should not load model')
    monkeypatch.setattr(semantic, 'get_encoder', fail)
    assert not semantic.match_products_semantic_5([{'name': ''}], [{'name': 'Краска'}])[0]['candidates']
    assert not semantic.match_products_semantic_5([{'name': 'Болгарка'}], [])[0]['candidates']


def test_unavailable_model_does_not_hide_other_algorithms(monkeypatch):
    def fail():
        raise SemanticUnavailable('Модель не подготовлена')
    monkeypatch.setattr(semantic, 'get_encoder', fail)
    monkeypatch.setattr(semantic, '_catalogs', semantic.OrderedDict())
    result = match_debug_payload([{'name': 'Болгарка'}], [{'id': 's', 'name': 'Болгарка'}], algorithm_id='all')
    failed = next(item for item in result['results_by_algorithm'] if item['algorithm_id'] == 'semantic_5')
    assert failed['status'] == 'unavailable'
    assert result['best_candidate']['stock_id'] == 's'


def test_query_expansion_preserves_characteristics():
    assert semantic.expand_query('Болгарка 125 мм') == 'угловая шлифовальная машина 125 мм'
    assert semantic.expand_query('Водоэмульсионка белая 5 л') == 'краска водоэмульсионная белая 5 л'


def test_reordered_ids_and_prices_do_not_reuse_old_candidate_metadata(encoder):
    first = [{'id': 'old', 'name': 'Угловая шлифовальная машина'}]
    second = [{'id': 'new', 'name': 'Угловая шлифовальная машина'}]
    semantic.match_products_semantic_5([{'name': 'Болгарка'}], first)
    result = semantic.match_products_semantic_5([{'name': 'Болгарка'}], second)
    assert result[0]['candidates'][0]['stock_id'] == 'new'
    assert sum(prefix == 'passage: ' for _, prefix in encoder[0].calls) == 1
