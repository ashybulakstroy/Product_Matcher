"""Synthetic tests: no download, no GPU, no actual ONNX model needed.

Run from Product_Matcher repository root: python -m pytest App/tests/test_semantic_extra_12_13.py -q
"""
import numpy as np
import pytest

from App.matching.algorithms.semantic import extra_runtime as engine
from App.matching.algorithms.semantic.rubert_12 import match_products_semantic_rubert_12
from App.matching.algorithms.semantic.minilm_13 import match_products_semantic_minilm_13


class FakeEncoder:
    def __init__(self, spec):
        self.spec = spec
        self.signature = f'fake:{spec.algorithm_id}:version1'
        self.calls = []

    def encode(self, names):
        self.calls.append(list(names))
        vectors = np.zeros((len(names), self.spec.dim), np.float32)
        for idx, name in enumerate(names):
            if 'саморез' in name.lower() or 'шуруп' in name.lower():
                vectors[idx, 0] = 1
            elif 'труба' in name.lower():
                vectors[idx, 1] = 1
            elif name:
                vectors[idx, 2] = 1
        return vectors


@pytest.fixture()
def fake_models(tmp_path, monkeypatch):
    instances = {}
    monkeypatch.setattr(engine, 'CACHE_ROOT', tmp_path)
    monkeypatch.setattr(engine, 'get_encoder', lambda spec: instances.setdefault(spec.algorithm_id, FakeEncoder(spec)))
    engine._catalogs.clear()
    engine.features.cache_clear()
    yield instances
    engine._catalogs.clear()
    engine.features.cache_clear()


@pytest.mark.parametrize('algorithm,runner,dim', [
    ('semantic_rubert_12', match_products_semantic_rubert_12, 312),
    ('semantic_minilm_13', match_products_semantic_minilm_13, 384),
])
def test_produces_ranked_candidates_preserves_1c_zero_codes(fake_models, algorithm, runner, dim):
    checked = [{'id': 'T-1', 'name': 'Саморез кровельный'}]
    stock = [
        {'id': '00000000456', 'name': 'Труба ПВХ'},
        {'id': '00000000012', 'name': 'Саморез кровельный'},
        {'id': '00000000013', 'name': 'Шуруп монтажный'},
    ]
    original = [dict(row) for row in stock]
    result = runner(checked, stock, top_n=2)
    assert result[0]['position_no'] == 1
    assert result[0]['checked_id'] == 'T-1'
    assert result[0]['candidates'][0]['stock_id'] == '00000000012'
    assert all(isinstance(candidate['stock_id'], str) for candidate in result[0]['candidates'])
    assert stock == original
    assert fake_models[algorithm].spec.dim == dim


def test_cache_reuses_vectors_if_reordered_or_price_stock_changes(fake_models):
    stock = [{'id': '00001', 'name': 'Саморез'},{'id': '00002', 'name': 'Труба'}]
    query = [{'id': 'T', 'name': 'Саморез'}]
    match_products_semantic_rubert_12(query, stock)
    encoder = fake_models['semantic_rubert_12']
    assert encoder.calls[0] == ['Саморез', 'Труба']
    assert encoder.calls[1] == ['Саморез']
    updated = [dict(stock[1], price=900, quantity=1), dict(stock[0], price=500, quantity=20)]
    match_products_semantic_rubert_12(query, updated)
    assert len(encoder.calls) == 3  # one query, no catalog preparation
    assert encoder.calls[2] == ['Саморез']
    updated.append({'id': '00003', 'name': 'Краска акриловая'})
    match_products_semantic_rubert_12(query, updated)
    assert encoder.calls[3] == ['Краска акриловая']  # only new name encoded
    assert encoder.calls[4] == ['Саморез']


def test_no_model_load_when_empty_catalog_or_empty_query(monkeypatch):
    def bad_encoder(spec):
        raise AssertionError('Must not load model')
    monkeypatch.setattr(engine, 'get_encoder', bad_encoder)
    assert match_products_semantic_minilm_13([{'name': 'Труба'}], []) == [
        {'position_no':1, 'checked_id':'', 'checked_name':'Труба','candidates':[]}]
    assert match_products_semantic_minilm_13([{'name': ''}], [{'name': 'Труба'}])[0]['candidates'] == []


def test_unique_results_multiple_positions_do_not_mix(fake_models):
    result = match_products_semantic_minilm_13([
        {'id': 'T1', 'name': 'Саморез'},
        {'id': 'T2', 'name': 'Труба'},
    ], [{'id': '0001', 'name': 'Труба'}, {'id': '0002', 'name': 'Саморез'}])
    assert len(result) == 2
    assert result[0]['candidates'][0]['stock_id'] == '0002'
    assert result[1]['candidates'][0]['stock_id'] == '0001'


def test_single_no_candidate_not_fabricated(fake_models):
    result = match_products_semantic_rubert_12([{'name': 'Саморез'}], [{'id': '0001', 'name': ''}])
    assert result[0]['candidates'] == []


def test_two_model_indexes_are_independent(fake_models):
    stock = [{'id': '01', 'name': 'Саморез'}]
    match_products_semantic_rubert_12([{'name':'Саморез'}], stock)
    match_products_semantic_minilm_13([{'name':'Саморез'}], stock)
    assert set(fake_models) == {'semantic_rubert_12', 'semantic_minilm_13'}
    assert all((engine.CACHE_ROOT / key / 'names.npz').exists() for key in fake_models)


def test_cache_survives_restart_without_reencoding_catalog(fake_models):
    stock = [{'name':'Саморез'}, {'name':'Труба'}]
    query = [{'name': 'Саморез'}]
    match_products_semantic_rubert_12(query, stock)
    engine._catalogs.clear()
    fake_models['semantic_rubert_12'].calls.clear()
    match_products_semantic_rubert_12(query, stock)
    assert fake_models['semantic_rubert_12'].calls == [['Саморез']]


def test_version_change_invalidates_vectors(fake_models):
    stock = [{'name':'Труба'}]
    query = [{'name': 'Труба'}]
    match_products_semantic_minilm_13(query, stock)
    encoder = fake_models['semantic_minilm_13']
    encoder.calls.clear()
    encoder.signature = 'version2'
    match_products_semantic_minilm_13(query, stock)
    assert encoder.calls == [['Труба'], ['Труба']]


def test_registry_patch_is_idempotent():
    # Script is in the experiment root when copied; does not alter repo in this test.
    from pathlib import Path
    import importlib.util
    helper = Path(__file__).resolve().parents[2] / 'integration' / 'install.py'
    if not helper.exists():
        pytest.skip('Integration installer is not distributed inside target repo')
    spec = importlib.util.spec_from_file_location('extra_installer', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = "ALGORITHMS = {\n    'foo': AlgorithmSpec(id='foo'),\n}\n\nALGORITHM_REGISTRY = {}\n"
    once = module.patch_registry(original)
    assert "'semantic_rubert_12': AlgorithmSpec(" in once
    assert "'semantic_minilm_13': AlgorithmSpec(" in once
    assert module.patch_registry(once) == once
