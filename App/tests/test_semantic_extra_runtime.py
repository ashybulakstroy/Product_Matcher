import threading
from types import SimpleNamespace

import numpy as np
import pytest

from App.matching.algorithms.semantic import extra_runtime as runtime
from App.matching.algorithms.semantic.model import SemanticUnavailable
from App.matching.engine import match


@pytest.mark.parametrize('pooling,expected', [('cls', [1, 0]), ('mean', [.6, .8])])
def test_pooling_uses_cls_or_masked_mean_and_normalizes(pooling, expected):
    encoder = runtime.DenseONNX.__new__(runtime.DenseONNX)
    encoder.spec = runtime.ExtraSpec('test', dim=2, pooling=pooling)
    encoder.lock = threading.Lock()
    encoder.input_names = {'input_ids', 'attention_mask'}
    encoder.output_name = 'last_hidden_state'
    encoder.tokenizer = SimpleNamespace(encode_batch=lambda names: [
        SimpleNamespace(ids=[1, 2, 0], attention_mask=[1, 1, 0], type_ids=[0, 0, 0])])
    encoder.session = SimpleNamespace(run=lambda outputs, feeds:
        [np.array([[[3, 0], [0, 4], [1000, 1000]]], dtype=np.float32)])
    vector = encoder.encode(['test'])
    assert np.allclose(vector[0], expected)
    assert np.isclose(np.linalg.norm(vector[0]), 1)


def test_missing_models_are_visible_without_breaking_other_results(monkeypatch):
    def unavailable(spec):
        raise SemanticUnavailable('Model not prepared: ' + spec.algorithm_id)
    monkeypatch.setattr(runtime, 'get_encoder', unavailable)
    result = match([{'name': 'болт'}], [{'id': '0001', 'name': 'болт', 'quantity': 10}],
        algorithms=['exact_normalized', 'semantic_rubert_12', 'semantic_minilm_13'], min_stock=3)
    assert result['results_by_algorithm'][0]['best_candidate']['stock_id'] == '0001'
    assert [row['status'] for row in result['results_by_algorithm'][1:]] == ['unavailable', 'unavailable']
    assert result['voting']['positions'][0]['voting_algorithms'] == 1


def test_prepared_manifest_is_reused_without_export_dependencies(tmp_path, monkeypatch):
    from App import semantic_extra_setup as setup
    folder = tmp_path / 'rubert'
    folder.mkdir()
    (folder / 'model.onnx').write_bytes(b'onnx')
    (folder / 'tokenizer.json').write_bytes(b'tokenizer')
    monkeypatch.setattr(runtime, 'CACHE_ROOT', tmp_path)
    monkeypatch.setattr(setup, 'model_directory', lambda identifier: folder)
    monkeypatch.setattr(runtime, 'model_ready', lambda identifier: True)
    setup.manifest(folder, 'semantic_rubert_12', 'test', 'test')
    setup.prepare_rubert()
