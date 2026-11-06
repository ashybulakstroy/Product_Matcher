"""Real local models only: no download or export is ever performed by pytest."""
import urllib.request

import numpy as np
import pytest

from App.matching.algorithms.semantic import extra_runtime as runtime


@pytest.mark.parametrize('identifier', ['semantic_rubert_12', 'semantic_minilm_13'])
def test_prepared_model_encodes_offline_and_finds_identical_name(identifier, monkeypatch):
    if not runtime.model_ready(identifier):
        pytest.skip('Run App.semantic_extra_setup to prepare this local model')
    def no_network(*args, **kwargs):
        raise AssertionError('Search must not access the network')
    monkeypatch.setattr(urllib.request, 'urlopen', no_network)
    spec = runtime.SPECS[identifier]
    encoder = runtime.get_encoder(spec)
    vectors = encoder.encode(['Саморез кровельный 4,8x35', 'Краска белая 1 л'])
    assert vectors.shape == (2, spec.dim)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    result = runtime.match_dense(identifier, [{'name': 'Саморез кровельный 4,8x35'}],
        [{'id': '0001', 'name': 'Саморез кровельный 4,8x35'},
         {'id': '0002', 'name': 'Краска белая 1 л'}])
    assert result[0]['candidates'][0]['stock_id'] == '0001'
