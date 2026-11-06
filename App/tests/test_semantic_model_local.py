"""Optional real-model smoke check; never downloads anything during pytest."""
import pytest

from App.services.semantic_model import FILES, MODEL_DIR

pytestmark = pytest.mark.skipif(not all((MODEL_DIR / name).is_file() for name in FILES),
                                reason='Run python -m App.semantic_setup for the local E5 model')


def test_real_model_finds_meaning_without_query_aliases():
    from App.services.semantic_matching import match_products_semantic_5
    result = match_products_semantic_5(
        [{'name': 'инструмент для закручивания шурупов'}],
        [{'id': 's', 'name': 'Аккумуляторный шуруповерт'},
         {'id': 'p', 'name': 'Краска белая'}, {'id': 'c', 'name': 'Садовая тачка'}],
    )
    assert result[0]['candidates'][0]['stock_id'] == 's'
    assert 'Смысловая близость E5' in result[0]['candidates'][0]['match_reasons'][0]
