"""Compatibility imports. Implementations now live in App.matching."""
from App.matching.normalization import normalize_name
from App.matching.algorithms.fuzzy.copilot_1 import _score_name, match_products
from App.matching.algorithms.lexical.sql_str_2 import _meaningful_tokens, _sql_like_score, match_products_sql_str_2
from App.matching.algorithms.lexical.sql_str_3 import match_products_sql_str_3
from App.matching.attributes import _SQL3_UNITS, _sql3_features
from App.matching.token_scoring import _sql3_compare
