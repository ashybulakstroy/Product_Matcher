"""Dense semantic search using a NumPy file cache; no SQL or vector database."""
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from zipfile import BadZipFile
from collections import OrderedDict
from pathlib import Path

import numpy as np

from App.matching.attributes import _features, apply_attribute_penalties
from App.matching.algorithms.semantic.model import E5Encoder, MODEL_SIGNATURE, ROOT, SemanticUnavailable

CACHE_DIR = ROOT / '.cache' / 'semantic_5'
_lock = threading.RLock()
_encoder = None
_catalogs = OrderedDict()

# Short colloquial queries can be ambiguous even for a multilingual encoder.
# Explicit expansions are visible in the reasons, never learned from unverified pairs.
QUERY_ALIASES = (
    (r'\b(?:болгарка|болгарки|ушм)\b', 'угловая шлифовальная машина'),
    (r'\bводоэмульсионка\b', 'краска водоэмульсионная'),
    (r'\bпена\s+для\s+монтажа\b', 'пена полиуретановая монтажная'),
)


def expand_query(text):
    expanded = text
    for pattern, replacement in QUERY_ALIASES:
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded


def get_encoder():
    global _encoder
    with _lock:
        if _encoder is None:
            try:
                _encoder = E5Encoder()
            except SemanticUnavailable:
                raise
            except Exception as error:
                logging.exception('Cannot initialize semantic_5')
                raise SemanticUnavailable('Не удалось загрузить локальную модель semantic_5. Проверьте журнал сервера.') from error
        return _encoder


def _names(stock):
    return [str(item.get('name') or item.get('product_name') or '') for item in stock]


def _signature(names):
    data = json.dumps([MODEL_SIGNATURE, names], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def prepare_catalog(stock, progress=None, cache_dir=CACHE_DIR):
    names = _names(stock)
    signature = _signature(names)
    with _lock:
        if signature in _catalogs:
            _catalogs.move_to_end(signature)
            return _catalogs[signature]
        encoder = get_encoder()
        path = Path(cache_dir) / f'catalog-{signature}.npz'
        vectors = None
        try:
            with np.load(path, allow_pickle=False) as saved:
                candidate = saved['vectors']
                if (str(saved['signature'].item()) == signature and candidate.shape == (len(names), 384)
                        and np.isfinite(candidate).all()
                        and np.allclose(np.linalg.norm(candidate, axis=1), 1, atol=1e-3)):
                    vectors = candidate.astype(np.float32)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            pass
        if vectors is None:
            vectors = encoder.encode(names, 'passage: ', progress=progress)
            if vectors.shape != (len(names), 384) or not np.isfinite(vectors).all():
                raise SemanticUnavailable('Некорректный размер векторов semantic_5')
            # A read-only cache directory must not prevent searching in memory.
            temporary = None
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as output:
                    temporary = Path(output.name)
                    np.savez_compressed(output, signature=np.asarray(signature), vectors=vectors)
                os.replace(temporary, path)
            except OSError:
                logging.warning('Cannot persist semantic_5 catalog cache', exc_info=True)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink(missing_ok=True)
        features = [_features(name) for name in names]
        vectors.setflags(write=False)
        _catalogs[signature] = vectors, features
        while len(_catalogs) > 2:
            _catalogs.popitem(last=False)
        return _catalogs[signature]


def match_products_semantic_5(checked_products, stock_products, top_n=5):
    try:
        return _match_products(checked_products, stock_products, top_n)
    except SemanticUnavailable:
        raise
    except Exception as error:
        logging.exception('semantic_5 inference failed')
        raise SemanticUnavailable('Ошибка локального semantic_5. Проверьте журнал сервера; остальные алгоритмы доступны.') from error


def _match_products(checked_products, stock_products, top_n):
    checked, stock = list(checked_products), list(stock_products)
    result = [dict(position_no=i, checked_id=item.get('id') or item.get('sku') or '',
                   checked_name=str(item.get('name') or item.get('product_name') or ''), candidates=[])
              for i, item in enumerate(checked, 1)]
    valid = [i for i, row in enumerate(result) if row['checked_name'].strip()]
    if not stock or not valid:
        return result
    vectors, target_features = prepare_catalog(stock)
    expanded = [expand_query(result[i]['checked_name']) for i in valid]
    query_vectors = get_encoder().encode(expanded, 'query: ')
    similarities = query_vectors @ vectors.T
    for query_index, row_index in enumerate(valid):
        row = result[row_index]
        query_features = _features(expanded[query_index])
        for index, target in enumerate(stock):
            cosine = float(np.clip(similarities[query_index, index], 0, 1))
            if cosine <= 0:
                continue
            reasons = [f'Смысловая близость E5: {cosine * 100:.1f}%']
            if expanded[query_index] != row['checked_name']:
                reasons.append(f'Уточнение запроса: {expanded[query_index]}')
            score, reasons = apply_attribute_penalties(query_features, target_features[index], cosine * 100, reasons)
            if score > 0:
                row['candidates'].append(dict(stock_id=target.get('id') or target.get('sku') or '',
                    stock_name=str(target.get('name') or target.get('product_name') or ''),
                    similarity_pct=score, semantic_similarity_pct=round(cosine * 100, 2),
                    match_reasons=reasons))
        row['candidates'].sort(key=lambda c: (-c['similarity_pct'], str(c['stock_id'])))
        row['candidates'] = row['candidates'][:max(0, top_n)]
    return result
