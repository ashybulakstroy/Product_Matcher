"""Additional ONNX semantic algorithms 12/13. No network, no torch at runtime.

Integration target: https://github.com/ashybulakstroy/Product_Matcher
The existing shared_model executor runs these models in one dedicated thread.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Literal
from zipfile import BadZipFile

import numpy as np

from App.matching.algorithms.semantic.model import SemanticUnavailable
from App.matching.attributes import _features, apply_attribute_penalties

ROOT = Path(__file__).resolve().parents[4]
CACHE_ROOT = ROOT / '.cache' / 'semantic_extra'
CACHE_VERSION = 'dense-v1-positive-score-characteristics'


@dataclass(frozen=True)
class ExtraSpec:
    algorithm_id: str
    dim: int
    pooling: Literal['cls', 'mean']
    max_length: int = 96
    batch_size: int = 32
    pad_id: int = 0
    pad_token: str = '[PAD]'


SPECS = {
    'semantic_rubert_12': ExtraSpec('semantic_rubert_12', 312, 'cls'),
    'semantic_minilm_13': ExtraSpec('semantic_minilm_13', 384, 'mean'),
}

_model_lock = threading.RLock()
_models: OrderedDict[str, 'DenseONNX'] = OrderedDict()
_catalog_lock = threading.RLock()
_catalogs: dict[str, 'NameVectorCache'] = {}


def model_directory(algorithm_id: str) -> Path:
    return CACHE_ROOT / algorithm_id / 'model'


def model_ready(algorithm_id: str) -> bool:
    directory = model_directory(algorithm_id)
    return all((directory / name).is_file() for name in ('model.onnx', 'tokenizer.json', 'manifest.json'))


def _file_digest(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def _manifest(directory: Path, spec: ExtraSpec):
    try:
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['algorithm_id'] != spec.algorithm_id:
            raise ValueError('Wrong algorithm in model manifest')
        for file, field in (('model.onnx', 'onnx_sha256'), ('tokenizer.json', 'tokenizer_sha256')):
            if _file_digest(directory / file) != manifest[field]:
                raise ValueError(f'Checksum mismatch: {file}')
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SemanticUnavailable(f'{spec.algorithm_id}: модель не подготовлена или повреждена. '
                                  f'Выполните python -m App.semantic_extra_setup --model '
                                  f'{"rubert" if spec.dim == 312 else "minilm"}') from error


class DenseONNX:
    """One already installed ONNX Runtime model; never loads any ML weight on import."""
    def __init__(self, spec: ExtraSpec):
        self.spec = spec
        directory = model_directory(spec.algorithm_id)
        manifest = _manifest(directory, spec)
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = min(2, os.cpu_count() or 1)
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.session = ort.InferenceSession(str(directory / 'model.onnx'), sess_options=opts,
                                               providers=['CPUExecutionProvider'])
            self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
            self.tokenizer.enable_truncation(max_length=spec.max_length)
            self.tokenizer.enable_padding(pad_id=spec.pad_id, pad_token=spec.pad_token)
            outputs = self.session.get_outputs()
            matching = [v.name for v in outputs if v.name == 'last_hidden_state']
            self.output_name = matching[0] if matching else outputs[0].name
            self.input_names = {v.name for v in self.session.get_inputs()}
            if not {'input_ids', 'attention_mask'} <= self.input_names:
                raise ValueError('Unsupported ONNX encoder inputs')
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            raise SemanticUnavailable(f'{spec.algorithm_id}: ошибка загрузки локальной ONNX-модели: {error}') from error
        self.signature = f'{CACHE_VERSION}:{spec.algorithm_id}:{spec.pooling}:{spec.max_length}:' \
                         f'{manifest["onnx_sha256"]}:{manifest["tokenizer_sha256"]}'
        self.lock = threading.Lock()

    def encode(self, names: list[str]) -> np.ndarray:
        if not names:
            return np.empty((0, self.spec.dim), dtype=np.float32)
        result = []
        with self.lock:
            for start in range(0, len(names), self.spec.batch_size):
                batch = self.tokenizer.encode_batch(names[start:start + self.spec.batch_size])
                ids = np.asarray([v.ids for v in batch], dtype=np.int64)
                mask = np.asarray([v.attention_mask for v in batch], dtype=np.int64)
                tokens = np.asarray([v.type_ids for v in batch], dtype=np.int64)
                all_inputs = {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': tokens}
                feeds = {key: all_inputs[key] for key in self.input_names if key in all_inputs}
                hidden = self.session.run([self.output_name], feeds)[0]
                if hidden.ndim != 3 or hidden.shape[0] != len(batch) or hidden.shape[2] != self.spec.dim:
                    raise SemanticUnavailable(f'{self.spec.algorithm_id}: ONNX не возвращает ожидаемые '
                                              f'эмбеддинги [batch, seq, {self.spec.dim}]')
                if self.spec.pooling == 'cls':
                    vectors = hidden[:, 0, :].astype(np.float32)
                else:
                    weights = mask[..., None].astype(np.float32)
                    vectors = (hidden.astype(np.float32) * weights).sum(axis=1) / \
                              np.maximum(weights.sum(axis=1), 1)
                vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
                if not np.isfinite(vectors).all():
                    raise SemanticUnavailable(f'{self.spec.algorithm_id}: некорректные векторы')
                result.append(vectors)
        return np.concatenate(result, axis=0)


def get_encoder(spec: ExtraSpec) -> DenseONNX:
    with _model_lock:
        if spec.algorithm_id not in _models:
            _models[spec.algorithm_id] = DenseONNX(spec)
        _models.move_to_end(spec.algorithm_id)
        # OPTIONAL memory-saving mode. Never unload a model during its use:
        # all calls to get_encoder run in the existing shared_model thread (max_workers=1).
        count = min(2, max(1, int(os.environ.get('MATCHING_EXTRA_MAX_RESIDENT_MODELS', '2'))))
        while len(_models) > count:
            _models.popitem(last=False)
        return _models[spec.algorithm_id]


class NameVectorCache:
    """Disk-backed per-name embeddings; re-encode only new/changed names.

    Updating stock qty, price, code or row order does not invalidate the vectors.
    Every model has a separate NPZ snapshot and immutable signature.
    """
    def __init__(self, spec: ExtraSpec, encoder: DenseONNX):
        self.spec = spec
        self.signature = encoder.signature
        self.path = CACHE_ROOT / spec.algorithm_id / 'names.npz'
        self.values: dict[str, np.ndarray] = {}
        try:
            with np.load(self.path, allow_pickle=False) as saved:
                version = str(saved['version'].item())
                names = saved['names'].tolist()
                vectors = saved['vectors']
                if version == self.signature and vectors.shape == (len(names), spec.dim) and \
                        np.isfinite(vectors).all():
                    self.values = {name: vector for name, vector in zip(names, vectors)}
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            pass

    def build_matrix(self, names: list[str], encoder: DenseONNX) -> np.ndarray:
        missing = list(dict.fromkeys(name for name in names if name and name not in self.values))
        if missing:
            for start in range(0, len(missing), 128):
                portion = missing[start:start + 128]
                vectors = encoder.encode(portion)
                if vectors.shape != (len(portion), self.spec.dim):
                    raise SemanticUnavailable('Некорректный размер вектора после кодирования каталога')
                self.values.update(zip(portion, vectors))
            self.save()
        zero = np.zeros(self.spec.dim, dtype=np.float32)
        return np.stack([self.values.get(name, zero) for name in names]).astype(np.float32)

    def save(self):
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            names = list(self.values)
            vectors = np.stack([self.values[name] for name in names]).astype(np.float32)
            with tempfile.NamedTemporaryFile(dir=self.path.parent, suffix='.npz', delete=False) as file:
                temporary = Path(file.name)
                np.savez_compressed(file, version=np.asarray(self.signature),
                                    names=np.asarray(names, dtype=str), vectors=vectors)
            os.replace(temporary, self.path)
        except OSError:
            logging.warning('Cannot save extra semantic vector cache', exc_info=True)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink(missing_ok=True)


@lru_cache(maxsize=12000)
def features(name: str):
    return _features(name)


def _names(rows):
    return [str(row.get('name') or row.get('product_name') or '') for row in rows]


def _code(row):
    # 1C codes are opaque text. Never strip leading zeros or cast to int.
    for key in ('id', 'sku', 'code'):
        if row.get(key) is not None and str(row[key]) != '':
            return str(row[key])
    return ''


def match_dense(algorithm_id, checked_products, stock_products, top_n=5):
    spec = SPECS[algorithm_id]
    checked, stock = list(checked_products), list(stock_products)
    result = [dict(position_no=i, checked_id=_code(row),
                   checked_name=str(row.get('name') or row.get('product_name') or ''), candidates=[])
              for i, row in enumerate(checked, 1)]
    valid = [i for i, row in enumerate(result) if row['checked_name'].strip()]
    if not valid or not stock:
        return result
    try:
        # The worker is serial for shared_model: no overlapping encode+evict.
        with _catalog_lock:
            encoder = get_encoder(spec)
            stored = _catalogs.get(algorithm_id)
            if stored is None or stored.signature != encoder.signature:
                stored = NameVectorCache(spec, encoder)
                _catalogs[algorithm_id] = stored
            stock_names = _names(stock)
            vectors = stored.build_matrix(stock_names, encoder)
            query_names = [result[i]['checked_name'] for i in valid]
            query_vectors = encoder.encode(query_names)
            similarities = np.clip(query_vectors @ vectors.T, 0, 1)
        for query_index, row_index in enumerate(valid):
            row = result[row_index]
            query_name = row['checked_name']
            query_features = features(query_name)
            candidates = []
            for index, target in enumerate(stock):
                score = float(similarities[query_index, index] * 100)
                if score <= 0:
                    continue
                reasons = [f'{algorithm_id}: косинусная близость {score:.1f}%']
                score, reasons = apply_attribute_penalties(query_features, features(stock_names[index]),
                                                           score, reasons)
                if score > 0:
                    candidates.append(dict(stock_id=_code(target), stock_name=stock_names[index],
                                           similarity_pct=round(score, 2),
                                           semantic_similarity_pct=round(float(similarities[query_index, index] * 100), 2),
                                           match_reasons=reasons))
            candidates.sort(key=lambda c: (-c['similarity_pct'], str(c['stock_id'])))
            row['candidates'] = candidates[:max(0, top_n)]
        return result
    except SemanticUnavailable:
        raise
    except Exception as error:
        logging.exception('Extra semantic algorithm failed: %s', algorithm_id)
        raise SemanticUnavailable(f'{algorithm_id}: ошибка смыслового поиска. Проверьте журнал сервера.') from error
