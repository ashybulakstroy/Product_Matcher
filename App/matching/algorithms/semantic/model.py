"""Pinned E5 ONNX model: explicit download once, offline inference thereafter."""
import hashlib
import os
import threading
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = ROOT / '.cache' / 'semantic_5' / 'model'
REPOSITORY = 'Xenova/multilingual-e5-small'
REVISION = '761b726dd34fb83930e26aab4e9ac3899aa1fa78'
FILES = {
    'onnx/model_quantized.onnx': 'f80102d3f2a1229f387d3c81909990d8945513e347b0eab049f7de3c6f98c193',
    'tokenizer.json': '0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39',
}
MAX_LENGTH = 128
MODEL_SIGNATURE = f'{REPOSITORY}@{REVISION}:int8:mean-l2:{MAX_LENGTH}:v1'


class SemanticUnavailable(RuntimeError):
    pass


def _sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def download_model():
    """Only this explicit setup command accesses the network; no product data is sent."""
    for remote, expected in FILES.items():
        target = MODEL_DIR / remote
        if target.is_file() and _sha(target) == expected:
            print(f'Already verified: {remote}', flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.part')
        print(f'Downloading {remote}...', flush=True)
        request = urllib.request.Request(f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{remote}',
                                         headers={'User-Agent': 'Product-Matcher-setup/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if _sha(temporary) != expected:
            raise SemanticUnavailable(f'Контрольная сумма модели не совпала: {remote}')
        os.replace(temporary, target)
        print(f'Verified: {remote} ({target.stat().st_size} bytes)', flush=True)


class E5Encoder:
    def __init__(self, directory=MODEL_DIR):
        import onnxruntime as ort
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise SemanticUnavailable('Установите зависимости: python -m pip install -r requirements.txt') from error
        directory = Path(directory)
        for remote, expected in FILES.items():
            path = directory / remote
            if not path.is_file() or _sha(path) != expected:
                raise SemanticUnavailable('Модель semantic_5 не подготовлена. Выполните: python -m App.semantic_setup')
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / 'onnx/model_quantized.onnx'),
                                           sess_options=options, providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self.tokenizer.enable_padding(pad_id=1, pad_token='<pad>')
        self.lock = threading.Lock()

    def encode(self, texts, prefix, progress=None):
        output = []
        with self.lock:
            for start in range(0, len(texts), 32):
                batch = self.tokenizer.encode_batch([prefix + text for text in texts[start:start + 32]])
                ids = np.asarray([item.ids for item in batch], dtype=np.int64)
                mask = np.asarray([item.attention_mask for item in batch], dtype=np.int64)
                values = {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': np.zeros_like(ids)}
                feeds = {item.name: values[item.name] for item in self.session.get_inputs()}
                hidden = self.session.run(None, feeds)[0]
                weights = mask[..., None].astype(np.float32)
                vectors = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1)
                vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
                output.append(vectors.astype(np.float32))
                if progress:
                    progress(min(start + len(batch), len(texts)), len(texts))
        return np.concatenate(output) if output else np.empty((0, 384), dtype=np.float32)
