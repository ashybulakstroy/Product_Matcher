"""Explicit one-time preparation; NEVER imported by the search runtime.

Usage:
    python -m App.semantic_extra_setup --model rubert
    python -m App.semantic_extra_setup --model minilm
    python -m App.semantic_extra_setup --model all
    python -m App.semantic_extra_setup --model all --self-test
"""
import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from App.matching.algorithms.semantic.extra_runtime import SPECS, model_directory

RUBERT_SOURCE = 'cointegrated/rubert-tiny2'
RUBERT_REVISION = 'e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae'  # verified upstream commit, not a moving 'main'
MINILM_SOURCE = 'Xenova/paraphrase-multilingual-MiniLM-L12-v2'
MINILM_REVISION = '2c4055b12046f11709e9df2c122e59ffbdc2f900'  # verified commit with quantized ONNX weights
MINILM_FILES = {
    'onnx/model_quantized.onnx': ('model.onnx',
        '66fc00f5f29afcaff34092e1bdd20008ca3918265a82fb9695a551e510cc4ebc'),
    'tokenizer.json': ('tokenizer.json',
        'b60b6b43406a48bf3638526314f3d232d97058bc93472ff2de930d43686fa441'),
}


def digest(path: Path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def manifest(path: Path, algorithm: str, origin: str, revision: str):
    content = dict(algorithm_id=algorithm, source=origin, revision=revision,
                   onnx_sha256=digest(path / 'model.onnx'),
                   tokenizer_sha256=digest(path / 'tokenizer.json'))
    temporary = path / 'manifest.json.part'
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path / 'manifest.json')
    return content


def _download(url, dest: Path, sha256: str):
    if dest.is_file() and digest(dest) == sha256:
        print(f'Already verified: {dest.name}', flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + '.part')
    try:
        print(f'Downloading {dest.name}...', flush=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'Torgsoft-Matching-Semantic-Setup/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as file:
            while block := response.read(1024 * 1024):
                file.write(block)
        if digest(temporary) != sha256:
            raise ValueError(f'Контрольная сумма не совпала: {dest.name}')
        os.replace(temporary, dest)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_minilm():
    folder = model_directory('semantic_minilm_13')
    for remote, (target, sha) in MINILM_FILES.items():
        url = f'https://huggingface.co/{MINILM_SOURCE}/resolve/{MINILM_REVISION}/{remote}'
        _download(url, folder / target, sha)
    manifest(folder, 'semantic_minilm_13', MINILM_SOURCE, MINILM_REVISION)
    print('semantic_minilm_13 is prepared and checksummed', flush=True)


def prepare_rubert():
    """Export OFFICIAL cointegrated checkpoint to ONNX, not an unverified mirror.

    Installer only: requires torch, transformers, onnx, safetensors, huggingface_hub.
    They are NOT needed at application runtime.
    """
    from App.matching.algorithms.semantic.extra_runtime import ROOT
    os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
    folder = model_directory('semantic_rubert_12')
    folder.mkdir(parents=True, exist_ok=True)
    from App.matching.algorithms.semantic.extra_runtime import model_ready
    if model_ready('semantic_rubert_12'):
        try:
            from App.matching.algorithms.semantic.extra_runtime import _manifest
            _manifest(folder, SPECS['semantic_rubert_12'])
            print('RuBERT ONNX already verified', flush=True)
            return
        except Exception:
            pass
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Для однократного экспорта RuBERT установите зависимости из '
                           'requirements-setup-rubert.txt') from exc
    print('Downloading official RuBERT weights and exporting to ONNX on CPU...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(RUBERT_SOURCE, revision=RUBERT_REVISION, use_fast=True,
                                              trust_remote_code=False)
    model = AutoModel.from_pretrained(RUBERT_SOURCE, revision=RUBERT_REVISION, use_safetensors=True,
                                     trust_remote_code=False, attn_implementation='eager').eval().cpu()
    tokenizer.save_pretrained(folder)
    if not (folder / 'tokenizer.json').is_file():
        raise RuntimeError('FastTokenizer не создал tokenizer.json')

    class EncoderOnly(torch.nn.Module):
        def __init__(self, bert):
            super().__init__()
            self.bert = bert

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.bert(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids).last_hidden_state

    sample = tokenizer('саморез кровельный 4,8x35', return_tensors='pt')
    ids = sample['input_ids']
    mask = sample['attention_mask']
    type_ids = sample.get('token_type_ids', torch.zeros_like(ids))
    temporary = folder / 'model.pending.onnx'
    try:
        extra = {}
        import inspect
        if 'dynamo' in inspect.signature(torch.onnx.export).parameters:
            extra['dynamo'] = False
        with torch.no_grad():
            torch.onnx.export(EncoderOnly(model), (ids, mask, type_ids), str(temporary),
                              input_names=['input_ids', 'attention_mask', 'token_type_ids'],
                              output_names=['last_hidden_state'], opset_version=17,
                              dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'},
                                            'attention_mask': {0: 'batch', 1: 'sequence'},
                                            'token_type_ids': {0: 'batch', 1: 'sequence'},
                                            'last_hidden_state': {0: 'batch', 1: 'sequence'}},
                              **extra)
        import onnxruntime as ort
        check_session = ort.InferenceSession(str(temporary), providers=['CPUExecutionProvider'])
        input_keys = {item.name for item in check_session.get_inputs()}
        data = {'input_ids': ids.numpy(), 'attention_mask': mask.numpy(),
                'token_type_ids': type_ids.numpy()}
        output = check_session.run(None, {key: data[key] for key in input_keys})[0]
        if output.ndim != 3 or output.shape[2] != 312:
            raise RuntimeError(f'Unexpected RuBERT output shape: {output.shape}')
        os.replace(temporary, folder / 'model.onnx')
        manifest(folder, 'semantic_rubert_12', RUBERT_SOURCE, RUBERT_REVISION)
        print('semantic_rubert_12 ONNX is prepared and verified', flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def self_test(model):
    from App.matching.algorithms.semantic.extra_runtime import get_encoder
    from App.matching.algorithms.semantic.model import SemanticUnavailable
    import numpy as np
    requested = ['rubert', 'minilm'] if model == 'all' else [model]
    for name in requested:
        spec = SPECS['semantic_rubert_12' if name == 'rubert' else 'semantic_minilm_13']
        encoder = get_encoder(spec)
        vectors = encoder.encode(['болгарка ушм 125 мм', 'угловая шлифовальная машина 125 мм'])
        assert vectors.shape == (2, spec.dim)
        assert np.isfinite(vectors).all()
        print(f'{spec.algorithm_id}: verified dimension={spec.dim}, similarity={float(vectors[0] @ vectors[1]):.4f}')


def main():
    parser = argparse.ArgumentParser(description='Explicit offline ONNX model preparation for semantic 12/13')
    parser.add_argument('--model', choices=['rubert', 'minilm', 'all'], default='rubert')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.model in ('rubert', 'all'):
        prepare_rubert()
    if args.model in ('minilm', 'all'):
        prepare_minilm()
    if args.self_test:
        self_test(args.model)


if __name__ == '__main__':
    main()
