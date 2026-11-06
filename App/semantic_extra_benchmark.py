"""Optional real-model CPU/RSS benchmark after one-time setup; no product data required.

Use: python -m App.semantic_extra_benchmark --model rubert --catalog-size 3563
"""
import argparse
import os
from time import perf_counter

from App.matching.algorithms.semantic.extra_runtime import get_encoder, SPECS, NameVectorCache


def rss_mb():
    try:
        import psutil
    except ImportError:
        return None
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024**2), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('rubert', 'minilm'), default='rubert')
    parser.add_argument('--catalog-size', type=int, default=3563)
    args = parser.parse_args()
    spec = SPECS['semantic_rubert_12' if args.model == 'rubert' else 'semantic_minilm_13']
    print(f'Before model initialization: RSS MB={rss_mb()}')
    started = perf_counter()
    encoder = get_encoder(spec)
    print(f'After model initialization: RSS MB={rss_mb()}, elapsed={perf_counter()-started:.2f}s')
    samples = [f'Товар строительный болт саморез крепеж №{i} размер {i % 20 + 1} мм'
               for i in range(args.catalog_size)]
    # This generates a synthetic indexed corpus; first run includes vector cache preparation.
    started = perf_counter()
    cache = NameVectorCache(spec, encoder)
    matrix = cache.build_matrix(samples, encoder)
    print(f'Catalog embeddings: shape={matrix.shape}, elapsed={perf_counter()-started:.2f}s, RSS MB={rss_mb()}')
    started = perf_counter()
    query = encoder.encode(['Саморез строительный 8 мм'])
    similarities = query @ matrix.T
    print(f'One query: elapsed={1000*(perf_counter()-started):.2f}ms, '
          f'best={int(similarities.argmax())}, RSS MB={rss_mb()}')
    if rss_mb() is None:
        print('Install psutil optionally to measure process RSS: python -m pip install psutil')


if __name__ == '__main__':
    main()
