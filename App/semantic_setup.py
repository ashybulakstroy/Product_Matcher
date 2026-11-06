"""Prepare local semantic search: python -m App.semantic_setup."""
from App.matching.algorithms.semantic.model import download_model


def main():
    download_model()
    from App.matching.algorithms.semantic.e5 import prepare_catalog
    from App.web_debug import load_stock_items
    stock = load_stock_items()
    if not stock:
        print('Model ready. Stock catalog is empty; the vector cache will be built on first search.')
        return
    def progress(done, total):
        print(f'Catalog embeddings: {done}/{total}', flush=True)
    prepare_catalog(stock, progress=progress)
    print('semantic_5 ready. Model and catalog vectors are stored locally.', flush=True)


if __name__ == '__main__':
    main()
