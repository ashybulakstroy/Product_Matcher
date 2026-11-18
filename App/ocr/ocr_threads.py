"""Thread settings for the web OCR pipeline."""
import os
import threading
from pathlib import Path

import cv2
from dotenv import dotenv_values

_model_lock = threading.Lock()
_env_path = Path(__file__).resolve().parents[2] / '.env'


def _thread_count(name):
    value = os.environ.get(name, dotenv_values(_env_path, encoding='utf-8-sig').get(name))
    if value is None or not str(value).strip():
        return None
    try:
        count = int(value)
    except ValueError as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if count < 1:
        raise ValueError(f'{name} must be a positive integer')
    return count


def configure_opencv():
    count = _thread_count('OCR_OPENCV_THREADS')
    if count is not None:
        cv2.setNumThreads(count)


def create_rapidocr(factory, **kwargs):
    count = _thread_count('OCR_ONNX_THREADS')
    if count is None:
        return factory(**kwargs)
    # RapidOCR 1.2.x creates SessionOptions internally and ignores thread kwargs.
    # Restore the factory even if model initialization fails.
    import rapidocr_onnxruntime.utils as utils
    with _model_lock:
        original = utils.SessionOptions

        def session_options():
            options = original()
            options.intra_op_num_threads = count
            options.inter_op_num_threads = 1
            return options

        utils.SessionOptions = session_options
        try:
            return factory(**kwargs)
        finally:
            utils.SessionOptions = original
