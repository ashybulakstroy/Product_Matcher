import pytest
import cv2
import rapidocr_onnxruntime.utils as utils

from App.ocr import ocr_threads


def test_settings_apply_and_factory_is_restored(monkeypatch):
    monkeypatch.setenv('OCR_ONNX_THREADS', '2')
    monkeypatch.setenv('OCR_OPENCV_THREADS', '2')
    original = utils.SessionOptions
    before = cv2.getNumThreads()
    try:
        ocr_threads.configure_opencv()
        options = ocr_threads.create_rapidocr(lambda: utils.SessionOptions())
        assert cv2.getNumThreads() == 2
        assert options.intra_op_num_threads == 2
        assert options.inter_op_num_threads == 1
        assert utils.SessionOptions is original
        with pytest.raises(RuntimeError):
            ocr_threads.create_rapidocr(lambda: (_ for _ in ()).throw(RuntimeError()))
        assert utils.SessionOptions is original
    finally:
        cv2.setNumThreads(before)


@pytest.mark.parametrize('value', ['0', '-1', 'bad'])
def test_invalid_setting_rejected(monkeypatch, value):
    monkeypatch.setenv('OCR_ONNX_THREADS', value)
    with pytest.raises(ValueError, match='OCR_ONNX_THREADS'):
        ocr_threads.create_rapidocr(lambda: None)
