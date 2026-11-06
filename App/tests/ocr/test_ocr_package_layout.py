import importlib
from pathlib import Path

import pytest

from App.ocr.paths import PROJECT_ROOT

OCR_MODULES = [
    "contour_unwarp", "dewarp_necessity_detector", "full_grid_boundary",
    "manual_mappings", "numeric_cell_split", "numeric_recovery",
    "numeric_settings", "ocr_activity", "ocr_catalog_eval", "ocr_run",
    "quantity_cells", "receipt_editing", "receipt_math",
    "receipt_orientation_detector", "receipt_rectification",
    "rotation_rectification", "row_cell_split", "selective_number_ocr",
    "selective_price_ocr", "table_boundary", "table_headers", "visual_sync",
    "wave_dewarp", "web_pipeline", "artifact_paths", "batch_geometry_eval",
    "contour_review", "geometry_review", "grid_line_preprocessing",
    "ocr_eval", "ocr_geometry", "quantity_price_split_probe",
    "x_profile_split_probe",
]


@pytest.mark.parametrize("name", OCR_MODULES)
def test_ocr_module_imports_from_package(name):
    module = importlib.import_module(f"App.ocr.{name}")
    path = Path(module.__file__).resolve()
    assert path.parent == PROJECT_ROOT / "App" / "ocr"
    assert path.stem == name


def test_ocr_paths_point_to_project_root():
    from App.ocr import artifact_paths, manual_mappings, numeric_settings, ocr_catalog_eval, ocr_run, web_pipeline

    assert (PROJECT_ROOT / "App" / "ocr").is_dir()
    assert (PROJECT_ROOT / "App" / "web_server.py").is_file()
    assert ocr_catalog_eval.ROOT == PROJECT_ROOT
    assert artifact_paths.ROOT == PROJECT_ROOT
    assert web_pipeline.ROOT == PROJECT_ROOT
    assert manual_mappings.DEFAULT_PATH == PROJECT_ROOT / "data" / "knowledge" / "manual_mappings.json"
    assert ocr_run.MODEL == PROJECT_ROOT / "models" / "cyrillic_PP-OCRv5_rec_mobile.onnx"
    assert manual_mappings.DEFAULT_PATH.exists()
    assert ocr_run.MODEL.exists()
    settings = numeric_settings.load_numeric_settings()
    assert settings["vat_rate_percent"]
    assert settings["max_images"] >= 2
