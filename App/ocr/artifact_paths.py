"""OCR cache is persistent; disposable regression reports live on the RAM disk."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OCR_CACHE = ROOT / '.cache' / 'ocr_baselines'
OCR_REPORTS = Path(r'R:\Temp\ocr_regression')
EXPERIMENT_OUTPUTS = ROOT / 'experiment_outputs'
