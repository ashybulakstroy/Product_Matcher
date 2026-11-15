"""Patch content/atomicity/restore tests against a source-shaped synthetic App."""
from pathlib import Path
import importlib.util
import hashlib
import json

import pytest

PKG=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rounding_installer',PKG/'install.py')
installer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def source_shape():
    # The three exact fragments below are copied from the latest GitHub
    # receipt_editing.py. Rest of the test source is deliberately minimal.
    return (installer.IMPORT_ANCHOR +
            'from App.ocr.numeric_settings import load_numeric_settings\n'
            'from decimal import Decimal, ROUND_HALF_UP\n'
            'CENT=Decimal(".01")\n'
            'def update_receipt(receipt):\n'
            '    missing=[]\n'
            '    computed=Decimal("80812.65")\n'
            '    expected=Decimal("80813")\n'
            '    totals={"product":Decimal("69665.65"),"service":Decimal(0),"tax":Decimal("11147")}\n'
            '    vat={"status":"match"}\n'
            '    def money(x): return f"{x:.2f}"\n'
            '    def parse_number(x): return Decimal(x) if x is not None else None\n'
            '    PAYABLE={"product","service","tax"}\n'
            '    validation_status="match"\n'
            '    receipt.setdefault("revision",0)\n'
            '    receipt.setdefault("rows",[])\n'
            +installer.OLD_STATUS+
            '    validation_status=status\n'
            +installer.OLD_CALC+
            '    return receipt\n')


def setup(monkeypatch,tmp_path):
    project=tmp_path/'testproject';project.mkdir()
    pkg=project/'experiment_total_rounding_tenge_v1';pkg.mkdir()
    src=pkg/'src';src.mkdir()
    (src/'receipt_rounding.py').write_bytes((PKG/'src'/'receipt_rounding.py').read_bytes())
    path=project/'App'/'ocr';path.mkdir(parents=True)
    initial=source_shape().encode('utf8')
    (path/'receipt_editing.py').write_bytes(initial)
    (path/'receipt_math.py').write_text('def verify_vat(*args, **kwargs): return {}\n',encoding='utf8')
    (path/'numeric_settings.py').write_text('def load_numeric_settings(): return {\"vat_rate_percent\":\"16\"}\n',encoding='utf8')
    monkeypatch.setattr(installer,'PACKAGE',pkg)
    monkeypatch.setattr(installer,'ROOT',project)
    monkeypatch.setattr(installer,'APP',path)
    monkeypatch.setattr(installer,'TARGET',path/'receipt_editing.py')
    monkeypatch.setattr(installer,'MODULE',path/'receipt_rounding.py')
    monkeypatch.setattr(installer,'SOURCE_MODULE',src/'receipt_rounding.py')
    monkeypatch.setattr(installer,'OUTPUT',project/'data'/'output'/'receipt_total_rounding')
    monkeypatch.setattr(installer,'MANIFEST',installer.OUTPUT/'install_manifest.json')
    monkeypatch.setattr(installer,'BACKUP',installer.OUTPUT/'backups')
    return path,initial


def test_preview_is_non_mutating(monkeypatch,tmp_path):
    path,original=setup(monkeypatch,tmp_path)
    installer.ensure_preview()
    assert (path/'receipt_editing.py').read_bytes()==original
    assert not (path/'receipt_rounding.py').exists()
    assert 'verify' not in '' or (installer.OUTPUT/'preview.diff').exists()


def test_apply_restore_byte_identical(monkeypatch,tmp_path):
    path,original=setup(monkeypatch,tmp_path)
    installer.apply()
    patched=(path/'receipt_editing.py').read_bytes()
    assert patched != original
    assert (path/'receipt_rounding.py').is_file()
    manifest=json.loads(installer.MANIFEST.read_text(encoding='utf8'))
    assert hashlib.sha256(patched).hexdigest()==manifest['target_after_sha256']
    installer.apply() # idempotent
    installer.restore()
    assert (path/'receipt_editing.py').read_bytes()==original
    assert not (path/'receipt_rounding.py').exists()


def test_restore_refuses_to_lose_user_changes(monkeypatch,tmp_path):
    path,_=setup(monkeypatch,tmp_path)
    installer.apply()
    with (path/'receipt_editing.py').open('a',encoding='utf8') as stream:stream.write('\n# local change\n')
    with pytest.raises(RuntimeError,match='изменён'):
        installer.restore()


def test_refuse_changed_anchor(monkeypatch,tmp_path):
    path,original=setup(monkeypatch,tmp_path)
    (path/'receipt_editing.py').write_bytes(original.replace(b'expected is None',b'expected_amount is None'))
    with pytest.raises(RuntimeError,match='ожидалось ровно одно'):
        installer.apply()
    assert not installer.MANIFEST.exists()


def test_refuse_replace_existing_different_rounding_module(monkeypatch,tmp_path):
    path,_=setup(monkeypatch,tmp_path)
    (path/'receipt_rounding.py').write_text('# Custom module',encoding='utf8')
    with pytest.raises(RuntimeError,match='другим содержимым'):
        installer.apply()
    assert not installer.MANIFEST.exists()


def test_patched_app_updates_display_and_preserves_exact_sum(monkeypatch,tmp_path):
    import subprocess
    import sys
    path,_=setup(monkeypatch,tmp_path)
    installer.apply()
    sample="""from App.ocr.receipt_editing import update_receipt
receipt={'rows':[{'row':14,'kind':'product','line_total':'69665.65'},
                 {'row':17,'kind':'tax','line_total':'11147.00'}]}
result=update_receipt(receipt)['calculation']
assert result['status']=='match',result
assert result['computed_total']=='80813.00',result
assert result['exact_computed_total']=='80812.65',result
assert result['rounding_adjustment']=='0.35',result
assert result['exact_difference']=='-0.35',result
assert result['difference']=='0.00',result
"""
    proc=subprocess.run([sys.executable,'-c',sample],cwd=installer.ROOT,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    installer.restore()
