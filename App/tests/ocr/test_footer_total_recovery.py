from App.ocr import footer_total_recovery as footer


def test_auto_mode_respects_disabled_geometry(tmp_path):
    (tmp_path / '.env').write_text('OCR_PERSPECTIVE_2D_MODE=off\n', encoding='utf-8')
    assert footer.setting_mode(tmp_path) == 'off'


def test_disabled_footer_never_runs_ocr(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('Disabled recovery must not inspect the image')
    monkeypatch.setattr(footer, 'recover_footer_total', unexpected)
    total, audit = footer.maybe_recover_footer_total(None, None, '', mode_override='off')
    assert total == '' and audit['status'] == 'disabled'


def test_existing_printed_total_is_preserved(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('Existing total must not be replaced')
    monkeypatch.setattr(footer, 'recover_footer_total', unexpected)
    total, audit = footer.maybe_recover_footer_total(None, None, '12 345,67', mode_override='select')
    assert total == '12 345,67' and not audit['selected']


def test_unconfirmed_footer_never_becomes_expected_total(monkeypatch):
    monkeypatch.setattr(footer, 'recover_footer_total', lambda *a, **kw: (
        None, {'status': 'unconfirmed', 'reason': 'ocr_engine_disagreement'}))
    total, audit = footer.maybe_recover_footer_total(None, None, '', mode_override='select')
    assert total == '' and not audit['selected']
