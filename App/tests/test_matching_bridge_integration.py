import copy
import json
import threading
import urllib.error
import urllib.request
import uuid
from decimal import Decimal

import pytest

from App.exchange.store import ExchangeStore
from App.invoicing import catalog_1c, invoice_service, matching_bridge, stock_sync
from App.ocr.receipt_editing import update_receipt
from App.web_server import Job, create_server


def receipt(job_id):
    return update_receipt(dict(job_id=job_id, revision=0, order_number='TEST-1', date='06.11.2026',
        rows=[dict(row=1, kind='product', name_raw='Кабель медный', name='Кабель медный',
                   name_confirmed=True, quantity_raw='2', price_raw='100')], total_raw='200'))


def write_stock(directory, database, code, *, quantity='10'):
    item = dict(uuid=str(uuid.uuid4()), code=code, name='Кабель медный', unit='м', service=False,
                stock_quantity=quantity, stock_amount='500', accounting_price='50', account_code='1330')
    data = dict(schema_version='1.0', source='1c', snapshot_id=str(uuid.uuid4()),
        database_id=database, organization=dict(id=str(uuid.uuid4()), name='Тестовая организация'),
        warehouse=dict(id=str(uuid.uuid4()), name='Тестовый склад'), stock_as_of='2026-11-06T10:00:00+05:00',
        record_count=1, items=[item])
    raw = json.dumps(data, ensure_ascii=False).encode()
    stock_sync.validate(raw)
    directory.mkdir(parents=True, exist_ok=True)
    stock_sync.path_for(directory, database).write_bytes(raw)


@pytest.fixture
def bridge_server(tmp_path, monkeypatch):
    directory = tmp_path / 'catalog'
    monkeypatch.setattr(catalog_1c, 'CATALOG_DIR', directory)
    write_stock(directory, 'base-A', '00000123')
    write_stock(directory, 'base-B', '00000999')
    output = tmp_path / 'jobs'
    exchange = ExchangeStore(output / 'exchange')
    jobs = {}
    for database in ('base-A', 'base-B', 'base-missing'):
        identifier = uuid.uuid4().hex
        job_directory = output / identifier
        job_directory.mkdir()
        job = Job(identifier, 'synthetic.json', job_directory)
        job.status = 'done'
        job.result = receipt(identifier)
        job.persist()
        task, _ = exchange.create(database, database)
        exchange.bind_job(task.task_id, identifier)
        jobs[database] = identifier
    instance = create_server('127.0.0.1', 0, output, processor=lambda job: None)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance, f'http://127.0.0.1:{instance.server_port}', jobs
    instance.shutdown()
    instance.server_close()
    instance.store.pool.shutdown(wait=True)
    thread.join(timeout=3)


def post(base, job_id, payload):
    request = urllib.request.Request(base + f'/api/jobs/{job_id}/analogs',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def test_job_database_isolation_decimal_response_and_no_writes(bridge_server):
    instance, base, jobs = bridge_server
    for database, code in [('base-A', '00000123'), ('base-B', '00000999')]:
        job = instance.store.get(jobs[database])
        before = copy.deepcopy(job.result)
        files = {path.name: path.read_bytes() for path in job.directory.iterdir() if path.is_file()}
        data = post(base, job.id, dict(row_id='1', receipt_revision=0, database_id='untrusted-other-base'))
        assert data['database_id'] == database
        assert data['items'][0]['code'] == code
        assert data['items'][0]['accounting_price'] == '50'
        assert data['recommended_id'] == data['items'][0]['id']
        assert job.result == before
        assert {path.name: path.read_bytes() for path in job.directory.iterdir() if path.is_file()} == files
        assert not invoice_service.state_path(job).exists()


def test_missing_database_never_falls_back_to_another_catalog(bridge_server):
    _, base, jobs = bridge_server
    with pytest.raises(urllib.error.HTTPError) as error:
        post(base, jobs['base-missing'], dict(row_id='1'))
    assert error.value.code == 400


@pytest.mark.parametrize('payload,status', [
    ({'row_id': True}, 400), ({'row_id': '99'}, 400),
    ({'row_id': '1', 'receipt_revision': 99}, 409),
    ({'row_id': '1', 'receipt_revision': False}, 400),
])
def test_invalid_or_stale_requests(bridge_server, payload, status):
    _, base, jobs = bridge_server
    with pytest.raises(urllib.error.HTTPError) as error:
        post(base, jobs['base-A'], payload)
    assert error.value.code == status


def test_revision_change_during_search_returns_conflict(bridge_server, monkeypatch):
    instance, base, jobs = bridge_server
    original = matching_bridge.for_invoice_row
    def changed(*args, **kwargs):
        data = original(*args, **kwargs)
        instance.store.get(jobs['base-A']).result['revision'] = 1
        return data
    monkeypatch.setattr(matching_bridge, 'for_invoice_row', changed)
    with pytest.raises(urllib.error.HTTPError) as error:
        post(base, jobs['base-A'], dict(row_id='1'))
    assert error.value.code == 409


@pytest.mark.parametrize('quantity', [0, -1, '', None, 'NaN', '1e999'])
def test_invalid_quantity_does_not_relax_stock_requirement(quantity):
    with pytest.raises(ValueError):
        matching_bridge.vote_candidates('Кабель', {'items': []}, quantity=quantity)


def test_matcher_cannot_offer_unknown_or_insufficient_stock():
    index = dict(items=[dict(id='0001', name='Кабель', stock_quantity=Decimal('1'))])
    def invalid(*args, **kwargs):
        return dict(voting=dict(positions=[dict(recommended_stock_id='0001',
            candidates=[dict(stock_id='0001'), dict(stock_id='not-in-catalog')])]))
    data = matching_bridge.vote_candidates('Кабель', index, quantity=2, matcher=invalid)
    assert not data['items']
    assert data['recommended_id'] is None


def test_invoice_candidates_skip_matcher_for_confirmed_or_strong_search(monkeypatch):
    from App.invoicing import invoice_generation as generation
    row = dict(name='Кабель медный', quantity='2', price='100')
    monkeypatch.setattr(invoice_service, 'resolve_source', lambda *args: dict(status='missing'))
    monkeypatch.setattr(catalog_1c, 'catalog_item', lambda *args: None)
    monkeypatch.setattr(catalog_1c, 'search_catalog', lambda *args, **kwargs:
        dict(items=[dict(id='ordinary', name='Кабель медный', score=95, stock_quantity='2')], errors=[]))
    def fail(*args, **kwargs):
        raise AssertionError('Optional matcher must not run')
    monkeypatch.setattr(matching_bridge, 'vote_candidates', fail)
    monkeypatch.setattr(generation, 'confirmed_candidates', lambda *args: [])
    assert invoice_service._row_candidates(row)['items'][0]['id'] == 'ordinary'
    monkeypatch.setattr(catalog_1c, 'search_catalog', lambda *args, **kwargs: dict(items=[], errors=[]))
    monkeypatch.setattr(generation, 'confirmed_candidates', lambda *args: [dict(id='confirmed', stock_quantity='2')])
    assert invoice_service._row_candidates(row)['remembered'][0]['id'] == 'confirmed'
    monkeypatch.setattr(generation, 'confirmed_candidates', lambda *args: [])
    assert not invoice_service._row_candidates(row, allow_analog_fallback=False)['items']


def test_stock_filter_uses_exact_receipt_quantity_and_excludes_unknown():
    from App.invoicing.invoice_service import sufficient_stock
    assert sufficient_stock({'stock_quantity': '3.25'}, '3.25')
    assert not sufficient_stock({'stock_quantity': '3.24'}, '3.25')
    for stock in (None, 'NaN', 'Infinity', '0', '-1'):
        assert not sufficient_stock({'stock_quantity': stock}, '2')


def test_manual_search_filters_before_limit(monkeypatch):
    monkeypatch.setattr(catalog_1c, 'catalog', lambda *args, **kwargs: {'items': [1,2,3]})
    monkeypatch.setattr(catalog_1c, 'search_catalog', lambda *args, **kwargs: dict(items=[
        dict(id='low', stock_quantity='1'), dict(id='ok', stock_quantity='4.5')], errors=[]))
    result = invoice_service.search_catalog_1c.__wrapped__('test', limit=1, required_quantity='4.5')
    assert [item['id'] for item in result['items']] == ['ok']


def test_equal_scores_prefer_smaller_displayed_tax_difference():
    from App.invoicing.catalog_1c import sort_candidate_ties
    items = [dict(id='higher_score', score=99, accounting_price=10),
             dict(id='below', score=90, accounting_price=80),
             dict(id='above', score=90, accounting_price=120),
             dict(id='unknown', score=90, accounting_price=None)]
    assert [item['id'] for item in sort_candidate_ties(items, 100)] == ['higher_score', 'above', 'below', 'unknown']
    assert sort_candidate_ties(items, None) == items
