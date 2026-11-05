import json
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import mimetypes
import sys
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from App.config import WEB_HOST, WEB_PORT
from App.services.opencode_matching import match_products_opencode_4
from App.services.matching import match_products, match_products_sql_str_2, match_products_sql_str_3


BASE_DIR = Path(__file__).resolve().parents[1]
STOCK_FILE = BASE_DIR / "data" / "reference" / "1c_ref" / "catalog_1c_current_buh_kz_01_01e211419c01.json"
STATIC_ROOT = Path(__file__).resolve().parent / "web"

HTML_PAGE = """
<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8" />
    <title>Interactive Matching</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; }
      .panel { max-width: 1100px; margin: 0 auto; }
      .form-grid { display: grid; grid-template-columns: 1.5fr 0.7fr 0.7fr; gap: 12px; align-items: end; }
      label { display: block; font-weight: 600; margin-bottom: 6px; }
      input, select, button { width: 100%; box-sizing: border-box; padding: 10px; font-size: 14px; }
      button { background: #1d4ed8; color: white; border: none; border-radius: 6px; cursor: pointer; }
      table { width: 100%; border-collapse: collapse; margin-top: 20px; }
      th, td { border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; vertical-align: top; }
      th { background: #f3f4f6; }
      .small { color: #555; font-size: 12px; }
      .muted-table { background: #f3f4f6; }
      .low-stock { color: #b91c1c; font-weight: 700; }
      .empty { color: #666; font-style: italic; }
    </style>
  </head>
  <body>
    <div class="panel">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:16px; flex-wrap:wrap;">
        <h2 style="margin:0;">Интерактивный подбор аналогов</h2>
        <a href="/analog" style="font-weight:700; color:#1d4ed8; text-decoration:none;">Ашыбулак строй / /analog</a>
      </div>
      <p class="small">При равном проценте выше кандидат с ближайшей ценой. Сравнение с учётной ценой 1С; цены должны быть в одной валюте и за одинаковую единицу. Введите товар из Торгсофт, задайте минимальный остаток и количество кандидатов. Ответ будет показан в таблице.</p>

      <form id="match-form" onsubmit="event.preventDefault(); runMatch(); return false;">
        <div class="form-grid">
          <div>
            <label for="query">Название товара</label>
            <input id="query" value="Тачка садовая одноколесная усиленная ЭКСПЕРТ" />
          </div>
          <div>
            <label for="query-price">Цена за единицу</label>
            <input id="query-price" type="number" min="0" step="any" placeholder="Необязательно" />
          </div>
          <div>
            <label for="min_stock">Мин. остаток</label>
            <input id="min_stock" type="number" min="0" value="3" />
          </div>
        </div>
        <details style="margin-top:14px; border:1px solid #d1d5db; border-radius:8px; padding:10px 12px;">
          <summary style="cursor:pointer; color:#51617b;">Настройки поиска</summary>
          <div class="form-grid" style="margin-top:12px; grid-template-columns:1fr 1fr;">
          <div>
            <label for="algorithm_id">Алгоритм</label>
            <select id="algorithm_id">
              <option value="all" selected>Все алгоритмы</option>
              <option value="copilot_1">copilot_1</option>
              <option value="sql_str_2">sql_str_2</option>
              <option value="sql_str_3">sql_str_3</option>
              <option value="opencode_4">opencode_4</option>
            </select>
          </div>
          <div>
            <label for="top_n">Top X</label>
            <select id="top_n">
              <option value="3">3</option>
              <option value="5">5</option>
              <option value="10" selected>10</option>
            </select>
          </div>
        </div>

        </details>
        <div style="margin-top: 12px; max-width: 220px;">
          <button type="submit">Проверить</button>
        </div>
      </form>

      <div id="result-box" class="small" style="margin-top: 18px; min-height: 24px;"></div>
      <div id="algorithm-results"></div>
    </div>

    <script>
      function encodeResultPayload(value) {
        return btoa(unescape(encodeURIComponent(JSON.stringify(value))));
      }

      function buildFullResultsUrl(algorithmResult) {
        if (algorithmResult && algorithmResult.full_results_url) {
          return algorithmResult.full_results_url;
        }
        const algorithmId = algorithmResult && algorithmResult.algorithm_id ? algorithmResult.algorithm_id : 'copilot_1';
        const payload = encodeResultPayload(algorithmResult || {});
        return `/analog/list?algorithm_id=${encodeURIComponent(algorithmId)}&payload=${encodeURIComponent(payload)}`;
      }

      function renderAlgorithmBlock(algorithmResult) {
        const items = algorithmResult && algorithmResult.items ? algorithmResult.items : [];
        const item = items[0] || {};
        const mainCandidates = item.candidates || [];
        const excludedCandidates = item.excluded_candidates || [];
        const best = algorithmResult.best_candidate || mainCandidates[0] || null;
        const lowStock = algorithmResult.low_stock_candidate || excludedCandidates[0] || null;
        const algorithmId = algorithmResult.algorithm_id || 'copilot_1';
        const fullResultsUrl = buildFullResultsUrl(algorithmResult);

        const block = document.createElement('div');
        block.id = `algorithm-${algorithmId}`;
        block.style.marginTop = '20px';
        block.style.border = '1px solid #d1d5db';
        block.style.borderRadius = '8px';
        block.style.padding = '16px';

        const bestText = best ? `${best.stock_name} · ${best.similarity_pct}% · остаток ${best.stock_qty ?? '-'}` : 'Нет кандидатов';
        const lowText = lowStock ? `${lowStock.stock_name} · ${lowStock.similarity_pct}% · остаток ${lowStock.stock_qty ?? '-'}` : 'Недостаточного остатка нет';
        const summaryCandidates = mainCandidates.length ? [mainCandidates[0]] : [];
        const summaryExcluded = excludedCandidates.length ? [excludedCandidates[0]] : [];

        block.innerHTML = `
          <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:12px;">
            <div>
              <strong>Алгоритм:</strong>
              <a href="${fullResultsUrl}">${algorithmId}</a>
            </div>
            <div class="small">Лучший кандидат: ${bestText}</div>
          </div>

          <table style="width:100%; border-collapse:collapse; margin-top:8px;">
            <thead>
              <tr>
                <th>№</th>
                <th>Кандидат из 1С</th>
                <th>Цена за единицу (1С)</th>
                  <th>Остаток</th>
                <th>Сходство %</th>
                <th>Имя алгоритма</th>
                <th>Полный список кандидатов</th>
              </tr>
            </thead>
            <tbody>
              ${(summaryCandidates.length ? summaryCandidates : []).map((candidate, idx) => `
                <tr>
                  <td>${idx + 1}</td>
                  <td>${candidate.stock_name || '-'}</td>
                  <td>${candidate.stock_price_display ?? \'—\'}<div class="small">Отклонение: ${candidate.price_difference_display ?? \'—\'}</div></td>
                  <td>${candidate.stock_qty ?? '-'}</td>
                  <td>${candidate.similarity_pct ?? '-'}</td>
                  <td><a href="${fullResultsUrl}">${algorithmId}</a></td>
                  <td><a href="${fullResultsUrl}">Все кандидаты</a></td>
                </tr>
              `).join('') || `<tr><td colspan="7" class="small">Нет кандидатов</td></tr>`}
            </tbody>
          </table>

          <div style="margin-top:20px; border-top:1px solid #e5e7eb; padding-top:16px;">
            <h4 style="margin:0 0 8px;">Полный список кандидатов по алгоритму ${algorithmId}</h4>
            <p class="small"><a href="${fullResultsUrl}">Открыть полный список кандидатов на отдельной странице</a></p>
          </div>

          <div style="margin-top: 20px; border-top: 1px solid #e5e7eb; padding-top: 14px;">
            <h4 style="margin:0 0 8px;">Аналоги без достаточного остатка</h4>
            <table style="width:100%; border-collapse:collapse; background:#f3f4f6;">
              <thead>
                <tr>
                  <th>№</th>
                  <th>Кандидат из 1С</th>
                  <th>Цена за единицу (1С)</th>
                  <th>Остаток</th>
                  <th>Сходство %</th>
                  <th>Имя алгоритма</th>
                  <th>Полный список кандидатов</th>
                </tr>
              </thead>
              <tbody>
                ${(summaryExcluded.length ? summaryExcluded : []).map((candidate, idx) => `
                  <tr>
                    <td>${idx + 1}</td>
                    <td>${candidate.stock_name || '-'}</td>
                    <td>${candidate.stock_price_display ?? \'—\'}<div class="small">Отклонение: ${candidate.price_difference_display ?? \'—\'}</div></td>
                    <td class="low-stock" style="color:#b91c1c; font-weight:700;">${candidate.stock_qty ?? 0}</td>
                    <td>${candidate.similarity_pct ?? '-'}</td>
                    <td><a href="${fullResultsUrl}">${algorithmId}</a></td>
                    <td><a href="${fullResultsUrl}">Список кандидатов</a></td>
                  </tr>
                `).join('') || `<tr><td colspan="7" class="small">Нет кандидатов</td></tr>`}
              </tbody>
            </table>
            <div class="small" style="margin-top: 8px;">Первый кандидат с недостаточным остатком: ${lowText}</div>
          </div>
        `;

        return block;
      }

      function renderTable(data) {
        const box = document.getElementById('result-box');
        const container = document.getElementById('algorithm-results');
        container.innerHTML = '';

        if (!data || !data.results_by_algorithm || !data.results_by_algorithm.length) {
          box.textContent = 'Нет данных';
          return;
        }

        const positiveRows = [];
        const deficientRows = [];

        data.results_by_algorithm.forEach((algorithmResult) => {
          const algorithmId = algorithmResult && algorithmResult.algorithm_id ? algorithmResult.algorithm_id : 'unknown';
          const candidate = algorithmResult && algorithmResult.best_candidate ? { ...algorithmResult.best_candidate, algorithm_id: algorithmId, full_results_url: algorithmResult.full_results_url } : null;
          const lowStockCandidate = algorithmResult && algorithmResult.low_stock_candidate ? { ...algorithmResult.low_stock_candidate, algorithm_id: algorithmId, full_results_url: algorithmResult.full_results_url } : null;

          if (candidate) {
            positiveRows.push(candidate);
          }
          if (lowStockCandidate) {
            deficientRows.push(lowStockCandidate);
          }
        });

        box.textContent = 'ТОП-1 по всем алгоритмам';

        container.innerHTML = `
          <div style="margin-top: 18px; border: 1px solid #d1d5db; border-radius: 8px; padding: 16px;">
            <h3 style="margin: 0 0 12px;">Таблица 1. Положительные результаты (TOP-1 по всем алгоритмам)</h3>
            <table style="width:100%; border-collapse:collapse; margin-top:8px;">
              <thead>
                <tr>
                  <th>№</th>
                  <th>Алгоритм</th>
                  <th>Кандидат из 1С</th>
                  <th>Цена за единицу (1С)</th>
                  <th>Остаток</th>
                  <th>Сходство %</th>
                  <th>Полный список</th>
                </tr>
              </thead>
              <tbody>
                ${positiveRows.length ? positiveRows.map((candidate, index) => `
                  <tr>
                    <td>${index + 1}</td>
                    <td>${candidate.algorithm_id || '-'}</td>
                    <td>${candidate.stock_name || '-'}</td>
                    <td>${candidate.stock_price_display ?? \'—\'}<div class="small">Отклонение: ${candidate.price_difference_display ?? \'—\'}</div></td>
                    <td>${candidate.stock_qty ?? '-'}</td>
                    <td>${candidate.similarity_pct ?? '-'}</td>
                    <td><a href="${candidate.full_results_url || '#'}">Открыть</a></td>
                  </tr>
                `).join('') : '<tr><td colspan="7" class="small">Нет положительных кандидатов</td></tr>'}
              </tbody>
            </table>
          </div>

          <div style="margin-top: 20px; border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; background:#f3f4f6;">
            <h3 style="margin: 0 0 12px;">Таблица 2. Дефициты / недостаточный остаток</h3>
            <table style="width:100%; border-collapse:collapse; margin-top:8px;">
              <thead>
                <tr>
                  <th>№</th>
                  <th>Алгоритм</th>
                  <th>Кандидат из 1С</th>
                  <th>Цена за единицу (1С)</th>
                  <th>Остаток</th>
                  <th>Сходство %</th>
                  <th>Полный список</th>
                </tr>
              </thead>
              <tbody>
                ${deficientRows.length ? deficientRows.map((candidate, index) => `
                  <tr>
                    <td>${index + 1}</td>
                    <td>${candidate.algorithm_id || '-'}</td>
                    <td>${candidate.stock_name || '-'}</td>
                    <td>${candidate.stock_price_display ?? \'—\'}<div class="small">Отклонение: ${candidate.price_difference_display ?? \'—\'}</div></td>
                    <td style="color:#b91c1c; font-weight:700;">${candidate.stock_qty ?? '-'}</td>
                    <td>${candidate.similarity_pct ?? '-'}</td>
                    <td><a href="${candidate.full_results_url || '#'}">Открыть</a></td>
                  </tr>
                `).join('') : '<tr><td colspan="7" class="small">Нет кандидатов с недостаточным остатком</td></tr>'}
              </tbody>
            </table>
          </div>
        `;
      }

      async function runMatch() {
        const query = document.getElementById('query').value.trim();
        const min_stock = Number(document.getElementById('min_stock').value || 0);
        const price = document.getElementById('query-price').value;
        const top_n = Number(document.getElementById('top_n').value || 10);
        const algorithm_id = document.getElementById('algorithm_id').value || 'all';
        const allAlgorithms = ['copilot_1', 'sql_str_2', 'sql_str_3', 'opencode_4'];
        const selectedAlgorithms = algorithm_id === 'all' ? allAlgorithms : [algorithm_id];

        if (!query) {
          document.getElementById('result-box').textContent = 'Введите название товара';
          return;
        }

        const response = await fetch('/analog', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, price, min_stock, top_n, algorithm_id: algorithm_id === 'all' ? null : algorithm_id, algorithms: selectedAlgorithms })
        });

        const data = await response.json();
        renderTable(data);
      }

      document.addEventListener('DOMContentLoaded', function () {
        const input = document.getElementById('query');
        if (input) {
          input.addEventListener('keydown', function (event) {
            if (event.key === 'Enter') {
              event.preventDefault();
              runMatch();
            }
          });
        }
      });
    </script>
  </body>
</html>
"""

FULL_RESULTS_PAGE = """
<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8" />
    <title>Полный список кандидатов</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; }
      .panel { max-width: 1100px; margin: 0 auto; }
      table { width: 100%; border-collapse: collapse; margin-top: 20px; }
      th, td { border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; }
      th { background: #f3f4f6; }
      .small { color: #555; font-size: 12px; }
      .link { color: #1d4ed8; text-decoration: none; }
    </style>
  </head>
  <body>
    <div class="panel">
      <div id="result-heading"><h2>Полный список кандидатов</h2></div>
      <p class="small"><a class="link" href="/analog">Назад к поиску</a></p>
      <div id="result-container"></div>
    </div>

    <script>
      function escapeHtml(value) {
        return String(value ?? '')
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#39;');
      }

      function decodePayload(rawValue) {
        if (!rawValue) {
          return null;
        }
        try {
          const normalized = rawValue.replace(/-/g, '+').replace(/_/g, '/');
          const binary = atob(normalized);
          const bytes = Uint8Array.from(binary, ch => ch.charCodeAt(0));
          const text = new TextDecoder('utf-8').decode(bytes);
          return JSON.parse(text);
        } catch (error) {
          return null;
        }
      }

      async function loadResultData() {
        const params = new URLSearchParams(window.location.search);
        const resultId = params.get('result_id');
        if (resultId) {
          const response = await fetch(`/analog/list/data?result_id=${encodeURIComponent(resultId)}`);
          if (!response.ok) {
            return null;
          }
          return await response.json();
        }
        return decodePayload(params.get('payload'));
      }

      function renderRows(list, algorithmId, lowStock = false) {
        return list.map((candidate, idx) => `
          <tr>
            <td>${idx + 1}</td>
            <td>${escapeHtml(candidate.stock_name || '-')}<div class="small">${(candidate.match_reasons || []).map(escapeHtml).join('; ')}</div></td>
            <td>${candidate.stock_price_display ?? \'—\'}<div class="small">Отклонение: ${candidate.price_difference_display ?? \'—\'}</div></td>
            <td class="${lowStock ? 'low-stock' : ''}" style="${lowStock ? 'color:#b91c1c; font-weight:700;' : ''}">${candidate.stock_qty ?? '-'}</td>
            <td>${candidate.similarity_pct ?? '-'}</td>
            <td>${escapeHtml(algorithmId)}</td>
          </tr>
        `).join('');
      }

      function renderItem(item, index, algorithmId) {
        const candidates = Array.isArray(item.candidates) ? item.candidates : [];
        const excluded = Array.isArray(item.excluded_candidates) ? item.excluded_candidates : [];
        const itemTitle = item.checked_name || `Позиция ${item.position_no || index + 1}`;
        const mainRows = renderRows(candidates, algorithmId, false);
        const excludedRows = renderRows(excluded, algorithmId, true);
        return `
          <section style="margin-top: 28px;">
            <h3>${escapeHtml(itemTitle)}</h3>
            <h4>Таблица 1. Достаточный остаток</h4>
            <table>
              <thead>
                <tr>
                  <th>№</th>
                  <th>Кандидат из 1С</th>
                  <th>Цена за единицу (1С)</th>
                  <th>Остаток</th>
                  <th>Сходство %</th>
                  <th>Имя алгоритма</th>
                </tr>
              </thead>
              <tbody>${mainRows || '<tr><td colspan="6" class="small">Нет кандидатов с достаточным остатком</td></tr>'}</tbody>
            </table>

            <h4 style="margin-top: 24px;">Таблица 2. Недостаточный остаток</h4>
            <table>
              <thead>
                <tr>
                  <th>№</th>
                  <th>Кандидат из 1С</th>
                  <th>Цена за единицу (1С)</th>
                  <th>Остаток</th>
                  <th>Сходство %</th>
                  <th>Имя алгоритма</th>
                </tr>
              </thead>
              <tbody>${excludedRows || '<tr><td colspan="6" class="small">Нет кандидатов с недостаточным остатком</td></tr>'}</tbody>
            </table>
          </section>
        `;
      }

      async function renderList() {
        const params = new URLSearchParams(window.location.search);
        const algorithmId = params.get('algorithm_id') || 'copilot_1';
        const container = document.getElementById('result-container');
        const heading = document.getElementById('result-heading');

        try {
          const data = await loadResultData();
          if (!data || !Array.isArray(data.items) || !data.items.length) {
            if (heading) {
              heading.innerHTML = '<h2>Полный список кандидатов</h2>';
            }
            container.innerHTML = '<p class="small">Нет данных для этого алгоритма.</p>';
            return;
          }

          const queryText = data.query || data.checked_name || 'Полный список кандидатов';
          if (heading) {
            heading.innerHTML = `<h2>${escapeHtml(queryText)}</h2><p class="small">Алгоритм: ${escapeHtml(algorithmId)}</p>`;
          }

          container.innerHTML = data.items
            .map((item, index) => renderItem(item, index, algorithmId))
            .join('');
        } catch (error) {
          if (heading) {
            heading.innerHTML = '<h2>Полный список кандидатов</h2>';
          }
          container.innerHTML = '<p class="small">Не удалось загрузить список кандидатов.</p>';
        }
      }

      renderList();
    </script>
  </body>
</html>
"""


def _price(value):
    """Nonnegative finite unit price; missing/invalid catalog prices are unknown."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).replace("\u00a0", "").replace(" ", "").replace(",", "."))
    except InvalidOperation:
        return None
    return number if number.is_finite() and number >= 0 else None


def _item_price(item):
    for key in ("price", "unit_price", "accounting_price"):
        value = _price(item.get(key))
        if value is not None:
            return value
    return None


def load_stock_items(min_stock: int = 0):
    if not STOCK_FILE.exists():
        return []
    with STOCK_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload.get("items") or []
    result = []
    for item in items:
        qty_raw = item.get("stock_quantity")
        if qty_raw is None:
            qty_raw = item.get("quantity")
        if qty_raw is None:
            qty_raw = item.get("qty")
        if qty_raw is not None:
            try:
                qty_value = float(qty_raw)
            except (TypeError, ValueError):
                qty_value = 0
        else:
            qty_value = 0
        if min_stock and qty_value < min_stock:
            continue
        result.append({
            "id": item.get("code") or item.get("uuid"),
            "name": item.get("name") or item.get("code", ""),
            "quantity": qty_value,
            "accounting_price": item.get("accounting_price"),
        })
    return result


ALGORITHM_REGISTRY = {
    "copilot_1": {
        "id": "copilot_1",
        "name": "copilot_1",
        "title": "Copilot 1",
        "description": "Стандартный алгоритм сопоставления по нормализации и сходству RapidFuzz.",
    },
    "sql_str_2": {
        "id": "sql_str_2",
        "name": "sql_str_2",
        "title": "SQL STR 2",
        "description": "Строгий SQL-подобный поиск по смысловым словам: вхождение ключевых слов, без артикулов и цифрового шума.",
    },
    "sql_str_3": {
        "id": "sql_str_3", "name": "sql_str_3", "title": "SQL STR 3",
        "description": "Взвешенные слова, строгие совпадения, размеры, масса, объём и модели.",
    },
    "opencode_4": {
        "id": "opencode_4", "name": "opencode_4", "title": "OpenCode 4",
        "description": "Вид товара, артикул, модель, размеры, цвет и марка по идеям анализа ошибок OpenCode.",
    },
}

_RESULT_CACHE = OrderedDict()
_RESULT_CACHE_LOCK = threading.Lock()
_RESULT_CACHE_TTL_SECONDS = 3600
_RESULT_CACHE_MAX_ENTRIES = 32


def _store_full_result(result: dict) -> str:
    result_id = uuid.uuid4().hex
    stored_at = time.time()
    with _RESULT_CACHE_LOCK:
        _RESULT_CACHE[result_id] = (stored_at, result)
        while len(_RESULT_CACHE) > _RESULT_CACHE_MAX_ENTRIES:
            _RESULT_CACHE.popitem(last=False)
    return result_id


def _load_full_result(result_id: str):
    if not result_id:
        return None
    now = time.time()
    with _RESULT_CACHE_LOCK:
        item = _RESULT_CACHE.get(result_id)
        if item is None:
            return None
        stored_at, result = item
        if now - stored_at > _RESULT_CACHE_TTL_SECONDS:
            _RESULT_CACHE.pop(result_id, None)
            return None
        _RESULT_CACHE.move_to_end(result_id)
        return result


def clear_full_result_cache() -> None:
    with _RESULT_CACHE_LOCK:
        _RESULT_CACHE.clear()


def _extract_query(checked_products):
    if not checked_products:
        return ""

    for item in checked_products:
        if not isinstance(item, dict):
            continue
        for key in ("query", "checked_name", "name", "product_name"):
            value = item.get(key)
            if value:
                return str(value)
    return ""


def _build_algorithm_result(checked_products, stock_products, top_n=5, min_stock=0, algorithm_id="copilot_1"):
    if algorithm_id == "opencode_4":
        items = match_products_opencode_4(checked_products, stock_products, top_n=max(top_n, len(stock_products) or 0))
    elif algorithm_id == "sql_str_3":
        items = match_products_sql_str_3(checked_products, stock_products, top_n=max(top_n, len(stock_products) or 0))
    elif algorithm_id == "sql_str_2":
        items = match_products_sql_str_2(checked_products, stock_products, top_n=max(top_n, len(stock_products) or 0))
    else:
        items = match_products(checked_products, stock_products, top_n=max(top_n, len(stock_products) or 0))

    stock_by_id = {}
    for stock_item in stock_products:
        stock_by_id.setdefault(str(stock_item.get("id") or stock_item.get("sku") or ""), stock_item)
    full_candidate_lists = []
    for entry, checked in zip(items, checked_products):
        query_price = _item_price(checked)
        entry["checked_price"] = float(query_price) if query_price is not None else None
        for candidate in entry["candidates"]:
            stock_match = stock_by_id.get(str(candidate["stock_id"]))
            stock_price = _item_price(stock_match or {})
            candidate["stock_price"] = float(stock_price) if stock_price is not None else None
            difference = abs(stock_price - query_price) if stock_price is not None and query_price is not None else None
            candidate["price_difference"] = float(difference) if difference is not None else None
            candidate["stock_price_display"] = str(stock_price.quantize(Decimal("1"), rounding=ROUND_HALF_UP)) if stock_price is not None else "—"
            percent = difference / query_price * 100 if difference is not None and query_price else (Decimal(0) if difference == 0 else None)
            candidate["price_difference_pct"] = float(percent) if percent is not None else None
            candidate["price_difference_display"] = str(percent.quantize(Decimal("1"), rounding=ROUND_HALF_UP)) + "%" if percent is not None else "—"
            if stock_match is not None:
                candidate["stock_qty"] = (
                    stock_match.get("quantity")
                    or stock_match.get("stock_quantity")
                    or stock_match.get("qty")
                )

        # Price only breaks ties in the displayed name-similarity score.
        entry["candidates"].sort(key=lambda candidate: (
            -candidate["similarity_pct"],
            candidate["price_difference"] is None if query_price is not None else False,
            abs(_item_price(stock_by_id.get(str(candidate["stock_id"]), {})) - query_price)
                if query_price is not None and candidate["price_difference"] is not None else Decimal(0),
            str(candidate["stock_id"]),
        ))
        all_candidates = list(entry["candidates"])
        if min_stock:
            full_main_candidates = [
                candidate
                for candidate in all_candidates
                if float(candidate.get("stock_qty") or 0) >= float(min_stock)
            ]
        else:
            full_main_candidates = list(all_candidates)

        main_candidates = full_main_candidates[:top_n]

        cutoff_score = None
        if main_candidates:
            cutoff_score = min(candidate.get("similarity_pct", 0) for candidate in main_candidates)

        excluded_candidates = []
        for candidate in all_candidates:
            qty = float(candidate.get("stock_qty") or 0)
            if qty < float(min_stock) if min_stock else False:
                if cutoff_score is None or float(candidate.get("similarity_pct") or 0) >= float(cutoff_score):
                    excluded_candidates.append(candidate)

        full_excluded_candidates = list(excluded_candidates)
        entry["candidates"] = main_candidates
        entry["excluded_candidates"] = full_excluded_candidates[:top_n]

        entry["algorithm_id"] = algorithm_id
        entry["algorithm_name"] = ALGORITHM_REGISTRY.get(algorithm_id, {}).get("name", algorithm_id)
        entry["query"] = _extract_query(checked_products)
        entry["best_candidate"] = main_candidates[0] if main_candidates else None
        entry["low_stock_candidate"] = excluded_candidates[0] if excluded_candidates else None
        full_candidate_lists.append((full_main_candidates, full_excluded_candidates))

    first_best = None
    for entry in items:
        if entry.get("best_candidate"):
            first_best = entry["best_candidate"]
            break

    query_value = _extract_query(checked_products)
    result = {
        "status": "ok",
        "top_n": top_n,
        "algorithm_id": algorithm_id,
        "algorithm_name": ALGORITHM_REGISTRY.get(algorithm_id, {}).get("name", algorithm_id),
        "algorithm_title": ALGORITHM_REGISTRY.get(algorithm_id, {}).get("title", algorithm_id),
        "description": ALGORITHM_REGISTRY.get(algorithm_id, {}).get("description", ""),
        "query": query_value,
        "items": items,
        "best_candidate": first_best,
        "low_stock_candidate": next((candidate for entry in items for candidate in entry.get("excluded_candidates", []) if candidate), None),
    }
    full_items = []
    for entry, (full_main, full_excluded) in zip(items, full_candidate_lists):
        full_entry = dict(entry)
        full_entry["candidates"] = full_main
        full_entry["excluded_candidates"] = full_excluded
        full_items.append(full_entry)

    full_result = {**result, "items": full_items}
    result_id = _store_full_result(full_result)
    result["full_results_url"] = f"/analog/list?algorithm_id={quote(algorithm_id)}&result_id={quote(result_id)}"
    return result


def match_debug_payload(checked_products, stock_products, top_n=5, min_stock=0, algorithm_id="copilot_1", algorithms=None):
    if algorithms is None:
        if algorithm_id in (None, "", "all"):
            selected_algorithms = list(ALGORITHM_REGISTRY)
        else:
            selected_algorithms = [algorithm_id]
    else:
        selected_algorithms = list(algorithms)

    if not selected_algorithms:
        selected_algorithms = list(ALGORITHM_REGISTRY)

    results = [
        _build_algorithm_result(checked_products, stock_products, top_n=top_n, min_stock=min_stock, algorithm_id=current_algorithm)
        for current_algorithm in selected_algorithms
    ]

    primary = results[0] if results else {"status": "ok", "items": [], "algorithm_id": "copilot_1", "best_candidate": None}
    query_value = _extract_query(checked_products)
    fallback_query = primary.get("query") if primary else ""
    if not fallback_query and primary and primary.get("items"):
        fallback_query = primary.get("items", [{}])[0].get("checked_name") or ""

    payload = {
        "status": "ok",
        "top_n": top_n,
        "algorithm_id": primary.get("algorithm_id", algorithm_id or "copilot_1"),
        "algorithm_name": primary.get("algorithm_name", algorithm_id or "copilot_1"),
        "query": query_value or fallback_query,
        "best_candidate": primary.get("best_candidate"),
        "items": primary.get("items", []),
        "results_by_algorithm": results,
        "low_stock_candidate": primary.get("low_stock_candidate"),
    }
    return payload


class MatchingDebugHandler(BaseHTTPRequestHandler):
    def _serve_static(self, relative_path: str):
        if relative_path in ("", "/"):
            relative_path = "/index.html"

        safe_path = relative_path.split("?", 1)[0].split("#", 1)[0]
        file_path = (STATIC_ROOT / safe_path.lstrip("/")).resolve()
        try:
            file_path.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self.send_response(403)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "forbidden"}, ensure_ascii=False).encode("utf-8"))
            return

        if not file_path.exists() or not file_path.is_file():
            self.send_response(404)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "not_found"}, ensure_ascii=False).encode("utf-8"))
            return

        content_type, _ = mimetypes.guess_type(str(file_path))
        if not content_type:
            content_type = "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_static("/index.html")
            return

        if parsed.path == "/analog":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if parsed.path == "/analog/list":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(FULL_RESULTS_PAGE.encode("utf-8"))
            return

        if parsed.path == "/analog/list/data":
            query = parse_qs(parsed.query)
            result_id = (query.get("result_id") or [""])[0]
            result = _load_full_result(result_id)
            if result is None:
                self.send_response(404)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "not_found"}, ensure_ascii=False).encode("utf-8"))
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
            return

        if parsed.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}, ensure_ascii=False).encode("utf-8"))
            return

        if parsed.path.startswith("/"):
            try:
                self._serve_static(parsed.path)
                return
            except OSError:
                pass

        self.send_response(404)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "not_found"}, ensure_ascii=False).encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/analog", "/api/matching/debug", "/api/matching/test"):
            self.send_response(404)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "not_found"}, ensure_ascii=False).encode("utf-8"))
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "bad_request", "error": "invalid_json"}, ensure_ascii=False).encode("utf-8"))
            return

        if payload.get("price") not in (None, "") and _price(payload["price"]) is None:
            data = json.dumps({"error": "Цена должна быть неотрицательным числом"}, ensure_ascii=False).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        checked = payload.get("checked") or payload.get("checked_products") or []
        stock = payload.get("stock") or payload.get("stock_products") or []
        query = payload.get("query")
        algorithm_id = payload.get("algorithm_id") or payload.get("algorithm")
        algorithms = payload.get("algorithms")
        if algorithm_id in (None, "", "all") and not algorithms:
            algorithms = list(ALGORITHM_REGISTRY)
        if algorithms is None and algorithm_id not in (None, "", "all"):
            algorithms = [algorithm_id]
        top_n = int(payload.get("top_n", 10) or 10)
        min_stock = float(payload.get("min_stock", 3) or 0)

        if query and not checked:
            checked = [{"id": "manual_input", "name": query, "price": payload.get("price")}]
            stock = load_stock_items(min_stock=0)

        if not checked:
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "bad_request", "error": "checked_is_required"}, ensure_ascii=False).encode("utf-8"))
            return

        if not stock:
            stock = load_stock_items(min_stock=0)

        result = match_debug_payload(checked, stock, top_n=top_n, min_stock=min_stock, algorithm_id=algorithm_id, algorithms=algorithms)
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        return


def run_debug_server(host: str | None = None, port: int | None = None):
    host = host or WEB_HOST
    port = int(port or WEB_PORT)
    from App.web_server import create_server
    server = create_server(host, port)
    print(f"Matching debug server running on http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        server.store.pool.shutdown(wait=True)


if __name__ == "__main__":
    run_debug_server()
