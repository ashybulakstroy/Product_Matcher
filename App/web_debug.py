import json
from html import escape as _html_escape
import mimetypes
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from App.config import WEB_HOST, WEB_PORT
from App.matching.registry import ALGORITHM_REGISTRY
from App.matching.cache import _load_full_result, clear_full_result_cache
from App.matching.ranking import _price
from App.matching.engine import build_algorithm_result as _build_algorithm_result, match as match_debug_payload


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
      .search-running::before{content:'';display:inline-block;width:14px;height:14px;margin-right:8px;border:2px solid #cbd5e1;border-top-color:#2563eb;border-radius:50%;animation:search-spin .8s linear infinite;vertical-align:middle}
      @keyframes search-spin{to{transform:rotate(360deg)}}
      @media(prefers-reduced-motion:reduce){.search-running::before{animation:none}}
    </style>
  </head>
  <body>
    <div class="panel">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:16px; flex-wrap:wrap;">
        <h2 style="margin:0;">Интерактивный подбор аналогов</h2>
        <a href="/analog" style="font-weight:700; color:#1d4ed8; text-decoration:none;">Ашыбулак строй / /analog</a>
      </div>
      <p class="small">Цена участвует в ранжировании: её влияние ограничено 3 пунктами оценки, исходный процент сходства не меняется. Сравнение с учётной ценой 1С; цены должны быть в одной валюте и за одинаковую единицу. Введите товар из Торгсофт, задайте минимальный остаток и количество кандидатов. Ответ будет показан в таблице.</p>

      <form id="match-form" onsubmit="event.preventDefault(); runMatch(); return false;">
        <div class="form-grid">
          <div>
            <label for="query">Название товара</label>
            <div style="display:flex;gap:6px"><input id="query" type="search" value="Тачка садовая одноколесная усиленная ЭКСПЕРТ" /><button type="button" id="clear-query" aria-label="Очистить строку поиска" style="width:auto">×</button></div>
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
        <label style="margin-top:12px;font-weight:400"><input type="checkbox" id="price-search" style="width:auto"> По цене: от указанной цены и выше, по возрастанию. Название необязательно.</label>
        <details style="margin-top:14px; border:1px solid #d1d5db; border-radius:8px; padding:10px 12px;">
          <summary style="cursor:pointer; color:#51617b;">Настройки поиска</summary>
          <div class="form-grid" style="margin-top:12px; grid-template-columns:1fr 1fr;">
          <div>
            <label for="algorithm_id">Алгоритм</label>
            <select id="algorithm_id">
              <option value="all" selected>Все алгоритмы</option>
              __ALGORITHM_OPTIONS__
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

          <label style="display:flex;align-items:center;gap:8px;margin-top:12px;font-weight:400;"><input id="parallel-search" type="checkbox" checked style="width:auto;">Параллельный запуск алгоритмов</label>
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
          const candidate = algorithmResult && algorithmResult.best_candidate ? { ...algorithmResult.best_candidate, algorithm_id: algorithmId, full_results_url: algorithmResult.full_results_url, elapsed_ms: algorithmResult.elapsed_ms } : null;
          const lowStockCandidate = algorithmResult && algorithmResult.low_stock_candidate ? { ...algorithmResult.low_stock_candidate, algorithm_id: algorithmId, full_results_url: algorithmResult.full_results_url, elapsed_ms: algorithmResult.elapsed_ms } : null;

          if (candidate) {
            positiveRows.push(candidate);
          } else if (algorithmResult.status === 'ok') {
            positiveRows.push({algorithm_id: algorithmId, elapsed_ms: algorithmResult.elapsed_ms, no_match: true});
          }
          if (lowStockCandidate) {
            deficientRows.push(lowStockCandidate);
          }
        });

        box.textContent = 'ТОП-1 по всем алгоритмам · ' + (data.execution_mode === 'parallel' ? 'Параллельно' : 'Последовательно') + ' · всего ' + Math.round(data.elapsed_ms ?? 0) + ' мс';
        const unavailable = data.results_by_algorithm.filter(result => result.status !== 'ok');
        if (unavailable.length) box.textContent += ' · ' + unavailable.map(result => result.algorithm_id + ': ' + result.error).join(' · ');

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
                    <td>${candidate.algorithm_id || '-'}<div class="small">${Math.round(candidate.elapsed_ms ?? 0)} мс</div></td>
                    ${candidate.no_match ? '<td colspan="5" class="small">Нет кандидатов с достаточным остатком</td>' : `
                    <td>${candidate.stock_name || '-'}</td>
                    <td>${candidate.stock_price_display ?? \'—\'}<div class="small">Отклонение: ${candidate.price_difference_display ?? \'—\'}</div></td>
                    <td>${candidate.stock_qty ?? '-'}</td>
                    <td>${candidate.similarity_pct ?? '-'}</td>
                    <td><a href="${candidate.full_results_url || '#'}">Открыть</a></td>
                    `}
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
                    <td>${candidate.algorithm_id || '-'}<div class="small">${Math.round(candidate.elapsed_ms ?? 0)} мс</div></td>
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

      function renderPricePage(data,filters){
        const host=document.getElementById('algorithm-results');host.replaceChildren();
        const section=document.createElement('section');
        const title=document.createElement('h3');title.textContent='Таблица 1. Товары из остатков по цене';section.append(title);
        const table=document.createElement('table');table.id='price-results-table';
        const header=table.createTHead().insertRow();
        for(const text of ['№','Товар 1С','Цена за единицу (1С)','Остаток',...(window.AnalogPicker?['Выбрать']:[])]){const th=document.createElement('th');th.textContent=text;header.append(th);}
        const body=table.createTBody();
        for(const [index,item] of data.items.entries()){
          const row=body.insertRow();
          for(const value of [(data.page-1)*100+index+1,item.name,Math.round(Number(item.accounting_price)),Math.round(Number(item.quantity))])row.insertCell().textContent=value??'—';
          if(window.AnalogPicker){const button=document.createElement('button');button.textContent='Выбрать';button.addEventListener('click',()=>window.AnalogPicker.choose(item.id));row.insertCell().append(button);}
        }
        section.append(table);
        const nav=document.createElement('nav');nav.setAttribute('aria-label','Страницы результатов поиска');nav.style='display:flex;align-items:center;gap:12px;margin:16px 0;flex-wrap:wrap';
        const previous=document.createElement('button');previous.type='button';previous.textContent='← Назад';previous.disabled=data.page<=1;previous.style.width='auto';previous.onclick=()=>runMatch(data.page-1,filters);
        const label=document.createElement('span');label.textContent=`Страница ${data.page} из ${data.pages} · Всего товаров: ${data.total} · По 100 на странице`;
        const select=document.createElement('select');select.setAttribute('aria-label','Перейти на страницу');select.style.width='auto';
        for(let page=1;page<=data.pages;page++)select.add(new Option(String(page),String(page),false,page===data.page));
        select.onchange=()=>runMatch(Number(select.value),filters);
        const next=document.createElement('button');next.type='button';next.textContent='Вперёд →';next.disabled=data.page>=data.pages;next.style.width='auto';next.onclick=()=>runMatch(data.page+1,filters);
        nav.append(previous,label,select,next);section.append(nav);host.append(section);
        const excluded=document.createElement('section');const excludedTitle=document.createElement('h3');excludedTitle.textContent='Таблица 2. Дефициты / недостаточный остаток';excluded.append(excludedTitle);
        const empty=document.createElement('table');empty.id='price-excluded-table';empty.createTBody();excluded.append(empty);host.append(excluded);
        document.getElementById('result-box').textContent='Найдено: '+data.total+' · Прямой отбор из остатков, без алгоритмов аналогов';
      }

      const pickerToken=new URLSearchParams(location.search).get('picker');
      if(pickerToken){
        const channel=new BroadcastChannel('invoice-analog-'+pickerToken);
        let choosing=false;
        window.AnalogPicker={choose(id){
          if(choosing)return;
          choosing=true;
          document.getElementById('result-box').textContent='Сохраняем выбранный аналог в квитанции…';
          channel.postMessage({type:'choose',id:String(id)});
          setTimeout(()=>{if(choosing){choosing=false;document.getElementById('result-box').textContent='Родительское окно не ответило. Проверьте, что квитанция открыта, и повторите выбор.';}},30000);
        }};
        channel.onmessage=event=>{
          if(event.data?.type==='accepted'){choosing=false;window.close();}
          if(event.data?.type==='error'){choosing=false;document.getElementById('result-box').textContent=event.data.message;}
        };
      }
      let searchBusy=false;
      async function runMatch(pageNumber=1,savedFilters=null) {
        if(searchBusy)return;
        const query = savedFilters?savedFilters.query:document.getElementById('query').value.trim();
        const min_stock = savedFilters?savedFilters.min_stock:Number(document.getElementById('min_stock').value || 0);
        const price = savedFilters?savedFilters.price:document.getElementById('query-price').value;
        const priceMode=savedFilters?true:document.getElementById('price-search').checked;
        const top_n = Number(document.getElementById('top_n').value || 10);
        const algorithm_id = document.getElementById('algorithm_id').value || 'all';

        if (!query && !priceMode) {
          document.getElementById('result-box').textContent = 'Введите название товара';
          return;
        }

        window.VotingSummary?.clear();
        if(priceMode && price===''){
          document.getElementById('result-box').textContent='Укажите начальную цену';return;
        }
        searchBusy=true;
        const box=document.getElementById('result-box');
        const submit=document.querySelector('#match-form button[type=submit]');
        box.textContent='Поиск идёт…';box.classList.add('search-running');box.setAttribute('aria-busy','true');submit.disabled=true;
        try{ 
        const response = await fetch('/analog', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, price, min_stock, top_n, price_search:priceMode, page:pageNumber, job_id:new URLSearchParams(location.search).get('job_id'), algorithm_id: algorithm_id === 'all' ? null : algorithm_id, parallel: document.getElementById('parallel-search').checked })
        });

        const data = await response.json();
        if(!response.ok){document.getElementById('result-box').textContent=data.error||'Ошибка поиска';return;}
        if(data.price_search){
          renderPricePage(data,{query,price,min_stock});return;
        }
        window.VotingSummary?.render(data.voting, data.results_by_algorithm);
        renderTable(data);
        }catch(error){box.textContent='Не удалось выполнить поиск: '+error.message;}
        finally{searchBusy=false;box.classList.remove('search-running');box.setAttribute('aria-busy','false');submit.disabled=false;}
      }

      document.addEventListener('DOMContentLoaded', function () {
        const params=new URLSearchParams(location.search);
        for(const [key,id] of [['query','query'],['price','query-price'],['min_stock','min_stock']]){
          if(params.has(key))document.getElementById(id).value=params.get(key);
        }
        document.getElementById('clear-query').addEventListener('click',()=>{const input=document.getElementById('query');input.value='';input.focus();});
        if(params.get('query')?.trim())runMatch();
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
    <script src="/voting.js"></script>
  </body>
</html>
"""

HTML_PAGE = HTML_PAGE.replace('__ALGORITHM_OPTIONS__', '\n'.join(
    f'<option value="{_html_escape(key, quote=True)}">{_html_escape(meta["name"])}</option>'
    for key, meta in ALGORITHM_REGISTRY.items()
))

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
            heading.innerHTML = `<h2>${escapeHtml(queryText)}</h2><p class="small">Алгоритм: ${escapeHtml(algorithmId)} · ${Math.round(data.elapsed_ms ?? 0)} мс</p>`;
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


def price_search(query, minimum_price, stock, min_stock=0):
    """Literal case-insensitive name containment, inclusive price floor, no Top-X cut."""
    floor = _price(minimum_price)
    if floor is None:
        raise ValueError('Укажите неотрицательную начальную цену')
    query = str(query or '').strip().casefold()
    items = [item for item in stock
             if query in str(item.get('name') or '').casefold()
             and _price(item.get('accounting_price')) is not None
             and _price(item['accounting_price']) >= floor
             and float(item.get('quantity') or 0) >= min_stock]
    items.sort(key=lambda item: (_price(item['accounting_price']), str(item.get('name') or '').casefold()))
    return dict(price_search=True, items=items)


def price_catalog_page(query, minimum_price, stock, min_stock=0, page=1):
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError('Номер страницы должен быть положительным целым числом')
    result = price_search(query, minimum_price, stock, min_stock)
    total = len(result['items'])
    pages = max(1, (total + 99) // 100)
    page = min(page, pages)
    # Only portable fields of the current page cross the JSON boundary.
    from App.invoicing.invoice_service import _plain
    items = [dict(id=item.get('id'), name=item.get('name'),
                  accounting_price=item.get('accounting_price'), quantity=item.get('quantity'))
             for item in result['items'][(page-1)*100:page*100]]
    return dict(price_search=True, items=_plain(items), total=total, page=page,
                page_size=100, pages=pages, results_by_algorithm=[])


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
        if parsed.path == "/api/matching/algorithms":
            data = json.dumps({"algorithms": list(ALGORITHM_REGISTRY.values()), "parallel_default": True}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
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

        scoped_stock = None
        if payload.get('job_id') and hasattr(self, 'invoice_context'):
            job = self.server.store.get(payload['job_id'])
            if job is None:
                return self.send_json({'error': 'Обработка не найдена'}, 404)
            from App.invoicing import catalog_1c
            from App.invoicing.matching_bridge import validate_database
            database_id = self.invoice_context(job)
            with catalog_1c.catalog_scope(database_id) as index:
                try:
                    validate_database(index, database_id)
                except ValueError as error:
                    return self.send_json({'error': str(error)}, 400)
                scoped_stock = [dict(item, quantity=float(item.get('stock_quantity') or 0),
                                     accounting_price=float(item['accounting_price']) if item.get('accounting_price') is not None else None)
                                for item in index['items']]
        if payload.get('price_search'):
            try:
                result = price_catalog_page(query, payload.get('price'), scoped_stock if scoped_stock is not None else load_stock_items(), min_stock, payload.get('page', 1))
            except ValueError as error:
                result = {'error': str(error)}
            data = json.dumps(result, ensure_ascii=False).encode('utf-8')
            self.send_response(400 if 'error' in result else 200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if query and not checked:
            checked = [{"id": "manual_input", "name": query, "price": payload.get("price")}]
            stock = scoped_stock if scoped_stock is not None else load_stock_items(min_stock=0)

        if not checked:
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "bad_request", "error": "checked_is_required"}, ensure_ascii=False).encode("utf-8"))
            return

        if not stock:
            stock = scoped_stock if scoped_stock is not None else load_stock_items(min_stock=0)

        try:
            result = match_debug_payload(checked, stock, top_n=top_n, min_stock=min_stock,
                                         algorithm_id=algorithm_id, algorithms=algorithms,
                                         parallel=payload.get("parallel", True))
        except ValueError as error:
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(error)}, ensure_ascii=False).encode("utf-8"))
            return
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
        from App.matching.cpu_executor import shutdown_executors
        shutdown_executors()


if __name__ == "__main__":
    run_debug_server()
