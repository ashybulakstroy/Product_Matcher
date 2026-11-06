/* Voting table for the existing /analog page. No dependencies or network calls.
 * WARNING: radio choice is browser-only until an explicit invoice selection adapter exists.
 */
(function () {
  'use strict';

  const styleId = 'voting-summary-styles';
  function addStyles() {
    if (document.getElementById(styleId)) return;
    const sheet = document.createElement('style');
    sheet.id = styleId;
    sheet.textContent = `
      .voting-summary { margin: 22px 0; padding: 16px; border: 1px solid #cbd5e1; border-radius: 9px; }
      .voting-summary h3 { margin: 0 0 10px; }
      .voting-summary .vote-note { color:#475569; font-size:12px; margin: 7px 0 12px; }
      .voting-summary .vote-scroll { overflow-x:auto; }
      .voting-summary table { border-collapse:collapse; width:100%; margin:10px 0; }
      .voting-summary th,.voting-summary td { border:1px solid #cbd5e1; padding:8px; text-align:left; vertical-align:top; }
      .voting-summary th { background:#f1f5f9; }
      .voting-summary tr.vote-leader { background:#ecfdf5; }
      .voting-summary tr.vote-selected { outline:2px solid #2563eb; outline-offset:-2px; }
      .voting-summary .vote-small { color:#475569; font-size:12px; }
      .voting-summary input[type=radio] { width:auto; margin:0 auto; display:block; }
      .voting-summary .vote-total { font-weight:700; white-space:nowrap; }
      .voting-summary .vote-position { border-top:1px solid #cbd5e1; margin-top:14px; padding-top:12px; }
    `;
    document.head.appendChild(sheet);
  }
  function node(tag, text, className) {
    const el = document.createElement(tag);
    if (text !== undefined && text !== null) el.textContent = String(text);
    if (className) el.className = className;
    return el;
  }
  function appendCell(row, tag, text, className) {
    const cell = node(tag, text, className);
    row.appendChild(cell);
    return cell;
  }
  function format(value) {
    return value === null || value === undefined || value === '' ? '—' : String(value);
  }
  function validListUrl(value) {
    return typeof value === 'string' && value.startsWith('/analog/list?') ? value : null;
  }
  function render(voting, algorithmResults, options = {}) {
    const anchor = document.getElementById(options.anchorId || 'algorithm-results');
    if (!anchor) return;
    let root = document.getElementById('voting-summary');
    if (!root) {
      root = node('section', undefined, 'voting-summary');
      root.id = 'voting-summary';
      anchor.parentNode.insertBefore(root, anchor);
    }
    root.replaceChildren();
    delete root.dataset.selectedStockId;
    addStyles();
    const positions = voting && Array.isArray(voting.positions) ? voting.positions : [];
    if (!positions.length) {
      root.appendChild(node('p', 'Голосование: нет позиций для обработки.'));
      return;
    }
    root.appendChild(node('h3', 'Таблица 0.5. Голосование алгоритмов'));
    root.appendChild(node('p', 'Один алгоритм — один голос. Проценты сходства не усредняются. Выбор ниже является предварительным и не записывается в 1С.', 'vote-note'));
    const links = new Map();
    for (const result of Array.isArray(algorithmResults) ? algorithmResults : []) {
      if (result && result.algorithm_id) links.set(String(result.algorithm_id), validListUrl(result.full_results_url));
    }

    for (const position of positions) {
      const section = node('section', undefined, 'vote-position');
      section.appendChild(node('h4', `Позиция ${position.position_no}: ${format(position.checked_name)}`));
      const status = position.is_tie
        ? `Ничья: ${position.leader_count} лидера. Подтверждённый выбор отсутствует.`
        : position.recommended_stock_id !== null
          ? `Рекомендованный код 1С: ${position.recommended_stock_id}`
          : 'Нет кандидатов для голосования.';
      section.appendChild(node('p', `${position.voting_algorithms} проголосовавших алгоритмов · ${position.unique_candidates} уникальных кандидатов · ${status}`, 'vote-small'));
      const candidates = Array.isArray(position.candidates) ? position.candidates : [];
      if (!candidates.length) { root.appendChild(section); continue; }
      const scroll = node('div', undefined, 'vote-scroll');
      const table = node('table');
      const thead = node('thead');
      const titles = ['Место', 'Код 1С', 'Номенклатура', 'Голоса', 'Доля голосов', 'Алгоритмы', 'Остаток', 'Цена 1С', 'Выбрать'];
      const header = node('tr');
      for (const title of titles) appendCell(header, 'th', title);
      thead.appendChild(header);
      table.appendChild(thead);
      const tbody = node('tbody');
      const selectedText = node('p', 'Пользователь пока не выбрал товар.', 'vote-small');
      for (const candidate of candidates) {
        const tr = node('tr');
        if ((position.leader_stock_ids || []).includes(candidate.stock_id)) tr.classList.add('vote-leader');
        appendCell(tr, 'td', format(candidate.rank));
        appendCell(tr, 'td', format(candidate.stock_id));
        appendCell(tr, 'td', format(candidate.stock_name));
        appendCell(tr, 'td', format(candidate.votes), 'vote-total');
        appendCell(tr, 'td', `${format(candidate.vote_pct)}%`);
        const votersCell = appendCell(tr, 'td');
        for (const [index, algorithm] of (candidate.voters || []).entries()) {
          if (index) votersCell.appendChild(document.createTextNode(', '));
          const url = links.get(algorithm);
          if (url) {
            const link = node('a', algorithm);
            link.href = url;
            link.target = '_blank';
            link.rel = 'noopener';
            votersCell.appendChild(link);
          } else {
            votersCell.appendChild(document.createTextNode(algorithm));
          }
        }
        appendCell(tr, 'td', format(candidate.stock_qty));
        const roundedPrice = candidate.stock_price_display ?? (
          typeof candidate.stock_price === 'number' && Number.isFinite(candidate.stock_price)
            ? Math.round(candidate.stock_price) : null);
        const priceCell = appendCell(tr, 'td', format(roundedPrice));
        priceCell.appendChild(node('div', `Отклонение: ${format(candidate.price_difference_display)}`, 'vote-small'));
        const chooseCell = appendCell(tr, 'td');
        const radio = node('input');
        radio.type = 'radio';
        radio.name = `vote-position-${position.position_no}`;
        radio.value = candidate.stock_id;
        radio.setAttribute('aria-label', `Выбрать ${candidate.stock_name}, код ${candidate.stock_id}`);
        radio.addEventListener('change', () => {
          for (const row of tbody.querySelectorAll('tr')) row.classList.remove('vote-selected');
          tr.classList.add('vote-selected');
          root.dataset.selectedStockId = candidate.stock_id;
          // Event intended for future invoice-picker integration; no write happens here.
          root.dispatchEvent(new CustomEvent('product-matcher:vote-selection', {
            bubbles: true,
            detail: { position_no: position.position_no, checked_id: position.checked_id,
                      selected_stock_id: candidate.stock_id, selected_stock_name: candidate.stock_name }
          }));
          selectedText.textContent = `Предварительно выбран товар 1С: ${candidate.stock_id} — ${candidate.stock_name}. В 1С не отправлено.`;
        });
        chooseCell.appendChild(radio);
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      scroll.appendChild(table);
      section.appendChild(scroll);
      section.appendChild(selectedText);
      root.appendChild(section);
    }
  }
  function clear() {
    document.getElementById('voting-summary')?.remove();
  }
  window.VotingSummary = { render, clear };
})();
