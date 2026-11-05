(() => {
  const byId = id => document.getElementById(id);
  const form = byId('matching-compare-form');
  const select = byId('matching-receipt-row');
  const input = byId('matching-query');
  const output = byId('matching-comparison-results');
  const status = byId('matching-status');
  let requestNumber = 0;
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function receiptChanged() {
    const previous = select.value;
    select.replaceChildren(new Option('Введите название вручную', ''));
    for (const row of state.result?.rows || []) {
      if (row.kind !== 'product' && row.kind !== 'service') continue;
      const name = row.name_raw || row.name || '';
      if (name) select.add(new Option(`${row.row}. ${name}`, name));
    }
    select.value = previous;
  }
  window.addEventListener('receipt-result-updated', receiptChanged);
  select.addEventListener('change', () => {
    if (select.value) { input.value = select.value;
      const row = (state.result?.rows || []).find(row => (row.name_raw || row.name || '') === select.value);
      byId('matching-price').value = String(row?.price ?? row?.price_raw ?? '').replace(',', '.');
      form.requestSubmit(); }
  });
  function candidate(value, low = false) {
    if (!value) return 'Нет кандидатов';
    return `<span class="${low ? 'insufficient' : ''}">${escape(value.stock_name)}</span>
      <small>Цена: ${escape(value.stock_price_display ?? '—')} · Отклонение: ${escape(value.price_difference_display ?? '—')}</small>
      <small>Оценка: ${escape(value.similarity_pct)}% · Остаток: ${escape(value.stock_qty ?? '—')}</small>
      <small>${(value.match_reasons || []).map(escape).join('; ')}</small>`;
  }
  form.addEventListener('submit', async event => {
    event.preventDefault();
    const current = ++requestNumber;
    status.textContent = 'Сравниваем copilot_1, sql_str_2, sql_str_3 и opencode_4…';
    output.replaceChildren();
    try {
      const response = await fetch('/api/matching/test', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({query: input.value.trim(), price:byId('matching-price').value, algorithm_id:'all', top_n:5,
          min_stock:Number(byId('matching-min-stock').value)})
      });
      if (!response.ok) throw Error('Ошибка поиска: ' + response.status);
      const data = await response.json();
      if (current !== requestNumber) return;
      output.innerHTML = `<table><thead><tr><th>Алгоритм</th><th>Лучший кандидат</th><th>Недостаточный остаток</th><th>Все кандидаты</th></tr></thead><tbody>${data.results_by_algorithm.map(result => {
        const url = result.full_results_url;
        const safeUrl = typeof url === 'string' && url.startsWith('/analog/list?') ? url : '/analog';
        return `<tr><td><strong>${escape(result.algorithm_name)}</strong></td><td>${candidate(result.best_candidate)}</td><td>${candidate(result.low_stock_candidate, true)}</td><td><a href="${escape(safeUrl)}" target="_blank" rel="noopener">Открыть Top-5 и полный список</a></td></tr>`;
      }).join('')}</tbody></table>`;
      status.textContent = 'Результаты для «' + input.value.trim() + '». Процент — оценка алгоритма, а не вероятность.';
    } catch (error) {
      if (current === requestNumber) status.textContent = error.message || 'Не удалось выполнить поиск';
    }
  });
  receiptChanged();
})();
