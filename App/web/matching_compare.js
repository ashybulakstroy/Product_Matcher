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
    window.VotingSummary?.clear();
    status.textContent = 'Запускаем алгоритмы параллельно… Первый расчёт каталога может занять несколько минут.';
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
      window.VotingSummary?.render(data.voting, data.results_by_algorithm,
                                    {anchorId: 'matching-comparison-results'});
      output.innerHTML = `<table><thead><tr><th>Алгоритм</th><th>Лучший кандидат</th><th>Недостаточный остаток</th><th>Все кандидаты</th></tr></thead><tbody>${data.results_by_algorithm.map(result => {
        if (result.status !== 'ok') return `<tr><td><strong>${escape(result.algorithm_name)}</strong><small>${Math.round(result.elapsed_ms ?? 0)} мс</small></td><td colspan="3">${escape(result.error)}</td></tr>`;
        const url = result.full_results_url;
        const safeUrl = typeof url === 'string' && url.startsWith('/analog/list?') ? url : '/analog';
        return `<tr><td><strong>${escape(result.algorithm_name)}</strong><small>${Math.round(result.elapsed_ms ?? 0)} мс</small></td><td>${candidate(result.best_candidate)}</td><td>${candidate(result.low_stock_candidate, true)}</td><td><a href="${escape(safeUrl)}" target="_blank" rel="noopener">Открыть Top-5 и полный список</a></td></tr>`;
      }).join('')}</tbody></table>`;
      status.textContent = 'Результаты для «' + input.value.trim() + '» · ' + (data.execution_mode === 'parallel' ? 'Параллельно' : 'Последовательно') + ' · всего ' + Math.round(data.elapsed_ms ?? 0) + ' мс';
    } catch (error) {
      if (current === requestNumber) status.textContent = error.message || 'Не удалось выполнить поиск';
    }
  });
  fetch('/api/matching/algorithms').then(response => response.json()).then(data => {
    if (!requestNumber) status.textContent = data.algorithms.map(item => item.name).join(' · ') + ' · параллельный запуск';
  }).catch(() => {});
  receiptChanged();
})();
