/* Отдельный виджет активности: не придумывает проценты, товары или OCR-результаты. */
(function () {
  'use strict';
  const el = id => document.getElementById(id);
  let timer = null, started = 0, lastAction = 0, lastHeartbeat = 0, working = false;
  let connected = false;
  function seconds(ms) { return Math.floor(Math.max(0, ms) / 1000); }
  function tick() {
    if (!working) return;
    const now = Date.now();
    el('ocr-activity-elapsed').textContent = `Прошло ${seconds(now-started)} с`;
    el('ocr-activity-last').textContent = `Новое действие ${seconds(now-lastAction)} с назад`;
    const silence = seconds(now-lastAction);
    const idle = el('ocr-activity-idle');
    idle.hidden = silence < 12;
    if (!idle.hidden) idle.textContent = silence >= 30
      ? 'Долго нет новых результатов. Операция может быть сложной.'
      : 'Обработка продолжается; новых результатов пока нет.';
    const live = connected && now-lastHeartbeat < 30000;
    el('ocr-activity-connection').textContent = live ? 'Сервер отвечает' : 'Связь не подтверждена';
    el('ocr-activity-dot').classList.toggle('disconnected', !live);
  }
  function action(title, details = '') {
    if (!working) return;
    if (title) el('ocr-activity-action').textContent = title;
    if (details) el('ocr-activity-details').textContent = details;
    lastAction = Date.now();
    tick();
  }
  function start() {
    started = lastAction = Date.now(); lastHeartbeat = 0;
    working = true; connected = false;
    el('ocr-activity').hidden = false;
    el('ocr-activity-dot').classList.remove('done');
    action('Начинаем обработку', 'Данные будут появляться по мере выполнения этапов');
    if (timer) clearInterval(timer);
    timer = setInterval(tick, 1000);
    tick();
  }
  function finish(ok) {
    working = false;
    clearInterval(timer); timer = null;
    el('ocr-activity-idle').hidden = true;
    el('ocr-activity').hidden = false;
    el('ocr-activity-dot').classList.toggle('done', ok);
    el('ocr-activity-dot').classList.toggle('disconnected', !ok);
    el('ocr-activity-action').textContent = ok ? 'Обработка завершена' : 'Обработка остановлена';
    el('ocr-activity-connection').textContent = ok ? 'Результат готов' : 'Требуется проверка';
  }
  function heartbeat() {
    connected = true; lastHeartbeat = Date.now(); tick();
  }
  function connectionLost() {
    connected = false; tick();
  }
  function uiStatus(event) {
    let details = event.details || '';
    if (Number.isInteger(event.rows_total) && event.rows_total > 0
        && Number.isInteger(event.rows_done) && event.rows_done >= 0) {
      details = `Обработано ${event.rows_done} из ${event.rows_total} строк чека`;
    } else if (Number.isInteger(event.matched_products) && Number.isInteger(event.total_products)
               && event.total_products > 0) {
      details = `Сопоставлено ${event.matched_products} из ${event.total_products} товаров`;
    }
    action(event.title, details);
  }
  function rows(rows) {
    if (!Array.isArray(rows) || !rows.length) return;
    const products = rows.filter(row => row.kind === 'product').length;
    const taxes = rows.filter(row => row.kind === 'tax').length;
    action('Формируем товарные строки',
      `Предварительно найдено ${products} товаров${taxes ? `, строк НДС: ${taxes}` : ''}`);
  }
  function phase(phase, title) {
    const steps = {input:'Проверяем фотографию',geometry:'Исправляем геометрию таблицы',
      layout:'Определяем строки и столбцы',ocr:'Распознаём товары, количество и цены',
      matching:'Сверяем товары со справочником',queued:'Ожидаем начала обработки'};
    action(title || steps[phase], steps[phase] || 'Обрабатываем чек');
  }
  window.OcrActivity = {start, finish, heartbeat, connectionLost, uiStatus, rows, phase,
    message: title => action(title), row: title => action(title)};
})();
