'use strict';
/* Invoice panel (Etap 5). Candidates are shown, but only the operator confirms a link.
   Reads the shared page state and helpers from app.js; touches no OCR behavior. */
(function invoicePanel(){
const $ = id => document.getElementById(id);
const TASK=window.OCRTask||null;
const INV = {jobId:null, revision:null, model:null, picks:new Map(), busy:false, dirty:false, pdfUrl:null, analogs:new Map(), openRows:new Set()};

function clearPdf(){
  const frame=$('invoice-print-preview');
  frame.hidden=true;frame.removeAttribute('src');
  if(INV.pdfUrl)URL.revokeObjectURL(INV.pdfUrl);
  INV.pdfUrl=null;
}
function markDirty(){
  $('ocr-confirm-success').hidden=true;
  INV.dirty=true;clearPdf();
  $('invoice-pdf').hidden=true;
  $('invoice-train').disabled=true;
  $('invoice-pdf-hint').textContent='Данные изменены: сохраните счёт или постройте предпросмотр.';
}

function money(value){
  if(value===null||value===undefined||value==='')return '—';
  const number=Number(String(value).replace(/\s+/g,'').replace(/,/g,'.'));
  if(!Number.isFinite(number))return String(value);
  return number.toLocaleString('ru-RU',{minimumFractionDigits:2,maximumFractionDigits:2})+' ₸';
}
function qty(value){
  const number=Number(String(value??'').replace(/,/g,'.'));
  return Number.isFinite(number)?number.toLocaleString('ru-RU',{maximumFractionDigits:9}):'—';
}
function result(){
  try{return state.result||null;}catch(error){return null;}
}
function badge(text,kind){
  const span=node('span','inv-badge '+(kind||''),text);return span;
}

async function fetchModel(){
  try{
    const response=await fetch(`/api/jobs/${INV.jobId}/invoice`);
    if(response.ok)return await response.json();
    let message='Счёт не построен';
    try{const data=await response.json();if(data&&data.error)message=data.error;}catch(error){}
    toast(message);
    return null;
  }catch(error){toast('Счёт недоступен: '+error.message);return null;}
}

function pickFromRows(model){
  INV.picks=new Map();
  for(const row of model.generation.items){
    if(row.match_status==='manual'||row.match_status==='remembered'){
      INV.picks.set(String(row.row_id),{name:row.name,unit:row.unit||'шт',
        quantity:String(row.quantity||'1'),service:!!row.service,
        catalog_id:row.catalog_id||null,code:row.code||null});
    }
  }
}

function selectionPayload(){
  const selections={};
  for(const [rowId,pick] of INV.picks)selections[String(rowId)]={
    name:pick.name,unit:pick.unit,quantity:pick.quantity,service:!!pick.service,
    catalog_id:pick.catalog_id||null,code:pick.code||null};
  return {selections};
}

function pickLabel(row){
  const pick=INV.picks.get(String(row.row_id));
  if(pick)return pick.name+' · '+pick.unit+' · '+qty(pick.quantity);
  if(row.match_status==='learned')return (row.name||'')+' · обучено';
  if(row.match_status==='suggested')return (row.name||'')+' · предложено';
  return null;
}

function renderRow(item,index){
  const rowKey=String(item.row_id);
  const li=node('li','inv-row');
  const details=node('details','inv-row-details');
  const head=node('summary','inv-row-head');
  // Buttons inside summary must not toggle the row.
  head.addEventListener('click',event=>{
    if(event.target.closest('button,a,input,select,textarea'))event.preventDefault();
  });
  const left=node('div','inv-row-title');
  const number=node('span','inv-row-number','#'+(index+1));
  const name=node('button','inv-row-name',item.source_name||'—');
  name.type='button';name.title='Показать строку в чеке';
  name.addEventListener('click',event=>{
    event.preventDefault();event.stopPropagation();
    try{focusRow(Number(item.row_id));}catch(error){}
  });
  const amount=node('span','inv-row-amount',money(item.source_amount));
  left.append(number,name,amount);
  const chosen=node('div','inv-row-chosen');
  chosen.append(badge(item.match_status==='unmatched'?'Товар 1С не определён':item.match_status==='suggested'?'Примерное соответствие':item.match_status,
                      item.match_status==='unmatched'?'missing':'ok'));
  const known=pickLabel(item);
  if(known)chosen.append(node('span','inv-row-known',known));
  head.append(left,chosen);

  const body=node('div','inv-row-body');
  const search=node('input','inv-search');
  search.type='search';search.placeholder='Поиск в справочнике 1С (название или код)';
  search.setAttribute('aria-label','Поиск 1С для строки '+(item.source_name||''));
  const list=node('div','inv-options');
  const draw=options=>{list.replaceChildren();for(const option of options.slice(0,10))list.append(optionButton(option,item));};
  let timer=null;
  search.addEventListener('input',()=>{
    clearTimeout(timer);const query=search.value.trim();
    timer=setTimeout(async()=>{
      if(query.length<2){draw(invCandidates(item));return;}
      try{
        const response=await fetch('/api/catalog-1c?q='+encodeURIComponent(query)+(TASK?'&task_id='+encodeURIComponent(TASK.id):''));
        if(!response.ok)return;
        const data=await response.json();
        const fresh=(data.items||[]).map(entry=>({rowId:item.row_id,sourceQuantity:item.source_quantity,
          origin:'search',code:entry.code,name:entry.name,unit:entry.unit,service:entry.service,
          stock_quantity:entry.stock_quantity,accounting_price:entry.accounting_price,catalog_id:entry.id,score:entry.score}));
        draw(fresh.length?fresh:invCandidates(item));
      }catch(error){/* поиск недоступен — показанные кандидаты остаются */}
    },300);
  });
  body.append(search,list);
  // OCR_ANALOG_BRIDGE_V1: suggestions stay local until explicit save.
  if(!INV.analogs.has(rowKey))INV.analogs.set(rowKey,{busy:false,options:null,message:''});
  const aiState=INV.analogs.get(rowKey);
  const aiTools=node('div','inv-ai-tools');
  const aiVote=node('button','button button-white','ИИ');
  aiVote.type='button';aiVote.title='Голосование алгоритмов и предварительный выбор';
  const aiSearch=node('button','button button-white','ИИ поиск');
  aiSearch.type='button';aiSearch.title='Кандидаты голосования для ручного выбора';
  const aiStatus=node('span','inv-ai-status',aiState.message);
  aiStatus.setAttribute('role','status');aiStatus.setAttribute('aria-live','polite');
  const updateAI=()=>{
    aiVote.disabled=aiSearch.disabled=INV.busy||aiState.busy;
    aiStatus.textContent=aiState.message;
  };
  const runAI=async mode=>{
    const currentJob=INV.jobId,currentRevision=INV.revision;
    if(!currentJob||aiState.busy||INV.busy)return;
    aiState.busy=true;aiState.message='Подбираем аналоги…';updateAI();
    try{
      const response=await fetch(`/api/jobs/${encodeURIComponent(currentJob)}/analogs`,{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({row_id:rowKey,receipt_revision:result()?.revision??0})});
      const data=await response.json();
      if(!response.ok)throw new Error(data.error||'Ошибка поиска');
      if(INV.jobId!==currentJob||INV.revision!==currentRevision)return;
      const currentItem=INV.model?.generation.items.find(row=>String(row.row_id)===rowKey)||item;
      const options=(data.items||[]).map(candidate=>({...candidate,rowId:rowKey,
        sourceQuantity:currentItem.source_quantity,origin:'analog_vote',catalog_id:candidate.id}));
      aiState.options=options;
      const protectedPick=INV.picks.has(rowKey)||['manual','remembered','learned'].includes(currentItem.match_status);
      const winner=options.find(candidate=>String(candidate.catalog_id)===String(data.recommended_id));
      if(mode==='vote'&&winner&&!data.is_tie&&!protectedPick){
        aiState.message=`Лидер голосования: ${winner.name}. Выбор предварительный — сохраните связи отдельно.`;
        optionButton(winner,currentItem).click();
        return;
      }
      aiState.message=data.is_tie?'Ничья голосования — выберите товар вручную.':
        !options.length?'Аналоги с достаточным остатком не найдены.':
        mode==='vote'&&protectedPick?'Текущая связь сохранена. Для замены выберите товар вручную.':
        `Найдено аналогов: ${options.length}. Выберите товар вручную.`;
      if(data.errors?.length)aiState.message+=' Часть алгоритмов недоступна.';
      draw(options.length?options:invCandidates(currentItem));
    }catch(error){
      if(INV.jobId===currentJob&&INV.revision===currentRevision)aiState.message='ИИ поиск: '+error.message;
    }finally{
      aiState.busy=false;
      if(INV.jobId===currentJob&&INV.revision===currentRevision){
        if(li.isConnected)updateAI();else refresh();
      }
    }
  };
  aiVote.addEventListener('click',()=>runAI('vote'));
  aiSearch.addEventListener('click',()=>runAI('list'));
  updateAI();aiTools.append(aiVote,aiSearch,aiStatus);body.append(aiTools);
  const picked=INV.picks.get(String(item.row_id));
  if(picked)body.append(selectionBar(item,picked));
  draw(aiState.options?.length?aiState.options:invCandidates(item));
  details.append(head,body);
  details.open=INV.openRows.has(rowKey);
  details.addEventListener('toggle',()=>{
    if(details.open)INV.openRows.add(rowKey);else INV.openRows.delete(rowKey);
  });
  li.append(details);
  return li;
}

function selectionBar(item,picked){
  const bar=node('div','inv-selection');
  bar.append(node('span','inv-selection-note','Выбрано'));
  const qtyInput=node('input','inv-qty');
  qtyInput.inputMode='decimal';qtyInput.value=picked.quantity;
  qtyInput.setAttribute('aria-label','Количество в 1С для '+picked.name);
  qtyInput.addEventListener('change',()=>{
    const value=String(qtyInput.value).replace(',','.').trim();
    const number=Number(value);
    if(Number.isFinite(number)&&number>0){picked.quantity=value;markDirty();refresh();}
    else{qtyInput.value=picked.quantity;toast('Количество в 1С должно быть положительным числом');}
  });
  bar.append(node('span','inv-selection-unit',picked.unit),qtyInput);
  const clear=node('button','inv-clear','Сбросить');
  clear.type='button';
  clear.addEventListener('click',()=>{INV.picks.delete(String(item.row_id));markDirty();refresh();});
  bar.append(clear);
  return bar;
}

function priceComparison(receiptPrice,catalogPrice){
  const parse=value=>value===null||value===undefined||String(value).trim()===''?NaN:Number(String(value).replace(/\s+/g,'').replace(/,/g,'.'));
  const receipt=parse(receiptPrice),catalog=parse(catalogPrice);
  if(!Number.isFinite(receipt)||receipt<0||!Number.isFinite(catalog)||catalog<=0)return null;
  const percent=Math.abs(receipt-catalog)/catalog*100;
  return {percent,label:percent<50?'Норма':percent<=100?'Внимание. Налог':'БОЛЬШОЙ НАЛОГ ИПН!!!'};
}

function optionButton(option,item){
  const button=node('button','inv-option');
  button.type='button';
  button.append(node('span','inv-option-name',option.name||'—'),
                node('span','inv-option-meta',[
                  option.origin==='analog_vote'?`Голосование: ${option.votes??0} голосов, ${option.vote_pct??0}% · ${(option.voters||[]).join(', ')}`:option.origin==='ai_matching'?'ИИ-сопоставление':option.origin==='proposal'?'в предложениях':option.origin==='remembered'?'подтверждено ранее':'справочник',
                  option.code?'код '+option.code:null,
                  option.unit?option.unit:null,
                  option.stock_quantity!=null?('остаток '+option.stock_quantity):'остаток unknown'
                ].filter(Boolean).join(' · ')));
  const comparison=priceComparison(item.source_price,option.accounting_price);
  const detail=node('span','inv-option-meta',
    'Цена 1С: '+money(option.accounting_price)+' · Цена эл. чека: '+money(item.source_price)+' · '+
    (comparison?'Разница: '+comparison.percent.toLocaleString('ru-RU',{maximumFractionDigits:2})+'%':'Разница: — (нет цены или цена 1С равна нулю)'));
  detail.title='Абсолютная разница: |цена эл. чека − цена 1С| / цена 1С × 100%. Сравнение за единицу без пересчёта единиц и НДС. Цена 1С — учётная цена справочника.';
  if(comparison){
    detail.append(document.createTextNode(' · '),node('span',
      'inv-price-status '+(comparison.percent<50?'normal':comparison.percent<=100?'warning':'high'),comparison.label));
  }
  button.append(detail);
  button.addEventListener('click',()=>{
    const defaultQuantity=option.service?'1':String(safeQuantity(option));
    INV.picks.set(String(option.rowId),{name:option.name,unit:option.unit||(option.service?'усл':'шт'),
      quantity:defaultQuantity,service:!!option.service,catalog_id:option.catalog_id||option.id||null,
      code:option.code||null});
    markDirty();
    refresh();
  });
  return button;
}

function safeQuantity(item){
  const number=Number(String(item.sourceQuantity??'1').replace(/,/g,'.'));
  return Number.isFinite(number)&&number>0?number:1;
}

function invCandidates(item){
  const data=(INV.model&&INV.model.candidates&&INV.model.candidates[String(item.row_id)])||{};
  const list=[];
  for(const option of data.items||[])list.push({rowId:item.row_id,sourceQuantity:item.source_quantity,origin:'proposal',...option});
  for(const option of data.remembered||[])list.push({rowId:item.row_id,sourceQuantity:item.source_quantity,origin:'remembered',...option});
  return list;
}

function renderDocument(model){
  const host=$('invoice-doc');
  host.replaceChildren();
  const form=model.document.print_form||{};
  const issuer=model.document.issuer||{};
  const heading=node('div','invoice-doc-head');
  heading.append(node('strong','invoice-doc-title',`Счёт ${model.document.number||''}`),
                 node('span','invoice-doc-date',form.date||model.document.date||''));
  const numbers=node('div','invoice-doc-sums');
  const cell=(label,value)=>{const wrap=node('div','invoice-doc-cell');wrap.append(node('span','',label),node('strong','',value));return wrap;};
  numbers.append(cell('Итого',form.total||money(model.document.total)),
                 cell((form.vat_label||'НДС').replace(/:$/,''),form.vat||'—'),
                 cell('Прописью',form.amount_words||''));
  const supplier=node('p','invoice-doc-supplier',(issuer.name||'')+((issuer.tax_id)?' · БИН '+issuer.tax_id:''));
  host.append(heading,numbers,supplier);
}

function renderBlockers(model){
  const reasons=$('invoice-reasons');
  reasons.replaceChildren();
  const notes=[...model.blocked,...(model.rounding_notes||[])];
  for(const reason of notes)reasons.append(node('li','',reason));
  $('invoice-blockers').hidden=!notes.length;
}

function renderPdf(model){
  const link=$('invoice-pdf');
  const hint=$('invoice-pdf-hint');
  if(INV.dirty){
    link.hidden=true;link.removeAttribute('href');
    hint.textContent='Данные изменены: сохраните счёт или постройте предпросмотр.';
  }else if(model.can_render){
    link.hidden=false;
    link.href=INV.pdfUrl||`/api/jobs/${INV.jobId}/invoice.pdf?revision=${encodeURIComponent(model.revision)}&token=${encodeURIComponent(model.token)}`;
    link.download=`invoice-${model.document.number||'preview'}.pdf`;
    hint.textContent='Счёт доступен без подтверждения. Замечания показаны в документе.';
  }else{
    link.hidden=true;link.removeAttribute('href');
    hint.textContent='Скачивание PDF станет доступно после подтверждения всех строк.';
  }
}

function render(){
  const model=INV.model;
  if(!model)return;
  const rows=$('invoice-rows');
  rows.replaceChildren();
  model.generation.items.forEach((item,index)=>rows.append(renderRow(item,index)));
  renderDocument(model);
  renderBlockers(model);
  renderPdf(model);
  const pill=$('invoice-state');
  pill.className='pill '+(model.ready?'done':'processing');
  pill.textContent=model.ready?'Счёт сформирован':'Сформирован · есть замечания';
  $('invoice-empty').hidden=true;$('invoice-content').hidden=false;
}

function refreshTaskConfirm(){
  const button=$('invoice-confirm'),note=$('invoice-confirm-state');
  if(!button||!note)return;
  if(!TASK){button.hidden=true;note.hidden=true;$('ocr-confirm-success').hidden=true;return;}
  const task=TASK.state||{};
  const revision=INV.model?.revision||null;
  const applied=task.state==='applied';
  const confirmed=(task.state==='confirmed'&&task.confirmed_by_operator)||applied;
  const current=confirmed&&!task.stale_reason&&!!revision&&task.revision===revision;
  button.hidden=false;note.hidden=false;
  button.disabled=INV.busy||(!INV.dirty&&!INV.model?.exchange_ready)||current||applied;
  button.textContent=applied?'Счёт записан в 1С':current?'Подтверждено для 1С':(confirmed||task.stale_reason?'Подтвердить заново':'Подтвердить и передать в 1С');
  note.className='task-badge'+((current||applied)?' done':task.stale_reason?' wait':'');
  note.textContent=applied?`Применено в 1С, версия ${task.result_version}`:
    current?`Снимок для 1С: версия ${task.result_version}`:
    task.stale_reason?'Счёт изменён — для автоимпорта подтвердите заново':
    'Для автоимпорта подтвердите; вручную можно получить готовый счёт из 1С';
  $('ocr-confirm-success').hidden=!(current&&task.confirmed_by_operator&&!INV.dirty&&$('invoice-confirm-error').hidden);
}

async function confirmFor1C(){
  if(!TASK||!INV.model||INV.busy)return;
  const errorPanel=$('invoice-confirm-error');
  errorPanel.hidden=true;
  INV.busy=true;refresh();
  try{
    {
      const response=await fetch(`/api/jobs/${INV.jobId}/invoice/selections`,{
        method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(selectionPayload())});
      const model=await response.json();
      if(!response.ok)throw Error(model.error||'Не удалось сохранить выбор товаров');
      INV.model=model;INV.dirty=false;pickFromRows(model);
      await loadPrintPreview(selectionPayload());
    }
    await TASK.confirm(INV.model.revision);
    refresh();
    const panel=$('ocr-confirm-success');
    panel.hidden=false;
    requestAnimationFrame(()=>{
      panel.scrollIntoView({behavior:'smooth',block:'center'});
      try{panel.focus({preventScroll:true});}catch{panel.focus();}
    });
  }catch(error){
    $('ocr-confirm-success').hidden=true;
    const reasons=error.reasons?.length?' '+error.reasons.join(' '):'';
    errorPanel.textContent=(error.message||'Не удалось подтвердить результат')+reasons;
    errorPanel.hidden=false;
  }finally{INV.busy=false;refresh();}
}

function refresh(){
  $('invoice-save').disabled=INV.busy;$('invoice-preview').disabled=INV.busy;
  $('invoice-train').disabled=INV.busy||INV.dirty||!INV.model?.ready;
  refreshTaskConfirm();
  if(INV.model)render();
}

async function refreshModel(){
  if(INV.busy)return;
  INV.busy=true;
  try{
    const model=await fetchModel();
    if(model&&model.generation){
      INV.model=model;INV.dirty=false;pickFromRows(model);refresh();
      await loadPrintPreview(selectionPayload());
    }
  }finally{INV.busy=false;refresh();}
}

async function loadPrintPreview(payload){
  const jobId=INV.jobId;
  const response=await fetch(`/api/jobs/${jobId}/invoice/preview-pdf`,{
    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  if(!response.ok){toast('Не удалось построить печатную форму');return;}
  const blob=await response.blob();
  if(jobId!==INV.jobId)return;
  clearPdf();INV.pdfUrl=URL.createObjectURL(blob);
  const frame=$('invoice-print-preview');
  frame.src=INV.pdfUrl;
  frame.hidden=false;
  frame.setAttribute('tabindex','0');
  frame.scrollIntoView({block:'start',behavior:'instant'});
  frame.focus({preventScroll:true});
}

async function send(action){
  const payload=selectionPayload();
  if(action==='train')payload.confirm=true;
  INV.busy=true;refresh();
  try{
    const response=await fetch(`/api/jobs/${INV.jobId}/invoice/${action}`,{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const model=await response.json();
    if(!response.ok){toast(model.error||'Сервер не принял выбор');return;}
    if(action==='train'){
      toast(`Обучение сохранено: ${model.training.remembered} связей`);
      if(TASK)await TASK.refresh();
      return;
    }
    INV.model=model;
    INV.dirty=false;clearPdf();
    if(action==='selections')pickFromRows(model);
    await loadPrintPreview(payload);
    if(TASK)await TASK.refresh();
    refresh();
    toast(action==='selections'?'Связи сохранены':'Предпросмотр построен (изменения не сохранены)');
  }catch(error){toast('Счёт недоступен: '+error.message);}
  finally{INV.busy=false;refresh();}
}

function sync(){
  const receipt=result();
  const ok=!!(receipt&&receipt.rows&&receipt.rows.length&&receipt.job_id);
  const panel=$('invoice-panel');
  if(!ok){
    if(panel.hidden&&INV.jobId===null)return;
    panel.hidden=true;INV.analogs.clear();INV.openRows.clear();INV.jobId=null;INV.model=null;INV.picks=new Map();INV.revision=null;
    return;
  }
  panel.hidden=false;
  const revision=receipt.job_id+':'+receipt.revision;
  if(revision===INV.revision)return;
  INV.analogs.clear();INV.openRows.clear();INV.jobId=receipt.job_id;INV.revision=revision;
  clearPdf();INV.dirty=false;
  INV.model=null;INV.picks=new Map();
  $('invoice-content').hidden=true;$('invoice-empty').hidden=false;
  if(TASK)TASK.refresh();
  refresh();
  refreshModel();
}

function start(){
  $('invoice-edit-products').addEventListener('click',()=>{
    const block=$('invoice-rows-block');
    if(block&&!block.open)block.open=true;
    const form=document.querySelector('#invoice-content .invoice-rows');
    form.scrollIntoView({block:'start',behavior:'instant'});
    const firstRow=form.querySelector('.inv-row-details');
    if(firstRow&&!firstRow.open)firstRow.open=true;
    form.querySelector('.inv-search')?.focus({preventScroll:true});
  });
  const rowsBlock=$('invoice-rows-block');
  // Action buttons live inside the summary: keep them from collapsing the block.
  rowsBlock?.addEventListener('click',event=>{
    if(event.target.closest('button,a,input,select,textarea'))event.preventDefault();
  });
  $('invoice-save').addEventListener('click',()=>send('selections'));
  $('invoice-preview').addEventListener('click',()=>send('preview'));
  $('invoice-train').addEventListener('click',()=>send('train'));
  $('invoice-confirm')?.addEventListener('click',confirmFor1C);
  window.addEventListener('ocr-task-state',refreshTaskConfirm);
  setInterval(sync,1200);
}
start();
})();

// Layout only: the separator never changes receipt or invoice data.
(function invoiceDivider(){
  const workspace=document.querySelector('.workspace');
  const panel=document.getElementById('invoice-panel');
  const divider=document.getElementById('invoice-divider');
  const desktop=matchMedia('(min-width:981px)');
  let dragging=false;
  function update(){
    workspace.classList.toggle('invoice-split',!panel.hidden&&desktop.matches);
    const rect=workspace.getBoundingClientRect();
    const height=panel.getBoundingClientRect().top-rect.top;
    divider.setAttribute('aria-valuemin','180');
    divider.setAttribute('aria-valuemax',String(Math.max(180,Math.round(rect.height-180))));
    divider.setAttribute('aria-valuenow',String(Math.round(height)));
  }
  function setHeight(value){
    const max=Math.max(180,workspace.getBoundingClientRect().height-180);
    workspace.style.setProperty('--receipt-panel-height',Math.round(Math.max(180,Math.min(max,value)))+'px');
    update();
  }
  divider.addEventListener('pointerdown',event=>{
    if(event.button!==0||!desktop.matches)return;
    event.preventDefault();dragging=true;
    divider.setPointerCapture(event.pointerId);
    document.body.classList.add('invoice-resizing');
  });
  divider.addEventListener('pointermove',event=>{
    if(dragging)setHeight(event.clientY-workspace.getBoundingClientRect().top);
  });
  function end(){dragging=false;document.body.classList.remove('invoice-resizing');}
  divider.addEventListener('pointerup',end);
  divider.addEventListener('pointercancel',end);
  divider.addEventListener('lostpointercapture',end);
  divider.addEventListener('keydown',event=>{
    if(!['ArrowUp','ArrowDown','Home','End'].includes(event.key))return;
    event.preventDefault();
    const height=panel.getBoundingClientRect().top-workspace.getBoundingClientRect().top;
    setHeight(event.key==='Home'?180:event.key==='End'?Infinity:height+(event.key==='ArrowUp'?-20:20));
  });
  new MutationObserver(update).observe(panel,{attributes:true,attributeFilter:['hidden']});
  window.addEventListener('resize',()=>{
    if(workspace.style.getPropertyValue('--receipt-panel-height')){
      setHeight(parseFloat(workspace.style.getPropertyValue('--receipt-panel-height')));
    }else update();
  });
  desktop.addEventListener('change',update);
  update();
})();
