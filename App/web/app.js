'use strict';
const TASK=(location.pathname.match(/^\/tasks\/([a-f0-9]{32})\/ui\/?$/)||[])[1]||null;
const freshStart=!TASK&&new URLSearchParams(location.search).get('new')==='1';
if(freshStart){try{localStorage.removeItem('receipt-last-job');}catch{}history.replaceState(null,'','/');}
const $ = id => document.getElementById(id);
const SVG_NS = 'http://www.w3.org/2000/svg';
const state = {job:null, stream:null, assets:new Map(), rows:new Map(), result:null, token:0,
  maxUpload:20*1024*1024, activePhase:null, completed:new Set(), reviewOnly:false, selectedRow:null, blob:null,
  editingRow:null, selectedCatalog:null, searchToken:0, saving:false, passSimilarity:87, totalFocused:false,
  lastEventId:0};
const stepKeys = ['geometry','ocr','layout','matching'];
function productLabel(count){const n=count%100;return count+' '+(n>=11&&n<=14?'товаров':count%10===1?'товар':count%10>=2&&count%10<=4?'товара':'товаров');}
function node(tag, className='', text=''){const element=document.createElement(tag);element.className=className;if(text!==undefined)element.textContent=text;return element;}
function icon(id){const svg=document.createElementNS(SVG_NS,'svg');const use=document.createElementNS(SVG_NS,'use');use.setAttribute('href','#i-'+id);svg.append(use);return svg;}
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').hidden=true,5000);}
function setPill(text, kind){$('result-state').textContent=text;$('result-state').className='pill '+kind;}
function setProgress(value){const progress=Math.max(0,Math.min(100,value));$('progress-fill').style.width=progress+'%';$('progress-label').textContent=progress+'%';document.querySelector('[role=progressbar]').setAttribute('aria-valuenow',progress);}
function resetWebProgress(){
  state.lastEventId=0;
  $('progress-history').replaceChildren(node('li','empty','Этапы появятся после загрузки фотографии.'));
  $('progress-preview').removeAttribute('src');$('progress-preview-wrap').hidden=true;
  $('progress-preview-wrap').className='web-progress-preview';delete $('progress-preview-wrap').dataset.effect;
}
function renderWebProgress(event){
  const history=$('progress-history');history.querySelector('.empty')?.remove();
  const key=String(event.stage_id??event.id??event.stage);
  let item=history.querySelector(`li[data-stage-id="${CSS.escape(key)}"]`);
  if(!item){item=node('li');item.dataset.stageId=key;item.append(node('span','mark','…'),node('span','label',event.label||event.stage),node('small'));history.append(item);}
  item.className=event.status||'running';item.querySelector('.mark').textContent=event.status==='done'?'✓':event.status==='error'?'!':'…';
  item.querySelector('.label').textContent=event.detail||event.label||event.stage;
  item.querySelector('small').textContent=event.elapsed_ms==null?'':`${Number(event.elapsed_ms).toFixed(0)} мс`;
  item.scrollIntoView({block:'nearest'});
  const preview=$('progress-preview-wrap');
  if(event.status==='running'){preview.classList.add('running');preview.dataset.effect=event.effect||'';$('progress-preview-label').textContent=event.label||'';}
  else if(preview.dataset.effect===(event.effect||preview.dataset.effect)){preview.classList.remove('running');}
}
function updateProgressPreview(artifact){
  if(!artifact?.url)return;
  const wrap=$('progress-preview-wrap');$('progress-preview').src=artifact.url;wrap.hidden=false;
  $('progress-preview-label').textContent=artifact.title||'Промежуточное изображение';
}
function setPhase(phase, title){
  if(state.activePhase&&phase!==state.activePhase&&stepKeys.includes(state.activePhase))state.completed.add(state.activePhase);
  state.completed.delete(phase);state.activePhase=phase;
  // VISUAL_SYNC_V2: no artificial flying dots during non-OCR stages.
  $('process-title').textContent=title;
  const subtitles={input:'Сохраняем оригинал и учитываем ориентацию',geometry:'Контур → crop → выпрямленная таблица',ocr:'Распознавание выполняется на вашем компьютере',layout:'Визуализируем рамки и отделяем НДС',matching:'Названия сверяются с локальными справочниками',queued:'Можно загрузить следующее фото; оно встанет в очередь'};
  $('process-subtitle').textContent=subtitles[phase]||'Обработка фотографии';
  document.querySelectorAll('[data-step]').forEach(step=>{step.classList.toggle('active',step.dataset.step===phase);step.classList.toggle('complete',state.completed.has(step.dataset.step));});
  document.querySelectorAll('.image-card').forEach(card=>card.classList.remove('active-operation'));
  const target=phase==='geometry'?'original':phase==='input'?null:'rectified';
  const activeAsset=state.assets.get(target);if(activeAsset){activeAsset.card.classList.add('active-operation');drawAsset(activeAsset);}
}
function logEvent(text){
  if(!text||text===$('event-log').lastElementChild?.dataset.message)return;
  const item=node('li');item.dataset.message=text;
  item.append(node('time','log-time',new Date().toLocaleTimeString('ru-RU',{hour12:false})),node('span','',text));
  $('event-log').append(item);$('activity').hidden=false;
  $('event-count').textContent=$('event-log').children.length+' событий';
}
function polygon(svg, points, className, delay=0){
  const shape=document.createElementNS(SVG_NS,'polygon');shape.setAttribute('points',points.map(p=>p.join(',')).join(' '));shape.setAttribute('class',className);shape.style.animationDelay=delay+'s';svg.append(shape);return shape;
}
function needsMatchReview(row){
  return Boolean(row&&row.kind==='product'&&row.stage!=='ocr'&&!row.name_confirmed&&
    (row.similarity==null||row.similarity<(state.result?.pass_similarity??state.passSimilarity)));
}
function syncTotalHighlight(asset){
  let shape=asset.overlay.querySelector('.receipt-total-focus');
  if(!state.totalFocused||!['original','rectified'].includes(asset.key)){shape?.remove();return;}
  const color=getComputedStyle($('receipt-total')).color;
  if(!shape)shape=polygon(asset.overlay,[[0,0],[asset.width,0],[asset.width,asset.height],[0,asset.height]],'receipt-total-focus');
  shape.style.fill=color;
  shape.style.stroke=color;
}
function focusReceiptTotal(){
  focusRow(null,true);state.totalFocused=true;
  state.assets.forEach(syncTotalHighlight);
}
function drawAsset(asset){
  const svg=asset.overlay;svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${asset.width} ${asset.height}`);svg.setAttribute('preserveAspectRatio','xMidYMid meet');
  if(asset.contour)polygon(svg,asset.contour,'boundary');
  if(asset.region)polygon(svg,asset.region,'region');
  if(asset.cells?.length){
    asset.cells.filter(cell=>cell.kind!=='total').forEach((cell,index)=>{
      const shape=polygon(svg,cell.box,'cell-box'+(cell.kind==='tax'?' tax':'')+(cell.row===state.selectedRow?' focused':''),(index%30)*.025);
      shape.dataset.row=cell.row;
      if(asset.key==='rectified'&&cell.role==='name'&&needsMatchReview(state.rows.get(cell.row))){
        const underline=document.createElementNS(SVG_NS,'line');
        underline.setAttribute('x1',cell.box[3][0]+3);underline.setAttribute('y1',cell.box[3][1]-3);
        underline.setAttribute('x2',cell.box[2][0]-3);underline.setAttribute('y2',cell.box[2][1]-3);
        underline.setAttribute('class','match-warning');underline.dataset.row=cell.row;svg.append(underline);
      }
    });
  }else if(asset.boxes){asset.boxes.forEach((box,index)=>polygon(svg,box,'ocr-box',(index%30)*.025));}
  if(asset.key===state.visualScanTarget && !state.visualScanPassed){
    state.visualScanPassed=true;
    const scan=document.createElementNS(SVG_NS,'line');
    scan.setAttribute('x1','0');scan.setAttribute('x2',asset.width);
    scan.setAttribute('y1','0');scan.setAttribute('y2','0');
    scan.setAttribute('class','scan-line');svg.append(scan);
    if(!matchMedia('(prefers-reduced-motion: reduce)').matches){
      scan.animate([{transform:'translateY(0px)',opacity:.8},
        {transform:`translateY(${asset.height}px)`,opacity:0}],
        {duration:1400,easing:'ease-out',fill:'forwards'});
    }else scan.remove();
  }
  syncTotalHighlight(asset);
}
function openImage(key){const asset=state.assets.get(key);if(!asset)return;$('dialog-title').textContent=asset.title;$('dialog-image').replaceChildren(asset.stage.cloneNode(true));$('image-dialog').showModal();}
function followProcessing(){
  cancelAnimationFrame(followProcessing.frame);
  followProcessing.frame=requestAnimationFrame(()=>{
    if(state.result||!state.activePhase)return;
    const asset=state.assets.get('rectified')||state.assets.get('crop')||state.assets.get('original');
    const targets=[[$('input-scroll'),asset?.stage],[$('output-scroll'),$('receipt-body').lastElementChild]];
    for(const [panel,target] of targets){
      if(!target||!target.getClientRects().length)continue;
      const bounds=target.getBoundingClientRect(),viewport=panel.getBoundingClientRect();
      panel.scrollTo({top:panel.scrollTop+bounds.bottom-viewport.bottom+16,behavior:'instant'});
    }
  });
}
function animateRowTransfer(row){
  if(row.kind==='total'||matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  const token=state.token;
  // Measure after the processing scroll has positioned both panels.
  requestAnimationFrame(()=>requestAnimationFrame(()=>{
    if(token!==state.token)return; // VISUAL_SYNC_V2: animation may finish after OCR
    const asset=state.assets.get('rectified');
    const target=[...$('receipt-body').children].find(element=>Number(element.dataset.row)===row.row)?.children[1];
    if(!asset||!target)return;
    const source=[...asset.overlay.querySelectorAll('.cell-box')].find(shape=>Number(shape.dataset.row)===row.row);
    if(!source)return; // VISUAL_SYNC_V2: never invent a source row location
    const from=source.getBoundingClientRect(),to=target.getBoundingClientRect();
    const left=$('input-scroll').getBoundingClientRect(),right=$('output-scroll').getBoundingClientRect();
    if(!target.getClientRects().length||left.right>=right.right)return;
    const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
    const width=Math.min(260,left.width*.65,right.width*.65);
    const x=clamp(from.left+from.width/2-width/2,left.left+12,left.right-width-12);
    const y=clamp(from.top+from.height/2-24,left.top+12,left.bottom-60);
    const endX=clamp(to.left,to.left,right.right-width-12),endY=clamp(to.top,right.top+12,right.bottom-60);
    document.querySelectorAll('.receipt-transfer').forEach((item,index,items)=>{if(index<items.length-3)item.remove();});
    const card=node('div','receipt-transfer');card.setAttribute('aria-hidden','true');
    card.append(node('strong','',row.name??row.name_raw??'Строка чека'),node('span','',`${row.quantity??row.quantity_raw??'—'} × ${displayPrice(row.price??row.price_raw)}`));
    Object.assign(card.style,{left:x+'px',top:y+'px',width:width+'px'});document.body.append(card);
    const dx=endX-x,dy=endY-y;
    const animation=card.animate([
      {transform:'translate(0,0) scale(.94)',opacity:0},
      {transform:`translate(${dx*.15}px,${dy*.15-18}px) scale(1)`,opacity:1,offset:.2},
      {transform:`translate(${dx}px,${dy}px) scale(.96)`,opacity:1,offset:.85},
      {transform:`translate(${dx}px,${dy}px) scale(1)`,opacity:0}
    ],{duration:750,easing:'ease-in-out'});
    animation.finished.catch(()=>{}).finally(()=>card.remove());
  }));
}
function stopTransferFlow(){
  clearInterval(startTransferFlow.timer);startTransferFlow.timer=null;
  document.querySelectorAll('.receipt-flow-dot,.receipt-transfer').forEach(element=>element.remove());
}
function startTransferFlow(){
  if(startTransferFlow.timer||matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  const pulse=()=>{
    if(state.result)return;
    const asset=state.assets.get('rectified');if(!asset)return;
    const left=$('input-scroll').getBoundingClientRect(),right=$('output-scroll').getBoundingClientRect();
    if(left.right>=right.right)return;
    const source=asset.stage.getBoundingClientRect();
    const last=$('receipt-body').lastElementChild;
    const target=(last||$('receipt-area')).getBoundingClientRect();
    const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
    const x=clamp(source.right-24,left.left+20,left.right-20);
    const y=clamp(source.bottom-45,left.top+30,left.bottom-30);
    const endX=right.left+Math.min(85,right.width*.18);
    const endY=clamp(target.bottom-25,right.top+50,right.bottom-35);
    const dot=node('div','receipt-flow-dot');dot.setAttribute('aria-hidden','true');
    Object.assign(dot.style,{left:x+'px',top:y+'px'});document.body.append(dot);
    const animation=dot.animate([
      {transform:'translate(0,0) scale(.6)',opacity:0},
      {transform:`translate(${(endX-x)*.2}px,${(endY-y)*.2-20}px) scale(1)`,opacity:1,offset:.2},
      {transform:`translate(${endX-x}px,${endY-y}px) scale(.5)`,opacity:0}
    ],{duration:1100,easing:'ease-in-out'});
    animation.finished.catch(()=>{}).finally(()=>dot.remove());
  };
  pulse();startTransferFlow.timer=setInterval(pulse,420);
}
function addArtifact(data){
  let asset=state.assets.get(data.key);
  if(asset){
    const dom={card:asset.card,heading:asset.heading,subtitle:asset.subtitle,stage:asset.stage,
               img:asset.img,overlay:asset.overlay,dimensions:asset.dimensions};
    Object.assign(asset,data,dom);asset.img.src=data.url;asset.heading.textContent=data.title;
    asset.subtitle.textContent=data.subtitle||'';asset.dimensions.textContent=`${data.width} × ${data.height} px`;drawAsset(asset);return;
  }
  const card=node('article','image-card'+(['header','crop'].includes(data.key)?' debug-only':''));card.dataset.key=data.key;
  const header=node('div','image-card-header');const headings=node('div');const heading=node('h3','',data.title);const subtitle=node('p','',data.subtitle||'');headings.append(heading,subtitle);
  header.append(headings,node('span','operation-tag',data.role==='original'?'SOURCE':data.key.toUpperCase()));
  const stage=node('div','image-stage');const img=document.createElement('img');img.addEventListener('load',followProcessing);img.src=data.url;img.alt=data.title;img.loading='eager';
  const overlay=document.createElementNS(SVG_NS,'svg');overlay.setAttribute('aria-hidden','true');stage.append(img,overlay);
  stage.addEventListener('click',()=>openImage(data.key));
  const tools=node('div','image-tools');const dimensions=node('span','',`${data.width} × ${data.height} px`);const expand=node('button','','Рассмотреть');expand.type='button';expand.append(icon('expand'));expand.addEventListener('click',()=>openImage(data.key));tools.append(dimensions,expand);
  card.append(header,stage,tools);$('image-list').append(card);
  const order=['original','header','crop','rectified'];
  [...$('image-list').children].sort((a,b)=>order.indexOf(a.dataset.key)-order.indexOf(b.dataset.key)).forEach(item=>$('image-list').append(item));
  asset={...data,card,heading,subtitle,stage,img,overlay,dimensions};state.assets.set(data.key,asset);drawAsset(asset);window.ReceiptUploadGate?.asset(data.key);
  followProcessing();
}
function focusRow(rowId, persist=false){
  state.totalFocused=false;state.assets.forEach(syncTotalHighlight);
  state.selectedRow=rowId;
  const updateFocus=()=>{const asset=state.assets.get('rectified');if(asset){asset.overlay.classList.add('static-focus');asset.overlay.querySelectorAll('.cell-box').forEach(shape=>shape.classList.toggle('focused',state.selectedRow!==null&&Number(shape.dataset.row)===state.selectedRow));}};
  updateFocus();
  $('receipt-body').querySelectorAll('tr').forEach(row=>row.classList.toggle('selected',Number(row.dataset.row)===rowId));
  clearTimeout(focusRow.timer);
  if(!persist)focusRow.timer=setTimeout(()=>{if(state.selectedRow===rowId){state.selectedRow=null;updateFocus();}},900);
}
function formatPercent(value){return value===null||value===undefined?'—':Number(value).toFixed(1)+'%';}
function formatMoney(value){if(value===null||value===undefined||value==='')return '—';const [whole,decimal='00']=String(value).split('.');return whole.replace(/\B(?=(\d{3})+(?!\d))/g,' ') + ',' + decimal.padEnd(2,'0')+' ₸';}
function displayPrice(value){return String(value??'').replace(/\s*(?:KZT|ТГ|ТЕНГЕ|[TТ₸])\s*$/iu,'').trim()||'—';}
function isReceiptNumber(value){
  if(value==null||typeof value==='boolean'||typeof value==='object')return false;
  let text=String(value).replace(/\s+/g,'').replace(/(?:KZT|ТГ|ТЕНГЕ|[TТ₸])$/iu,'');
  if(text.length>40||!/^\d+(?:[.,]\d+)*$/.test(text))return false;
  if(text.includes('.')&&text.includes(','))text=text.replace(text.lastIndexOf('.')>text.lastIndexOf(',')?/,/g:/\./g,'');
  text=text.replace(/,/g,'.');
  return (text.match(/\./g)||[]).length<=1&&Number.isFinite(Number(text))&&Number(text)>=0&&Number(text)<1e15;
}
function numericReceiptCell(row,field){
  const value=row[field]??row[field+'_raw'];
  const invalid=Array.isArray(row.invalid_fields)?row.invalid_fields.includes(field):!isReceiptNumber(value);
  const shown=field==='price'?displayPrice(value):(value==null||String(value).trim()===''?'—':String(value));
  const cell=node('td',invalid?'invalid-number':'',shown);cell.dataset.field=field;
  if(invalid){cell.setAttribute('aria-invalid','true');cell.title=field==='quantity'?'Количество не распознано числом. Нажмите для исправления.':'Цена не распознана числом. Нажмите для исправления.';}
  return cell;
}
function makeReceiptRow(row,ordinal){
  const draft=row.stage==='ocr';
  const lowMatch=needsMatchReview(row);
  const tr=node('tr',(lowMatch?'needs-check ':row.kind!=='product'?'special-row ':'')+(draft?'ocr-draft':''));tr.dataset.row=row.row;
  if(row.kind==='tax'&&row.vat_status)tr.classList.add(row.vat_status==='match'?'vat-passed':'vat-failed');
  tr.addEventListener('mouseenter',()=>focusRow(row.row,true));tr.addEventListener('click',()=>{focusRow(row.row,true);if(state.result)editRow(row.row);state.assets.get('rectified')?.card.scrollIntoView({behavior:'smooth',block:'nearest'});});
  const rawPrintedNumber=String(row.item_number_raw??'').trim();
  const recoveredNumber=String(row.item_number_recovered??'').trim();
  const printedNumber=/^\d+$/.test(rawPrintedNumber)?rawPrintedNumber:recoveredNumber;
  const hasPrintedNumber=/^\d+$/.test(printedNumber)&&Number(printedNumber)>0;
  const numberCell=node('td',hasPrintedNumber?'':'inferred-number',hasPrintedNumber?printedNumber:String(ordinal));
  if(recoveredNumber&&printedNumber===recoveredNumber)numberCell.title='Номер повторно прочитан только из ячейки №.';
  else if(!hasPrintedNumber)numberCell.title='Номер не распознан: указан порядковый номер строки таблицы.';
  tr.append(numberCell);
  const description=node('td');const kindNames={tax:'НДС',service:row.name_raw||'Доставка',total:'Итог',unreadable:'Не удалось прочитать название'};
  const name=row.name??row.name_raw;
  description.append(node('div','item-name',row.kind==='tax'&& !/[нН][дД][сС]/.test(name)?'НДС':name||kindNames[row.kind]||'—'));
  if(row.kind==='product'){
    const debug=node('div','receipt-debug debug-only');
    debug.append(node('span','item-reference','OCR: '+row.name_raw));
    if(row.match?.best)debug.append(node('span','item-reference','Справочник: '+row.match.best.name));
    if(row.article)debug.append(node('span','item-reference','Артикул '+row.article));
    if(row.quantity_source==='cell_ocr')debug.append(node('span','item-reference',`Количество из отдельной ячейки: ${row.quantity} · OCR ${formatPercent((row.quantity_ocr?.views?.[0]?.confidence??0)*100)}`));
    const metrics=node('div','item-metrics');metrics.append(node('span','metric','Уверенность OCR '+formatPercent(row.ocr_name_confidence_pct)));
    if(row.match?.source==='manual_mapping'){
      metrics.append(node('span','metric pass','База ручных сопоставлений'));
      metrics.append(node('span','metric','Автоматическое сходство '+formatPercent(row.match.automatic_match?.best?.score)));
    }else if(row.similarity!=null)metrics.append(node('span','metric '+(needsMatchReview(row)?'low':'pass'),'Сходство '+formatPercent(row.similarity)));
    else if(draft)metrics.append(node('span','metric','Распознавание…'));
    debug.append(metrics);
    if(row.review_reason)debug.append(node('div','item-reason',row.review_reason));
    description.append(debug);
  }
  tr.append(description,numericReceiptCell(row,'quantity'),numericReceiptCell(row,'price'));
  return tr;
}
function renderRows(){
  const rows=[...state.rows.values()].filter(row=>row.kind!=='total').sort((a,b)=>a.row-b.row);const products=rows.filter(row=>row.kind==='product');const review=rows.filter(row=>row.needs_review);
  $('product-count').textContent=productLabel(products.length);$('review-count').textContent=review.length;
  const selected=state.reviewOnly?rows.filter(row=>row.needs_review):rows;
  const ordinals=new Map(rows.map((row,index)=>[row.row,index+1]));
  $('receipt-body').replaceChildren(...selected.map(row=>makeReceiptRow(row,ordinals.get(row.row))));$('receipt-empty-filter').hidden=selected.length!==0;
  $('receipt-area').hidden=false;$('review-note').hidden=!review.length;
  const asset=state.assets.get('rectified');if(asset)drawAsset(asset);
  renderReceiptErrors(state.result||{rows:[...state.rows.values()]});
}
function showMathSuccess(){
  document.querySelector('.math-success-flash')?.remove();
  const panel=$('output-scroll').getBoundingClientRect();
  const top=Math.max(0,panel.top),bottom=Math.min(innerHeight,panel.bottom);
  if(bottom<=top)return;
  const badge=node('div','math-success-flash');badge.setAttribute('role','status');badge.setAttribute('aria-label','Сумма и НДС проверены: всё сходится');
  const svg=document.createElementNS(SVG_NS,'svg');svg.setAttribute('viewBox','0 0 100 100');svg.setAttribute('aria-hidden','true');
  const check=document.createElementNS(SVG_NS,'path');check.setAttribute('d','M22 51 L42 71 L79 30');svg.append(check);badge.append(svg);
  Object.assign(badge.style,{left:(panel.left+panel.width/2)+'px',top:((top+bottom)/2)+'px'});document.body.append(badge);
  clearTimeout(showMathSuccess.timer);showMathSuccess.timer=setTimeout(()=>badge.remove(),1300);
}
function renderReceiptErrors(result){
  const list=$('receipt-error-reasons');list.replaceChildren();
  const calc=result?.calculation, messages=[];
  const ordinals=new Map((result?.rows||[]).filter(row=>row.kind!=='total').map((row,index)=>[row.row,index+1]));
  for(const row of result?.rows||[]){
    if(row.kind==='total')continue;
    const fields=Array.isArray(row.invalid_fields)?row.invalid_fields:['quantity','price'].filter(field=>!isReceiptNumber(row[field]??row[field+'_raw']));
    if(!fields.length)continue;
    const number=/^\d+$/.test(String(row.item_number_raw||''))?row.item_number_raw:ordinals.get(row.row);
    const values=fields.map(field=>`${field==='quantity'?'количество':'цена'} «${String(row[field]??row[field+'_raw']??'').trim()||'не прочитано'}»`).join(', ');
    messages.push(`Строка ${number} «${row.name??row.name_raw??'Без названия'}»: ${values} — требуется числовое значение.`);
  }
  if(calc){
    if(calc.status==='incomplete')messages.push('Итог отмечен красным: проверка суммы не завершена, потому что не все количества или цены прочитаны. Даже если видимая сумма совпадает, чек пока не проверен.');
    else if(calc.status==='missing_original_total')messages.push('Итог отмечен красным: итог с фотографии не прочитан. Укажите его в редакторе чека.');
    else if(calc.status==='mismatch')messages.push(`Итог не совпадает: по строкам ${formatMoney(calc.computed_total)}, на фотографии ${formatMoney(calc.expected_total)}, разница (по строкам минус фото) ${formatMoney(calc.difference)}.`);
    const vat=calc.vat;
    if(vat?.status==='incomplete')messages.push(`Строка НДС отмечена красным: ${vat.reason||'не все количества и цены прочитаны, поэтому НДС пока нельзя проверить'}. Это не подтверждённая ошибка суммы НДС.`);
    else if(vat?.status==='mismatch')messages.push(`НДС не совпадает: в чеке ${formatMoney(vat.actual_tax)}, ожидается ${formatMoney(vat.expected_tax)} при ставке ${vat.rate_percent}%. База расчёта ${formatMoney(vat.taxable_base)}, разница ${formatMoney(vat.difference)}.`);
    else if(vat?.status==='invalid_rate')messages.push(`НДС отмечен красным: ${vat.reason||'ставка в настройках недействительна'}. Проверьте VAT_RATE_PERCENT в .env.`);
  }
  messages.forEach(message=>list.append(node('li','',message)));
  $('receipt-errors').hidden=messages.length===0;
}
function renderMath(){
  renderReceiptErrors(state.result);
  const result=state.result, calc=result?.calculation;if(!calc){$('math-panel').hidden=true;return;}
  const status=calc.validation_status??calc.status;
  if(status!=='match'){clearTimeout(showMathSuccess.timer);document.querySelector('.math-success-flash')?.remove();}
  if(status==='match'&&renderMath.previousStatus!=='match')showMathSuccess();
  renderMath.previousStatus=status;
  $('math-panel').hidden=true;$('math-panel').dataset.status=status;
  document.querySelector('.receipt-total').classList.toggle('math-passed',status==='match');
  document.querySelector('.receipt-total').classList.toggle('math-failed',status!=='match');
  const reasons={match:'Сумма и НДС проверены',mismatch:'Сумма не совпадает с фотографией',incomplete:'Не все количества или цены прочитаны',missing_original_total:'Итог с фотографии не прочитан',vat_mismatch:calc.vat?.reason||'НДС не прошёл проверку'};
  document.querySelector('.receipt-total').title=reasons[status]||'Требуется проверка';
  if(state.totalFocused)state.assets.forEach(syncTotalHighlight);
  const labels={match:'Суммы совпадают',mismatch:'Суммы не совпадают',incomplete:'Не все числа прочитаны',missing_original_total:'Укажите итог с фото',vat_mismatch:'Проверьте НДС'};
  $('math-status').textContent=labels[status]||'Проверка';$('math-status').className='pill '+(status==='match'?'done':'error');
  $('calculated-total').textContent=formatMoney(calc.computed_total);$('expected-total-label').textContent=formatMoney(calc.expected_total);$('total-difference').textContent=formatMoney(calc.difference);
  $('math-breakdown').textContent=`Товары: ${formatMoney(calc.products_subtotal)} · Доставка: ${formatMoney(calc.services_subtotal)} · НДС: ${formatMoney(calc.tax_subtotal)}`;
  $('math-issues').replaceChildren();
  (calc.missing_rows||[]).forEach(issue=>{const button=node('button','math-issue',`${issue.name||'Строка '+issue.row}: проверьте ${issue.fields.map(f=>f==='quantity'?'количество':'цену').join(' и ')}`);button.type='button';button.addEventListener('click',()=>editRow(issue.row));$('math-issues').append(button);});
  if(calc.status==='mismatch')$('math-issues').append(node('p','math-explanation','Сверьте количество и цену строк с фотографией. Нажмите на строку чека для редактирования.'));
  $('finalize-receipt').disabled=!result.can_finalize||result.finalized||state.saving;
  $('finalize-receipt').textContent=result.finalized?'Чек проверен':'Завершить проверку';
  $('finalize-hint').textContent=result.finalized?'Названия утверждены, сумма совпадает.':result.review_count?`Осталось проверить строк: ${result.review_count}`:calc.status==='match'?'Можно завершить проверку.':'Исправьте числа или итог с фотографии.';
}
function showCandidates(items){
  $('catalog-options').replaceChildren();
  if(!items.length){$('catalog-options').append(node('p','editor-evidence','Точный товар не найден. Уточните запрос или отредактируйте и утвердите название вручную.'));return;}
  items.forEach(item=>{const button=node('button','catalog-option');button.type='button';button.append(node('strong','',item.name),node('span','',item.articles.length?'Артикул '+item.articles.join(' / '):'Артикул не указан'));button.addEventListener('click',()=>selectCandidate(item));$('catalog-options').append(button);});
}
function selectCandidate(item, explicit=true){
  state.catalogChosen=explicit;
  state.selectedCatalog=item;$('edit-name').value=item.name;$('article-field').hidden=item.articles.length<=1;$('edit-article').replaceChildren();
  if(item.articles.length>1)$('edit-article').append(new Option('Выберите артикул',''));
  item.articles.forEach(article=>$('edit-article').append(new Option(article,article)));
  const row=state.rows.get(state.editingRow);if(row?.article&&item.articles.includes(row.article))$('edit-article').value=row.article;
  $('editor-hint').textContent='Выбран товар из справочника';
}
function editRow(rowId, scroll=true, highlight=true){
  if(!state.result)return;const row=state.rows.get(Number(rowId));if(!row)return;
  state.editingRow=row.row;state.selectedCatalog=null;state.catalogChosen=false;state.searchToken++;
  $('editor-panel').hidden=false;$('editor-panel').open=true;$('edit-row').value=String(row.row);
  $('edit-name').value=row.name??row.name_raw;$('edit-quantity').value=row.quantity??row.quantity_raw;$('edit-price').value=row.price??row.price_raw;
  $('edit-ocr-evidence').textContent='Исходный OCR: '+row.name_raw;
  $('catalog-search').value='';$('catalog-picker').hidden=row.kind!=='product';$('article-field').hidden=true;$('editor-error').hidden=true;
  $('editor-hint').textContent=row.needs_review?'Требует проверки':'Можно исправить данные';
  const options=(row.match?.top_matches||[]).map(item=>({key:item.normalized,name:item.name,articles:[...new Set(item.sources.map(s=>s.article).filter(Boolean))]}));showCandidates(options);
  $('accept-match').hidden=!row.match?.best||row.kind!=='product';
  if(row.catalog_key){const current=options.find(item=>item.key===row.catalog_key)||{key:row.catalog_key,name:row.name,articles:row.articles||[]};selectCandidate(current,false);}
  if(scroll)$('editor-panel').scrollIntoView({behavior:'smooth',block:'nearest'});
  if(highlight)focusRow(row.row,true);
}
function setupEditor(){
  $('editor-panel').hidden=false;const selected=state.editingRow;
  const editable=[...state.rows.values()].filter(row=>row.kind!=='total').sort((a,b)=>a.row-b.row);
  $('edit-row').replaceChildren(...editable.map((row,index)=>new Option(`${index+1}. ${(row.name??row.name_raw)||'Не прочитано'}`,String(row.row))));
  $('edit-expected-total').value=state.result.calculation.expected_total??'';
  const row=editable.find(row=>row.row===selected)||editable.find(row=>row.needs_review)||editable[0];
  const wasOpen=$('editor-panel').open;
  $('editor-row-fields').hidden=!row;if(row)editRow(row.row,false,false);
  $('editor-panel').open=wasOpen||Boolean(row?.needs_review);
}
async function sendEdit(payload){
  if(!state.result||state.saving)return;
  state.saving=true;$('save-row').disabled=true;$('save-total').disabled=true;$('editor-error').hidden=true;renderMath();
  try{
    const response=await fetch(`/api/jobs/${state.job}/edits`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision:state.result.revision,...payload})});
    const data=await response.json();if(!response.ok)throw Error(data.error||'Не удалось сохранить изменения');
    finish(data.result);toast(data.mapping_saved?'Выбор сохранён в базе ручных сопоставлений':data.result.finalized?'Чек проверен. Суммы совпадают.':'Изменения сохранены, сумма пересчитана');
  }catch(error){$('editor-error').textContent=error.message;$('editor-error').hidden=false;toast(error.message);}
  finally{state.saving=false;$('save-row').disabled=false;$('save-total').disabled=false;renderMath();}
}
async function saveEditedRow(){
  const row=state.rows.get(state.editingRow);if(!row)return;
  const payload={row:row.row,name:$('edit-name').value,confirm_name:true};
  if(state.selectedCatalog&&$('edit-name').value===state.selectedCatalog.name){payload.catalog_key=state.selectedCatalog.key;payload.learn_mapping=Boolean(state.catalogChosen);payload.article=state.selectedCatalog.articles.length>1?$('edit-article').value:state.selectedCatalog.articles[0]||'';}
  for(const [field,id] of [['quantity','edit-quantity'],['price','edit-price']])if($(id).value!==(row[field]??row[field+'_raw']))payload[field]=$(id).value;
  await sendEdit(payload);
}
function reset(){
  window.ReceiptUploadGate?.reset();
  VisualSync.reset();
  window.OcrActivity?.finish(false);
  renderMath.previousStatus=null;clearTimeout(showMathSuccess.timer);document.querySelector('.math-success-flash')?.remove();
  stopTransferFlow();
  document.querySelectorAll('.receipt-transfer').forEach(card=>card.remove());
  state.token++;state.stream?.close();state.stream=null;state.job=null;state.assets.clear();state.rows.clear();state.result=null;state.completed.clear();state.activePhase=null;state.selectedRow=null;state.reviewOnly=false;resetWebProgress();
  if(state.blob){URL.revokeObjectURL(state.blob);state.blob=null;}
  $('receipt-math-alert').hidden=true;
  state.totalFocused=false;state.editingRow=null;state.selectedCatalog=null;state.searchToken++;$('editor-panel').hidden=true;$('editor-panel').open=false;$('math-panel').hidden=true;$('receipt-errors').hidden=true;$('receipt-error-reasons').replaceChildren();
  document.querySelector('.receipt-total').classList.remove('math-passed','math-failed');document.querySelector('.receipt-total').removeAttribute('title');
  $('image-list').replaceChildren();$('receipt-body').replaceChildren();$('event-log').replaceChildren();$('receipt-area').hidden=true;$('empty-result').hidden=true;$('error-card').hidden=true;$('geometry-note').hidden=true;$('exports').hidden=true;$('activity').hidden=true;$('raw-text').textContent='';$('order-number').textContent='';$('receipt-date').textContent='Данные с фотографии';$('receipt-total').textContent='—';$('receipt-meta').textContent='Распознавание выполняется…';$('review-toggle').setAttribute('aria-pressed','false');$('review-note').hidden=true;$('process-card').classList.add('busy');
  document.querySelectorAll('[data-step]').forEach(step=>step.classList.remove('active','complete'));setProgress(0);setPill('Обработка','processing');
}
function showError(message){
  window.ReceiptUploadGate?.failed(message);
  VisualSync.reset();
  window.OcrActivity?.finish(false);
  stopTransferFlow();
  state.stream?.close();$('process-card').classList.remove('busy');$('error-card').hidden=false;$('error-text').textContent=message;$('process-title').textContent='Обработка остановлена';$('process-subtitle').textContent='Исходное фото сохранено';setPill('Нужна проверка','error');logEvent(message);
  document.querySelectorAll('.image-card').forEach(card=>card.classList.remove('active-operation'));
}
function showReceiptMathWarning(result,previous){
  const calc=result?.calculation, status=calc?.validation_status??calc?.status;
  if(!status)return;
  const warning=$('receipt-math-alert');
  if(status==='match'){warning.hidden=true;return;}
  const previousStatus=previous?.calculation?.validation_status??previous?.calculation?.status;
  if(!previous||previous.job_id!==result.job_id||previousStatus==='match'){
    warning.hidden=false;
    requestAnimationFrame(()=>{
      const details=$('receipt-errors');
      if(!details.hidden){details.scrollIntoView({behavior:'smooth',block:'center'});details.focus({preventScroll:true});}
    });
  }
}
function finish(result){
  if(!result)return;
  window.ReceiptUploadGate?.finished(result);
  VisualSync.finished();
  window.OcrActivity?.finish(true);
  stopTransferFlow();
  const previous=state.result;
  if(!previous)focusRow(null,true);
  state.result=result;window.dispatchEvent(new Event('receipt-result-updated'));state.stream?.close();state.stream=null;state.rows=new Map(result.rows.map(row=>[row.row,row]));renderRows();
  $('empty-result').hidden=true;$('error-card').hidden=true;$('process-card').classList.remove('busy');setProgress(100);
  $('process-title').textContent=result.finalized?'Чек проверен и готов':'Распознавание завершено';$('process-subtitle').textContent=`${productLabel(result.product_count)} · ${(result.elapsed_ms/1000).toFixed(1)} с · ${result.catalog_names.toLocaleString('ru-RU')} названий в справочнике`;
  setPill(result.finalized?'Чек проверен':result.can_finalize?'Можно утвердить':'Нужна проверка',result.finalized?'done':result.can_finalize?'done':'processing');
  document.querySelectorAll('[data-step]').forEach(step=>{step.classList.remove('active');step.classList.add('complete');});
  document.querySelectorAll('.image-card').forEach(card=>card.classList.remove('active-operation'));
  $('order-number').textContent=result.order_number?'№ '+result.order_number:'';$('receipt-date').textContent=result.date||result.filename;
  $('receipt-total').textContent=result.calculation?formatMoney(result.calculation.computed_total):result.total_raw||'Не прочитан';$('receipt-meta').textContent=result.finalized?'Проверено пользователем · итог совпадает с фотографией':'Черновик · исходный OCR сохранён отдельно';
  renderMath();setupEditor();showReceiptMathWarning(result,previous);
  $('download-json').href=result.downloads.json;$('download-csv').href=result.downloads.csv;$('exports').hidden=false;$('raw-text').textContent=result.raw_text;
  if(result.warnings?.length){$('geometry-note').hidden=false;$('geometry-note').textContent=result.warnings.some(w=>w.includes('не удалось'))?result.warnings.join(' · '):'Граница таблицы содержит неоднозначные участки. Проверьте результат выпрямления по исходному фото.';}
}
function consume(event){
  if(event.id&&event.id<=state.lastEventId)return;
  if(event.id)state.lastEventId=event.id;
  window.ReceiptUploadGate?.onPipeline(event);
  if(event.progress!==undefined)setProgress(event.progress);
  if(event.type==='stage'){renderWebProgress(event);}
  else if(event.type==='phase'){setPhase(event.phase==='layout'&&event.progress<50?'geometry':event.phase,event.title);window.OcrActivity?.phase(event.phase,event.title);logEvent(event.title);}
  else if(event.type==='artifact'){addArtifact(event.artifact);updateProgressPreview(event.artifact);logEvent(event.artifact.title);}
  else if(event.type==='contour'){const asset=state.assets.get(event.target);if(asset){asset.contour=event.polygon;drawAsset(asset);}logEvent(event.title);}
  else if(event.type==='region'){const asset=state.assets.get(event.target);if(asset){asset.region=event.box;drawAsset(asset);}}
  else if(event.type==='boxes'){const asset=state.assets.get(event.target);if(asset){asset.boxes=event.boxes;drawAsset(asset); /* VISUAL_SYNC_V2: keep row cells during OCR */}logEvent(`${event.title}: ${event.boxes.length}`);}
  else if(event.type==='cells'){const asset=state.assets.get(event.target);if(asset){asset.cells=event.cells;drawAsset(asset);}logEvent(`${event.title} · ${event.cells.length} ячеек`);}
  else if(event.type==='header'){$('receipt-area').hidden=false;$('order-number').textContent=event.order_number?'№ '+event.order_number:'';$('receipt-date').textContent=event.date||'Дата не прочитана';}
  else if(event.type==='draft_rows'){window.OcrActivity?.rows(event.rows);if(event.pass_similarity!==undefined)state.passSimilarity=event.pass_similarity;state.rows=new Map(event.rows.map(row=>[row.row,row]));renderRows();$('receipt-meta').textContent='Черновик из OCR · названия ещё не сверены';}
  else if(event.type==='row'){state.rows.set(event.row.row,event.row);renderRows();focusRow(event.row.row);logEvent(event.title);}
  else if(event.type==='visual'){VisualSync.handle(event);}
  else if(event.type==='message'){window.OcrActivity?.message(event.title);$('process-subtitle').textContent=event.title;logEvent(event.title);}
  else if(event.type==='ui_status'){window.OcrActivity?.uiStatus(event);$('process-subtitle').textContent=event.title;if(event.rows_total||event.matched_products!==undefined)logEvent(window.OcrActivity ? document.getElementById('ocr-activity-details').textContent : event.title);}
  else if(event.type==='complete'){finish(event.result);$('progress-preview-wrap').classList.remove('running');logEvent('Обработка завершена');}
  else if(event.type==='error')showError(event.title);
  if(event.type!=='complete'&&event.type!=='error')followProcessing();
  // VISUAL_SYNC_V2: transfers are triggered by raw OCR row_ready events, not catalog matching.
}
function listen(jobId, token){
  state.stream=new EventSource(`/api/jobs/${jobId}/events`);
  state.stream.onopen=()=>{if(token===state.token)window.OcrActivity?.heartbeat();};
  state.stream.addEventListener('heartbeat',()=>{if(token===state.token)window.OcrActivity?.heartbeat();});
  state.stream.addEventListener('pipeline',event=>{if(token!==state.token)return;window.OcrActivity?.heartbeat();try{consume(JSON.parse(event.data));}catch(error){console.error(error);toast('Не удалось отобразить обновление. Обновите страницу.');}});
  state.stream.onerror=async()=>{
    if(token!==state.token)return;
    window.OcrActivity?.connectionLost();
    try{const response=await fetch(`/api/jobs/${jobId}`);if(!response.ok)throw Error();const job=await response.json();for(const event of job.events||[])consume(event);if(job.status==='done')finish(job.result);else if(job.status==='failed')showError(job.error);else $('process-subtitle').textContent='Восстанавливаем связь с обработкой…';}catch{ $('process-subtitle').textContent='Нет связи с сервером. Ожидаем подключения…'; }
  };
}
async function upload(file){
  if(!file)return;
  if(file.size>state.maxUpload){toast(`Максимальный размер фото — ${Math.floor(state.maxUpload/1024/1024)} МБ`);return;}
  if(file.type&&!['image/jpeg','image/png','image/webp'].includes(file.type)){toast('Выберите фотографию JPG, PNG или WEBP');return;}
  reset();window.OcrActivity?.start();const token=state.token;window.ReceiptUploadGate?.begin(token);$('drop-zone').hidden=true;$('input-note').hidden=false;state.blob=URL.createObjectURL(file);
  logEvent('Начало обработки: '+file.name);
  const preview=new Image();preview.src=state.blob;preview.onload=()=>{if(token===state.token&&!state.assets.has('original'))addArtifact({key:'original',role:'original',title:'Исходное фото',subtitle:file.name,url:preview.src,width:preview.naturalWidth,height:preview.naturalHeight});};
  $('process-title').textContent='Загружаем фотографию';$('process-subtitle').textContent=file.name;
  try{
    const response=await (TASK?postTaskPhoto(file):fetch('/api/jobs?filename='+encodeURIComponent(file.name||'clipboard.png'),{method:'POST',headers:{'Content-Type':file.type||'application/octet-stream'},body:file}));
    const job=await response.json();if(!response.ok)throw Error(job.error||'Не удалось загрузить фотографию');if(token!==state.token)return;
    state.job=job.job_id||job.id;if(TASK)updateTaskBadge(job);window.ReceiptUploadGate?.accepted(token);try{localStorage.setItem('receipt-last-job',state.job);}catch{}listen(state.job,token);
  }catch(error){if(token===state.token)showError(error.message||'Не удалось связаться с сервером');}
}
function postTaskPhoto(file){
  const form=new FormData();form.append('file',file,file.name||'clipboard.png');
  return fetch(`/tasks/${TASK}/photos`,{method:'POST',body:form});
}
const TASK_LABELS={waiting_photo:'ожидает фото',review:'проверка чека',confirmed:'готово для 1С',applied:'применено в 1С'};
let TASK_STATE=null;
function updateTaskBadge(task){
  const badge=$('task-badge');
  if(!TASK){return;}
  TASK_STATE=task||null;
  window.dispatchEvent(new Event('ocr-task-state'));
  if(!badge)return;
  badge.hidden=false;
  badge.className='task-badge'+(task?.state==='confirmed'||task?.state==='applied'?' done':task?.stale_reason?' wait':'');
  badge.textContent='Задание 1С: '+(TASK_LABELS[task?.state]||'недоступно')+(task?.stale_reason?' · нужно подтвердить заново':'');
  if(task?.job_id){$('choose-top').hidden=true;$('paste-top').hidden=true;}
}
window.OCRTask=TASK?{
  id:TASK,
  get state(){return TASK_STATE;},
  async refresh(){
    try{
      const response=await fetch(`/tasks/${TASK}/ui/state`);
      if(!response.ok)return null;
      const task=await response.json();updateTaskBadge(task);return task;
    }catch{return null;}
  },
  async confirm(revision){
    const response=await fetch(`/tasks/${TASK}/confirm`,{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision})});
    const data=await response.json().catch(()=>({}));
    if(!response.ok){
      const error=Error(data.error||'Не удалось подтвердить результат для 1С');
      error.reasons=data.reasons||[];throw error;
    }
    if(response.status!==200||data.state!=='confirmed'||!data.confirmed_by_operator){
      throw Error('Подтверждение ещё не сохранено сервером. Повторите попытку.');
    }
    updateTaskBadge(data);return data;
  }
}:null;
async function restoreTask(){
  const token=state.token;
  try{
    const response=await fetch(`/tasks/${TASK}/ui/state`);
    if(!response.ok){updateTaskBadge(null);showError('Задание 1С не найдено. Откройте его заново из 1С.');return;}
    const task=await response.json();if(token!==state.token)return;
    updateTaskBadge(task);
    if(task.job_id)await loadJob(task.job_id,token);
  }catch{toast('Нет связи с сервером задания 1С');}
}
async function loadJob(id,token){
  try{
    const response=await fetch(`/api/jobs/${id}`);if(!response.ok)return;const job=await response.json();if(token!==state.token)return;
    state.job=id;window.ReceiptUploadGate?.restored(job);$('drop-zone').hidden=true;$('empty-result').hidden=true;
    job.artifacts.forEach(artifact=>{addArtifact(artifact);updateProgressPreview(artifact);});for(const event of job.events||[])if(event.type==='stage'){state.lastEventId=Math.max(state.lastEventId,event.id||0);renderWebProgress(event);}setProgress(job.progress);
    if(job.status==='done'){
      finish(job.result);
      logEvent('Восстановлен сохранённый результат: '+job.filename);
      const layoutResponse=await fetch(`/api/jobs/${id}/files/layout.json`);
      if(layoutResponse.ok&&token===state.token){const data=await layoutResponse.json();const asset=state.assets.get('rectified');if(asset){asset.cells=data.cells;drawAsset(asset);}}
    }else if(job.status==='failed')showError(job.error);else{window.OcrActivity?.start();$('process-card').classList.add('busy');setPill('Обработка','processing');listen(id,token);}
  }catch{}
}
async function restore(){
  if(TASK)return restoreTask();
}
document.querySelector('.app-header').addEventListener('click',event=>{
  if(event.target.closest('.debug-toggle'))return;
  if(event.button!==0||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;
  event.preventDefault();location.assign('/');
});
$('debug-toggle').checked=false;
$('debug-toggle').addEventListener('change',event=>document.body.classList.toggle('debug-mode',event.target.checked));
['choose-main','choose-top','retry-button'].forEach(id=>$(id).addEventListener('click',event=>{event.stopPropagation();$('file-input').click();}));
$('drop-zone').addEventListener('click',event=>{if(!event.target.closest('button'))$('file-input').click();});
$('drop-zone').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();$('file-input').click();}});
$('file-input').addEventListener('change',event=>{upload(event.target.files[0]);event.target.value='';});
document.addEventListener('paste',event=>{const item=[...(event.clipboardData?.items||[])].find(item=>item.type.startsWith('image/'));if(item){event.preventDefault();upload(item.getAsFile());}});
let dragDepth=0;
document.addEventListener('dragenter',event=>{if(event.dataTransfer?.types.includes('Files')){event.preventDefault();dragDepth++;$('drop-overlay').hidden=false;}});
document.addEventListener('dragover',event=>{if(event.dataTransfer?.types.includes('Files')){event.preventDefault();event.dataTransfer.dropEffect='copy';}});
document.addEventListener('dragleave',()=>{dragDepth=Math.max(0,dragDepth-1);if(!dragDepth)$('drop-overlay').hidden=true;});
document.addEventListener('drop',event=>{event.preventDefault();dragDepth=0;$('drop-overlay').hidden=true;if(event.dataTransfer?.files.length)upload(event.dataTransfer.files[0]);});
$('receipt-math-alert-close').addEventListener('click',()=>$('receipt-math-alert').hidden=true);
$('receipt-math-alert-details').addEventListener('click',()=>{
  $('receipt-math-alert').hidden=true;
  const details=$('receipt-errors');
  if(!details.hidden){details.scrollIntoView({behavior:'smooth',block:'center'});details.focus({preventScroll:true});}
});
$('review-toggle').addEventListener('click',()=>{state.reviewOnly=!state.reviewOnly;$('review-toggle').setAttribute('aria-pressed',state.reviewOnly);renderRows();});
$('close-dialog').addEventListener('click',()=>$('image-dialog').close());
$('image-dialog').addEventListener('click',event=>{if(event.target===$('image-dialog'))$('image-dialog').close();});
$('print-button').addEventListener('click',()=>window.print());
document.querySelector('.receipt-total').addEventListener('mouseenter',focusReceiptTotal);
$('row-editor').addEventListener('submit',event=>{event.preventDefault();saveEditedRow();});
$('edit-row').addEventListener('change',()=>editRow(Number($('edit-row').value),false));
$('edit-name').addEventListener('input',()=>{if(state.selectedCatalog&&$('edit-name').value!==state.selectedCatalog.name){state.selectedCatalog=null;$('article-field').hidden=true;}});
$('catalog-search').addEventListener('input',()=>{
  clearTimeout(state.searchTimer);const query=$('catalog-search').value;const token=++state.searchToken;
  state.searchTimer=setTimeout(async()=>{if(query.trim().length<2)return;try{const response=await fetch('/api/catalog?q='+encodeURIComponent(query));const data=await response.json();if(token===state.searchToken)showCandidates(data.items||[]);}catch{toast('Не удалось открыть справочник');}},250);
});
$('accept-match').addEventListener('click',()=>{const row=state.rows.get(state.editingRow);const best=row?.match?.best;if(!best)return;selectCandidate({key:best.normalized,name:best.name,articles:[...new Set(best.sources.map(source=>source.article).filter(Boolean))]});if(state.selectedCatalog.articles.length<=1)saveEditedRow();});
$('save-total').addEventListener('click',()=>sendEdit({expected_total:$('edit-expected-total').value}));
$('finalize-receipt').addEventListener('click',()=>sendEdit({finalize:true}));
fetch('/api/health').then(response=>{if(!response.ok)throw Error();return response.json();}).then(config=>{state.maxUpload=config.max_upload_bytes;$('upload-limit').textContent=Math.floor(state.maxUpload/1024/1024);$('connection-text').textContent='Сервер доступен';restore();}).catch(()=>{$('connection-dot').style.background='#dba55a';$('connection-text').textContent='Нет связи с сервером';});

// STOCK_AS_OF_FOOTER_V1: дата среза остатков 1С в нижней строке главной страницы.
// Показываем только дату/время и число товаров; TTL и служебные id не выводим.
function stockDateText(iso){
  const match=/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(iso||'');
  return match?`${match[3]}.${match[2]}.${match[1]} ${match[4]}:${match[5]}`:'';
}
function renderStock(snapshots){
  const label=$('stock-as-of');
  if(!label)return;
  const rows=(snapshots||[]).filter(row=>row&&stockDateText(row.stock_as_of));
  label.hidden=false;
  if(!rows.length){label.textContent='Остатки 1С не загружены';label.title='';return;}
  const newest=rows[0];
  const extra=rows.length>1?` · баз: ${rows.length}`:'';
  label.textContent=`Остатки 1С на ${stockDateText(newest.stock_as_of)} · ${newest.record_count} тов.${extra}`;
  label.title=rows.map(row=>`${row.database_id}: ${stockDateText(row.stock_as_of)}, ${row.record_count} тов.`).join('\n');
}
function loadStock(){
  const label=$('stock-as-of');
  return fetch('/api/1c/stock-sync/summary')
    .then(response=>{if(!response.ok)throw Error();return response.json();})
    .then(data=>renderStock(data.snapshots))
    .catch(()=>{if(label){label.hidden=false;label.textContent='Остатки 1С: статус не получен';label.title='';}});
}
loadStock();
document.addEventListener('visibilitychange',()=>{if(!document.hidden)loadStock();});



// VISUAL_SYNC_V2_BEGIN — shared SSE state for the image and the electronic receipt.
// No timers simulate OCR progress. Row cards are an animation of ALREADY received OCR text.
const VisualSync = {
  scanTarget: null,
  scanPassUsed: false,
  queued: [],
  seen: new Set(),
  queueTimer: null,
  activeToken: 0,
  reset(){
    clearTimeout(this.queueTimer);this.queueTimer=null;
    this.queued.length=0;this.seen.clear();this.stopScan();
    this.activeToken=state.token;
    document.querySelectorAll('.receipt-transfer').forEach(el=>el.remove());
  },
  finished(){this.stopScan();},
  startScan(target){
    this.stopScan();
    this.scanTarget=target;this.scanPassUsed=false;
    state.visualScanTarget=target;state.visualScanPassed=false;
    const asset=state.assets.get(target);
    if(asset){asset.card.classList.add('scan-running');drawAsset(asset);}
  },
  stopScan(){
    const target=this.scanTarget;
    this.scanTarget=null;state.visualScanTarget=null;state.visualScanPassed=false;
    if(target){const asset=state.assets.get(target);if(asset){asset.card.classList.remove('scan-running');drawAsset(asset);}}
  },
  rowReady(row){
    if(!row||!Number.isInteger(Number(row.row))||!['product','tax','service'].includes(row.kind))return;
    const id=Number(row.row);
    if(this.seen.has(id)||(!row.name&&!row.quantity&&!row.price))return;
    this.seen.add(id);this.queued.push({...row,row:id});
    this.pump();
  },
  pump(){
    if(this.queueTimer!==null||!this.queued.length)return;
    const token=state.token;
    const next=()=>{
      this.queueTimer=null;
      if(token!==state.token){this.queued.length=0;return;}
      const row=this.queued.shift();if(!row)return;
      // The table and its cell coordinates may arrive after the event; defer briefly.
      const asset=state.assets.get('rectified');
      const hasSource=!!asset?.overlay.querySelector(`.cell-box[data-row="${row.row}"]`);
      const hasTarget=[...$('receipt-body').children].some(el=>Number(el.dataset.row)===row.row);
      if(hasSource&&hasTarget){
        focusRow(row.row);
        animateRowTransfer({row:row.row,kind:row.kind,name_raw:row.name,
          quantity_raw:row.quantity,price_raw:row.price});
        if(!state.result)$('process-subtitle').textContent=row.kind==='tax'
          ?'Прочитана строка НДС · ожидает проверки'
          :`Получена строка №${row.row} · название, количество и цена пока не проверены`;
      }
      // Missing layout or table is not a reason to invent a source position.
      this.queueTimer=setTimeout(next,this.queued.length>18?160:300);
    };
    next();
  },
  handle(event){
    if(event.schema!=='visual-sync-2.0')return;
    switch(event.action){
      case 'scan_start':this.startScan(event.target);break;
      case 'scan_stop':this.stopScan();break;
      case 'row_ready':this.rowReady(event.row);break;
      case 'geometry_found':
      case 'geometry_ready':
      case 'rows_ready':
      case 'checking':
      case 'matching':
        if(event.message)$('process-subtitle').textContent=event.message;
        break;
    }
  }
};
// VISUAL_SYNC_V2_END




// RECEIPT_START_SCREEN_V1_BEGIN
// Uses the app's existing upload(file) and paste event. No duplicate upload pipeline.
(function initReceiptStartScreen(){
  const zone = document.getElementById('drop-zone');
  const pasteButton = document.getElementById('paste-main');
  const pasteTop = document.getElementById('paste-top');
  if(pasteTop)pasteTop.addEventListener('click',event=>{
    event.preventDefault();event.stopPropagation();pasteButton?.click();
  });
  const overlay = document.getElementById('receipt-waiting-overlay');
  const resultScroll = document.getElementById('output-scroll');
  if(!zone || !pasteButton || !overlay || !resultScroll)return;

  // Restored jobs can start with the upload zone hidden; a MutationObserver catches
  // both an upload and a restored job without touching App's reset/restore logic.
  function syncWaitingState(){
    const waiting = !window.ReceiptUploadGate?.ready;
    document.body.classList.toggle('awaiting-receipt', waiting);
    overlay.setAttribute('aria-hidden', String(!waiting));
    resultScroll.inert = waiting;
  }
  new MutationObserver(syncWaitingState).observe(zone, {
    attributes: true, attributeFilter: ['hidden']
  });
  window.addEventListener('receipt-gate-change',syncWaitingState);
  syncWaitingState();

  pasteButton.addEventListener('click', async function(event){
    event.preventDefault();
    event.stopPropagation(); // The whole drop zone is also clickable.
    if(pasteButton.disabled)return;
    if(!navigator.clipboard || typeof navigator.clipboard.read !== 'function'){
      toast('Браузер не разрешает чтение буфера кнопкой. Нажмите Ctrl+V.');
      return;
    }
    pasteButton.disabled = true;
    try {
      const items = await navigator.clipboard.read();
      const allowed = ['image/png','image/jpeg','image/webp'];
      let imageFile = null;
      for(const item of items){
        const type = allowed.find(t=>item.types.includes(t));
        if(!type)continue;
        const blob = await item.getType(type);
        if(!blob.size)continue;
        const ext = {'image/png':'png','image/jpeg':'jpg','image/webp':'webp'}[type];
        imageFile = new File([blob], `clipboard-${Date.now()}.${ext}`, {type});
        break;
      }
      if(!imageFile){
        toast('В буфере нет изображения. Скопируйте фото чека и попробуйте снова.');
        return;
      }
      await upload(imageFile);
    } catch(error){
      // Browser permission can be denied even on localhost. Ctrl+V already works.
      toast('Не получилось прочитать буфер. Вставьте фото клавишами Ctrl+V.');
    } finally {
      pasteButton.disabled = false;
    }
  });
})();
// RECEIPT_START_SCREEN_V1_END


// RECEIPT_UPLOAD_GATE_V11_BEGIN — только события существующего upload(), SSE и DOM.
window.ReceiptUploadGate = (() => {
  let ready=false;
  let stateName='idle';
  let currentToken=null;
  let acceptedToken=null;
  let shownToken=null;
  let badgeTimer=null;

  const copy={
    idle:['Загрузите чек Торгсофта','Электронный чек появится после обнаружения товарной таблицы.'],
    uploading:['Загружаем фото чека','Проверяем, что фотография успешно передана на сервер.'],
    searching:['Фото получено','Ищем товарную таблицу. Электронный чек станет доступен после обнаружения её строк.'],
    ready:['Фото принято','Показываем фактический ход обработки и формируем электронный чек.'],
    failed:['Таблица пока не найдена','Не удалось продолжить обработку. Проверьте фото или загрузите другой чек.'],
  };
  const notify=() => window.dispatchEvent(new Event('receipt-gate-change'));
  function updateCopy(){
    const pair=copy[stateName]||copy.idle;
    const title=document.getElementById('receipt-gate-title');
    const message=document.getElementById('receipt-gate-description');
    if(title)title.textContent=pair[0];
    if(message)message.textContent=pair[1];
  }
  function transition(next){stateName=next;updateCopy();notify();}
  function clearBadge(){
    if(badgeTimer!==null){clearTimeout(badgeTimer);badgeTimer=null;}
    document.querySelectorAll('.receipt-upload-check').forEach(el=>el.remove());
  }
  function showBadge(){
    if(acceptedToken===null||acceptedToken!==currentToken||shownToken===currentToken)return;
    const original=state.assets?.get('original');
    if(!original?.stage)return;
    shownToken=currentToken;
    const badge=document.createElement('div');
    badge.className='receipt-upload-check';
    badge.setAttribute('role','status');badge.setAttribute('aria-label','Фото чека загружено');
    const circle=document.createElement('span');circle.className='receipt-upload-check-icon';
    circle.append(icon('check'));
    badge.append(circle,document.createTextNode('Фото загружено'));
    original.stage.append(badge);
    badgeTimer=setTimeout(()=>{
      badge.classList.add('receipt-upload-leave');
      badgeTimer=setTimeout(()=>{badge.remove();badgeTimer=null;},310);
    },1900);
  }
  function reset(){
    clearBadge();ready=false;stateName='idle';currentToken=null;
    acceptedToken=null;shownToken=null;transition('idle');
  }
  function begin(token){
    clearBadge();ready=false;currentToken=token;acceptedToken=null;
    shownToken=null;transition('uploading');
  }
  function accepted(token){
    if(token!==state.token||token!==currentToken)return;
    acceptedToken=token;
    // OCR_WEB_PROGRESS: a successful HTTP 202 is sufficient to open the
    // progress panel. Receipt rows still appear only from real SSE events.
    ready=true;transition('ready');
    showBadge();
  }
  function asset(key){if(key==='original')showBadge();}
  function hasTableCells(cells){
    if(!Array.isArray(cells)||!cells.length)return false;
    const rows=new Map();
    for(const cell of cells){
      if(!cell||!Number.isInteger(Number(cell.row))||!Array.isArray(cell.box)||cell.box.length!==4)continue;
      if(!['number','name','quantity','price'].includes(cell.role))continue;
      const valid=cell.box.every(p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite));
      if(!valid)continue;
      if(!rows.has(cell.row))rows.set(cell.row,new Set());
      rows.get(cell.row).add(cell.role);
    }
    return [...rows.values()].some(roles=>['number','name','quantity','price'].every(role=>roles.has(role)));
  }
  function tableReady(){
    if(ready)return;
    ready=true;transition('ready');
  }
  function onPipeline(event){
    if(!event||ready)return;
    // A contour or a phase alone does NOT unlock the result.
    if(event.type==='cells'&&hasTableCells(event.cells))tableReady();
    else if(event.type==='draft_rows'&&Array.isArray(event.rows)&&event.rows.some(
        row=>row?.kind==='product'&&String(row.name_raw??row.name??'').trim()))tableReady();
  }
  function finished(result){
    // Restore of an already completed job has no SSE history in the current browser.
    if(result&&Array.isArray(result.rows)&&result.rows.length)tableReady();
  }
  function restored(job){
    clearBadge();currentToken=state.token;acceptedToken=null;shownToken=null;
    ready=Boolean(job&&job.status!=='failed');
    if(job?.status==='done')finished(job.result);
    else if(job?.status==='failed')transition('failed');
    else transition('ready');
  }
  function failed(message){
    if(ready)return;
    transition('failed');
    const desc=document.getElementById('receipt-gate-description');
    if(desc&&message)desc.textContent=String(message).slice(0,250)+' Загрузите новое фото.';
  }
  updateCopy();
  return {get ready(){return ready;},reset,begin,accepted,asset,hasTableCells,
          tableReady,onPipeline,finished,restored,failed};
})();
window.dispatchEvent(new Event('receipt-gate-change'));
// RECEIPT_UPLOAD_GATE_V11_END

// RECEIPT_IDLE_INVITATION_V1_BEGIN — UI-only, no timers tied to OCR or upload.
(function receiptIdleInvitation(){
  const zone=document.getElementById('drop-zone');
  const landing=document.querySelector('.receipt-idle-landing');
  const label=document.querySelector('.receipt-waiting-card strong');
  if(!zone || !landing || !label)return;
  const xmlns='http://www.w3.org/2000/svg';
  const svg=document.createElementNS(xmlns,'svg');
  svg.classList.add('receipt-idle-guide');
  svg.setAttribute('aria-hidden','true');
  svg.setAttribute('focusable','false');
  const halo=document.createElementNS(xmlns,'path');halo.classList.add('receipt-idle-guide-halo');
  const arc=document.createElementNS(xmlns,'path');arc.classList.add('receipt-idle-guide-path');
  const head=document.createElementNS(xmlns,'path');head.classList.add('receipt-idle-guide-tip');
  svg.append(halo,arc,head);document.body.appendChild(svg);
  let frame=0;
  let active=false;
  const motion=window.matchMedia('(prefers-reduced-motion: reduce)');
  function geometry(){
    frame=0;
    if(!active || window.innerWidth<=760 || !label.isConnected || !zone.isConnected){
      svg.classList.remove('receipt-idle-guide-on');return;
    }
    const src=label.getBoundingClientRect();
    const dest=zone.getBoundingClientRect();
    // Начало непосредственно у левого края надписи справа;
    // наконечник направлен к правой стороне зоны загрузки слева.
    const sx=src.left-13,sy=src.top+src.height*.52;
    const ex=dest.right-Math.min(65,dest.width*.12);
    const ey=dest.top+Math.min(120,dest.height*.23);
    if(sx-ex<90){svg.classList.remove('receipt-idle-guide-on');return;}
    const d=`M ${sx.toFixed(1)} ${sy.toFixed(1)} C ${(sx-85).toFixed(1)} ${sy.toFixed(1)}, ${(ex+80).toFixed(1)} ${(ey-12).toFixed(1)}, ${ex.toFixed(1)} ${ey.toFixed(1)}`;
    halo.setAttribute('d',d);arc.setAttribute('d',d);
    head.setAttribute('d',`M ${(ex+13).toFixed(1)} ${(ey-8).toFixed(1)} L ${ex.toFixed(1)} ${ey.toFixed(1)} L ${(ex+14).toFixed(1)} ${(ey+7).toFixed(1)}`);
    svg.classList.add('receipt-idle-guide-on');
  }
  function schedule(){if(!frame)frame=requestAnimationFrame(geometry);}
  function refresh(){
    // Показываем анимацию только пока у пользователя нет фото.
    // Скрываем при начале upload(), когда drop-zone становится hidden.
    const next=!zone.hidden && !document.querySelector('#image-list .image-card');
    active=next;
    document.body.classList.toggle('receipt-idle-active',next);
    if(next)schedule();else svg.classList.remove('receipt-idle-guide-on');
  }
  new MutationObserver(refresh).observe(zone,{attributes:true,attributeFilter:['hidden']});
  // Копия правой панели может меняться в Upload Gate v1.1.
  window.addEventListener('receipt-gate-change',refresh);
  window.addEventListener('resize',schedule,{passive:true});
  window.addEventListener('scroll',schedule,{passive:true,capture:true});
  const resizer=typeof ResizeObserver==='function'?new ResizeObserver(schedule):null;
  if(resizer){resizer.observe(label);resizer.observe(zone);}
  motion.addEventListener?.('change',schedule);
  refresh();
})();
// RECEIPT_IDLE_INVITATION_V1_END
