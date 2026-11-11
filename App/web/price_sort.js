/* Client-side ordering of displayed results; matching scores and votes remain unchanged. */
(function(){
  function attach(table){
    const header=table.querySelector('th[data-price-header]');
    const body=table.tBodies[0];
    if(!header||!body||header.dataset.sortReady)return;
    header.dataset.sortReady='1';
    const button=document.createElement('button');button.type='button';button.className='price-sort-button';button.textContent=header.textContent.trim()+' ↕';button.title='Сортировать по цене';
    header.replaceChildren(button);header.setAttribute('aria-sort','none');
    let ascending=false;
    button.addEventListener('click',()=>{
      ascending=!ascending;
      const rows=Array.from(body.rows);
      rows.sort((a,b)=>{
        const left=a.querySelector('[data-price]')?.dataset.price;
        const right=b.querySelector('[data-price]')?.dataset.price;
        const x=left==null||left===''?null:Number(left),y=right==null||right===''?null:Number(right);
        const validX=x!==null&&Number.isFinite(x),validY=y!==null&&Number.isFinite(y);
        if(!validX||!validY)return validX?-1:validY?1:0;
        return ascending?x-y:y-x;
      });
      body.append(...rows);header.setAttribute('aria-sort',ascending?'ascending':'descending');
      button.textContent=button.textContent.replace(/ [↕↑↓]$/,ascending?' ↑':' ↓');
    });
  }
  window.PriceSort={attach};
})();
