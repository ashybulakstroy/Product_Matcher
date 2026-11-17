"""Offline reports. Paths only inside the new trial's private data/output run."""
from __future__ import annotations
import html
import json
from pathlib import Path
from datetime import datetime, timezone


def jdump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def html_report(outdir,records,metadata):
    def e(x):return html.escape(str(x),quote=True)
    rows=[]
    for row in records:
        gt=row.get('ground_truth',{})
        imgs=[]
        for label,key in [('Выпрямление','rectified'),('Линии/сетка','overlay'),('Перенос на фото','source_overlay')]:
            if row.get(key):imgs.append(f'<a href="{e(row[key])}">{e(label)}</a>')
        rows.append('<tr>'+''.join(f'<td>{e(val)}</td>' for val in
            (row['job_id'][:12],row['method'],row['status'],row.get('reason',''),
             row.get('duration_sec',''),row.get('metrics',{}).get('valid_fraction',''),
             gt.get('status','GT_MISSING'),gt.get('mean_px','')))+
             '<td>'+ ' · '.join(imgs)+'</td></tr>')
    heading=['Чек','Алгоритм','Статус','Причина','Сек.','Геом. доля valid','Эталон','Ошибка GT, px','Изображения']
    style='''body{font:15px system-ui,Segoe UI,sans-serif;margin:24px;color:#17212b;background:#f8fafc}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d0d8e0;padding:8px;text-align:left}th{background:#e6eff6}tr:nth-child(even){background:#f4f7f9}a{color:#1463a5}code{background:#eaeef1;padding:2px 5px} .warning{border-left:4px solid #c58b13;padding:12px;background:#fff9e9}'''
    page=f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Standalone Geometry Lab</title><style>{style}</style></head><body>
    <h1>Проверка геометрии чеков Торгсофта — автономный Trial V3</h1>
    <p>Создано {e(datetime.now(timezone.utc).isoformat())}. Обработано попыток: {len(records)}.</p>
    <div class="warning"><b>Статус SUCCESS означает только успешный расчёт, а НЕ правильную сетку.</b>
    Без независимой ручной разметки GT_MISSING запрещено заявлять о точности совмещения. Зелёные линии — автоматический повторный детектор, синие — сетка метода.</div>
    <p>Источник: только чтение основного <code>web_outputs</code>. Эти результаты не импортируют и не используют V1/V2 или старые отчёты.</p>
    <p>Информация о прогоне: <a href="run.json">run.json</a>. Для каждого метода есть JSON с метриками.</p>
    <table><thead><tr>{''.join(f'<th>{e(k)}</th>' for k in heading)}</tr></thead><tbody>{''.join(rows)}</tbody></table>
    </body></html>'''
    (outdir/'summary.html').write_text(page,encoding='utf-8')
