"""Read-only fallback proposals from the local Torgsoft→1C AI CSV.

Codes resolve against the live 1C catalog. CSV names, stocks and scores never
replace catalog identities, receipt prices, or explicit operator confirmations.
"""
import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from decimal import Decimal, InvalidOperation
from App.invoicing.user_mapping import normalize_mapping_key

DIRECTORY=Path(__file__).resolve().parents[2]/'data/reference/AI_matching'


def signature(directory=None):
    return tuple((str(p.resolve()),p.stat().st_mtime_ns,p.stat().st_size)
                 for p in sorted(Path(directory or DIRECTORY).glob('*.csv')) if not p.name.startswith('~$'))


def _value(row,*names):
    return next((str(row.get(name) or '').strip() for name in names if row.get(name)), '')


def _score(value):
    try:
        number=Decimal(str(value).replace(',','.'))
        return float(max(Decimal(0),min(Decimal(100),number))) if number.is_finite() else 0.
    except (InvalidOperation,ValueError):
        return 0.


@lru_cache(maxsize=2)
def _load(files):
    names=defaultdict(list);articles=defaultdict(list);errors=[];count=0
    for filename,_,_ in files:
        try:
            with Path(filename).open(encoding='utf-8-sig',newline='') as source:
                reader=csv.DictReader(source,delimiter=';')
                if not reader.fieldnames or 'Название Торгсофт' not in reader.fieldnames:
                    errors.append(f'{Path(filename).name}: отсутствует колонка Название Торгсофт');continue
                for row in reader:
                    count+=1
                    name=_value(row,'Название Торгсофт');article=_value(row,'Артикул Торгсофт')
                    if not name:continue
                    candidates=[dict(code=_value(row,'Рекомендуемый код 1С','Код кандидата 1С'),
                                     score=_score(_value(row,'Оценка совпадения, %','Оценка, %')),rank=1)]
                    for rank in range(2,6):
                        candidates.append(dict(code=_value(row,f'Альтернатива №{rank} код 1С',f'Кандидат №{rank} код 1С'),
                            score=_score(_value(row,f'Альтернатива №{rank} %',f'Кандидат №{rank} %')),rank=rank))
                    entry=dict(name=name,article=article,file=Path(filename).name,
                               status=_value(row,'Статус соответствия','Статус'),
                               risks=_value(row,'Риски и несовпадения','Конфликты характеристик'),
                               basis=_value(row,'Основание','Основание сопоставления'),
                               candidates=[c for c in candidates if c['code']])
                    names[normalize_mapping_key(name)].append(entry)
                    if article:articles[article].append(entry)
        except (OSError,UnicodeError,csv.Error) as error:
            errors.append(f'{Path(filename).name}: {error}')
    return dict(names=names,articles=articles,errors=errors,rows=count,files=len(files))


def proposals(query,catalog_index,*,article=None,limit=10,directory=None):
    data=_load(signature(directory));errors=list(data['errors'])
    key=article or query.strip()
    entries=(data['articles'].get(key,[]) if key in data['articles'] else
             data['names'].get(normalize_mapping_key(query),[]))
    found={}
    for entry in entries:
        for candidate in entry['candidates']:
            variants=catalog_index['by_code'].get(normalize_mapping_key(candidate['code']),[])
            if len(variants)!=1:
                errors.append(f"ИИ-сопоставление: код 1С {candidate['code']} отсутствует или неоднозначен")
                continue
            item=variants[0]
            result=dict(item,score=candidate['score'],match_method='ai_matching',
                        ai_source_name=entry['name'],ai_source_article=entry['article'],
                        ai_status=entry['status'],ai_risks=entry['risks'],ai_basis=entry['basis'],
                        ai_file=entry['file'],ai_candidate_rank=candidate['rank'])
            if item['id'] not in found or result['score']>found[item['id']]['score']:
                found[item['id']]=result
    items=sorted(found.values(),key=lambda row:(-row['score'],row['ai_candidate_rank'],row['code']))[:limit]
    return dict(items=items,errors=list(dict.fromkeys(errors)),rows=data['rows'],files=data['files'])
