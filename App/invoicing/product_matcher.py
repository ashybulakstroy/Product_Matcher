"""Product matcher accepted from the isolated v1 experiment; writes no links.

Ranks product-type/descriptor coverage before whole-string similarity. Scores
are lexical rankings, not calibrated probabilities or proof of interchangeability.
"""
from __future__ import annotations
import re
from decimal import Decimal
from rapidfuzz import fuzz

STOP={'для','при','под','над','без','или','бухте','бухта','упаковке','упаковка','офиц','официал'}


def measurements(text):
    units={'мм':('length',Decimal('.001')),'см':('length',Decimal('.01')),
           'м':('length',Decimal(1)),'вт':('power',Decimal(1)),
           'в':('voltage',Decimal(1)),'v':('voltage',Decimal(1))}
    found=set()
    for number,unit in re.findall(r'(\d+(?:[.,]\d+)?)\s*(мм|см|м|вт|в|v)(?![a-zа-я])',text.casefold()):
        dimension,multiplier=units[unit]
        found.add((dimension,Decimal(number.replace(',','.'))*multiplier))
    return found


def words(text):
    return [w for w in re.findall(r'[a-zа-яё]+',text.casefold().replace('ё','е'))
            if len(w)>=3 and w not in STOP]


def affinity(a,b):
    if a==b:return 1.0
    if 3<=len(a)<=4 and b.startswith(a):return .9  # Manual prefixes: «кабель наг».
    # Closely related word forms: нагревающийся / нагревательный.
    prefix=0
    for x,y in zip(a,b):
        if x!=y:break
        prefix+=1
    if prefix>=6:return .85
    if min(len(a),len(b))>=4 and fuzz.ratio(a,b)>=85:return .8
    return 0.0


def rank_products(query,items,limit=10):
    normalized=' '.join(query.casefold().split())
    exact=[item for item in items if item.get('code') and item['code'].casefold()==normalized]
    if exact:return [dict(item,match_score=100.,reasons=['точный код 1С'],exact_code=True) for item in exact[:limit]]
    terms=words(query)
    if not terms:return []
    ranked=[]
    for item in items:
        target=words(item['name'])
        hits=[max((affinity(word,other) for other in target),default=0) for word in terms]
        # A brand/number alone does not provide a product-type match.
        if hits[0]<.8:continue
        primary=hits[:2]
        type_score=primary[0]
        descriptor=primary[1] if len(primary)>1 else type_score
        lexical=fuzz.WRatio(query.casefold(),item['name'].casefold())/100
        numbers=set(re.findall(r'\d+(?:[.,]\d+)?',query))
        other_numbers=set(re.findall(r'\d+(?:[.,]\d+)?',item['name']))
        measures=measurements(query);other_measures=measurements(item['name'])
        numeric=len(measures&other_measures)/len(measures) if measures else 0
        score=100*(.35*type_score+.35*descriptor+.15*sum(hits)/len(hits)+.10*lexical+.05*numeric)
        reasons=['совпали: '+', '.join(w for w,h in zip(terms,hits) if h)]
        missing=[w for w,h in zip(terms,hits) if not h]
        if missing:reasons.append('не совпали: '+', '.join(missing))
        if numbers-other_numbers:reasons.append('числа требуют проверки: '+', '.join(sorted(numbers-other_numbers)))
        if measures-other_measures:reasons.append('размеры/единицы не совпали: '+str(sorted(measures-other_measures)))
        # A candidate may be useful for manual review without being a safe
        # automatic suggestion; no automatic acceptance is implemented here.
        ranked.append(dict(item,match_score=round(score,2),reasons=reasons,
                           missing_terms=missing,exact_code=False))
    ranked.sort(key=lambda item:(-item['match_score'],item['name'],item.get('code') or ''))
    return ranked[:limit]
