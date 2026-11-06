"""Build a complete local invoice without hiding unmatched receipt rows."""
import hashlib
import json
import os
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from App.invoicing.learned_mapping import apply_training_pairs, training_alternatives, CENT, VAT_TOLERANCE
from App.invoicing.user_mapping import normalize_mapping_key
from App.invoicing.catalog_1c import catalog_item, catalog_named_item, suggest_catalog
from App.invoicing.receipt_names import torgsoft_name
from App.invoicing.invoice_presentation import print_fields
from App.invoicing.transform_candidates import resolve_source

ROOT = Path(__file__).resolve().parents[2]
ISSUER_PATH = ROOT / "Data/invoice_issuer.json"
KNOWLEDGE_PATH = ROOT / "Data/reference/User_Mapping/Torgsoft_1C_mapping.json"
UNIT_PRICE_STEP = Decimal("0.000000000001")


def _backup_knowledge(path):
    """Versioned backup of user-confirmed mapping rules before any rewrite."""
    import shutil
    import time
    if not Path(path).exists():
        return
    history = Path(path).parent / "history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(path, history / f"{Path(path).stem}_{stamp}.json")
    backups = sorted(history.glob(f"{Path(path).stem}_*.json"))
    for stale in backups[:-5]:
        stale.unlink()


def transformed_numbers(amount, quantity, price=None):
    if price is not None:
        price = Decimal(str(price))
        quantity = (amount / price).quantize(UNIT_PRICE_STEP, rounding=ROUND_HALF_UP)
    else:
        quantity = Decimal(str(quantity))
        price = (amount / quantity).quantize(UNIT_PRICE_STEP, rounding=ROUND_HALF_UP) if quantity else Decimal(0)
    if quantity <= 0 or quantity > Decimal("1000000000") or price <= 0:
        raise ValueError("Проверьте количество и цену услуги")
    return {"quantity": str(quantity), "price": str(price)}


def database_context(task=None):
    binding = (task or {}).get("integration") or {}
    return binding.get("database_id") or binding.get("base_id") or os.environ.get("TORGSOFT_CATALOG_DATABASE_ID") or "local"


def _card_key(identity, database_id):
    return "card:" + hashlib.sha256(json.dumps([database_id, identity], ensure_ascii=False).encode("utf-8")).hexdigest()[:32]


def _variant_key(rule):
    return (rule.get("catalog_id") or normalize_mapping_key(rule["name"]), rule["unit"], rule["service"])


def confirmed_record(knowledge, source, database_id, article=None):
    explicit = article or source.get("torgsoft_article")
    resolved = resolve_source(source, explicit)
    identity = "torgsoft:article:" + explicit if explicit else (resolved.get("source") or {}).get("identity")
    if identity:
        record = knowledge.get(_card_key(identity, database_id))
        if record:
            return record
    # A proposal refresh may remove a card. Confirmed source identities survive it.
    if not identity and not resolved.get("source_options"):
        matches = [rule for rule in knowledge.values() if rule.get("source_identity") and
                   rule.get("database_id") == database_id and
                   normalize_mapping_key(rule.get("source_card", {}).get("name", "")) == normalize_mapping_key(torgsoft_name(source) or "")]
        if len(matches) == 1:
            return matches[0]
    return knowledge.get(normalize_mapping_key(torgsoft_name(source)))


def training_mapping_conflicts(pair):
    knowledge = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8")) if KNOWLEDGE_PATH.exists() else {}
    errors = []
    sources = pair.get("training_receipt", pair["predzakaz"])["items"]
    targets = {row["row_id"]: row for row in pair["invoice"]["items"]}
    for link in pair["links"]:
        source, target = sources[link["source_index"]], targets[link["invoice_row_id"]]
        record = confirmed_record(knowledge, source, pair.get("database_id") or database_context(), link.get("source_article"))
        product = catalog_item(link.get("catalog_id")) if link.get("catalog_id") else catalog_named_item(target["name"], target["unit"])
        service = bool(link.get("service") or (product and product.get("service")) or target["unit"] == "усл")
        for rule in record.get("alternatives", [record]) if record else []:
            same_target = ((product["id"] == rule.get("catalog_id")) if product and rule.get("catalog_id") else
                           normalize_mapping_key(target["name"]) == normalize_mapping_key(rule["name"]))
            if not same_target or target["unit"] != rule["unit"] or service != rule["service"]:
                continue
            factor = Decimal(str(target["quantity"])) / Decimal(str(source["quantity"]))
            differs = (Decimal(str(target["quantity"])) != Decimal(rule.get("service_quantity", "1"))) if service else factor != Decimal(rule["quantity_factor"])
            if differs:
                errors.append(f"Конфликт коэффициента с ручным маппингом: {torgsoft_name(source)} → {target['name']}")
    return errors


def _current_rule(rule):
    if not rule.get("catalog_id"):
        return dict(rule) if rule.get("service") else None
    product = catalog_item(rule["catalog_id"])
    if not product:
        return None
    if rule.get("source_identity") and (product["unit"] != rule.get("catalog_unit", rule["unit"]) or
                                        bool(product.get("service")) != bool(rule.get("catalog_service", rule["service"]))):
        return None
    return {**rule, "name": product["name"], "code": product.get("code"), "uuid": product.get("uuid")}


def _rule_target(rule, source_quantity, amount):
    quantity = Decimal(rule.get("service_quantity", "1")) if rule["service"] else source_quantity * Decimal(rule["quantity_factor"])
    if quantity <= 0:
        return None
    return {"name": rule["name"], "unit": rule["unit"], "code": rule.get("code"),
            "catalog_id": rule.get("catalog_id"), "uuid": rule.get("uuid"), "service": rule["service"],
            "quantity_mode": "service" if rule["service"] else "factor",
            "quantity_factor": str(quantity / source_quantity), "quantity": str(quantity),
            "price": str((amount / quantity).quantize(UNIT_PRICE_STEP, rounding=ROUND_HALF_UP)),
            "amount": str(amount), "conditions": rule.get("conditions", {})}


def _training_targets(source, amount, database_id, article=None, rules=None):
    quantity = Decimal(str(source.get("quantity") or 0))
    if quantity <= 0:
        return []
    targets = []
    for rule in rules if rules is not None else training_alternatives(source, database_id=database_id):
        if article:
            identity = rule.get("source_identity") or (resolve_source({"name": rule["source_name"]}).get("source") or {}).get("identity")
            if identity != "torgsoft:article:" + article:
                continue
        product = catalog_item(rule.get("catalog_id")) if rule.get("catalog_id") else catalog_named_item(rule["name"], rule["unit"])
        service = rule["quantity_mode"] == "service"
        if not product and not service:
            continue
        current = _current_rule({**rule, "catalog_id": product["id"] if product else None,
                                 "service": service})
        if current:
            target = _rule_target(current, quantity, Decimal(str(amount)))
            if target:
                targets.append(target)
    return targets


def _unique_targets(targets):
    unique = {}
    for target in targets:
        key = (*_variant_key(target), Decimal(target["quantity_factor"]))
        unique.setdefault(key, target)
    return list(unique.values())


def confirmed_candidates(source, amount, database_id=None, article=None):
    database_id = database_id or database_context()
    knowledge = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8")) if KNOWLEDGE_PATH.exists() else {}
    record = confirmed_record(knowledge, source, database_id, article)
    quantity = Decimal(str(source.get("quantity") or 0))
    if quantity <= 0:
        return []
    targets = _training_targets(source, amount, database_id, article)
    for rule in record.get("alternatives", []) if record and record.get("source_identity") else []:
        current = _current_rule(rule)
        target = _rule_target(current, quantity, Decimal(str(amount))) if current else None
        if target:
            targets.append(target)
    items = []
    for target in _unique_targets(targets):
        product = catalog_item(target.get("catalog_id")) or {}
        items.append({**product, **target, "id": target["catalog_id"], "approved": True,
                      "source": product.get("source", "Подтверждённая услуга"), "sheet": product.get("sheet", "Услуги"),
                      "rank": None, "stock_snapshot": None, "stock_date": None,
                      "review_notes": ["Подтверждённый коэффициент: " + target["quantity_factor"]]})
    return items


def remember_selection(source, selection, path=None):
    remember_selections([(source, selection)], path)


def remember_selections(pairs, path=None, database_id=None):
    path = path or KNOWLEDGE_PATH
    knowledge = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    updates = {}
    database_id = database_id or database_context()
    for source, selection in pairs:
        quantity = Decimal(str(source["quantity"]))
        if quantity <= 0:
            raise ValueError("Нельзя обучать связь с нулевым количеством")
        key = normalize_mapping_key(torgsoft_name(source))
        service = selection.get("service", selection["unit"].rstrip(".") == "усл")
        rule = {"catalog_id": selection.get("catalog_id"), "name": selection["name"], "unit": selection["unit"],
                "code": selection.get("code"),
                "quantity_factor": str(Decimal(selection["quantity"]) / quantity), "service": service}
        if service:
            rule["service_quantity"] = str(Decimal(str(selection["quantity"])).normalize())
            rule["quantity_factor"] = "1"
        if selection.get("source_identity"):
            key = _card_key(selection["source_identity"], database_id)
            product = catalog_item(selection.get("catalog_id"))
            rule.update(source_identity=selection["source_identity"], source_article=selection["source_article"],
                        source_card=selection["source_card"], database_id=database_id, conditions={},
                        catalog_unit=product["unit"] if product else selection["unit"],
                        catalog_service=bool(product.get("service")) if product else service)
            previous = updates.get(key) or knowledge.get(key) or {}
            alternatives = { _variant_key(value): value for value in previous.get("alternatives", []) }
            variant = _variant_key(rule)
            if key in updates and variant in alternatives and alternatives[variant] != rule:
                raise ValueError(f"Для одной карточки указаны разные коэффициенты: {torgsoft_name(source)}")
            alternatives[variant] = rule
            updates[key] = {**rule, "alternatives": list(alternatives.values())}
            continue
        if key in updates and updates[key] != rule:
            raise ValueError(f"Для одинакового имени указаны разные связи: {torgsoft_name(source)}")
        updates[key] = rule
    knowledge.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(knowledge, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def source_fingerprint(row):
    fields = {"name": torgsoft_name(row), **{key: str(Decimal(str(row.get(key) or 0)).normalize()) for key in ("quantity", "price")}}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def build_generation(receipt, selections=None, suggest=False, database_id=None, *, vat_in_line_amounts=True, preserve_receipt_quantities=False):
    database_id = database_id or database_context()
    learned = apply_training_pairs(receipt, database_id=database_id)
    mapped = {row["source_index"]: row for row in learned["items"]}
    selections = selections or {}
    knowledge = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8")) if KNOWLEDGE_PATH.exists() else {}
    items, warnings = [], []
    source_rows = receipt.get("items", [])
    net = [Decimal(str(row.get("quantity") or 0)) * Decimal(str(row.get("price") or 0)) for row in source_rows]
    net_total = sum(net, Decimal(0))
    vat = Decimal(str(receipt.get("vat_amount") or 0))
    allocated = Decimal(0)
    for index, source in enumerate(source_rows):
        row_id = source.get("row_id", f"predzakaz_row_{index+1:03d}")
        share = (vat * net[index] / net_total).quantize(CENT, rounding=ROUND_HALF_UP) if net_total else Decimal(0)
        if index == len(source_rows) - 1:
            share = vat - allocated
        allocated += share
        amount = (net[index] + (share if vat_in_line_amounts else Decimal(0))).quantize(CENT)
        quantity = Decimal(str(source.get("quantity") or 0))
        row = {"name": None, "unit": "", "quantity": str(quantity),
               "price": str((amount / quantity).quantize(CENT, rounding=ROUND_HALF_UP)) if quantity else "0",
               "amount": str(amount), "match_status": "unmatched", "service": False}
        if index in mapped:
            row.update(mapped[index], match_status="learned")
            row["service"] = row.get("quantity_mode") == "service"
        source_name = torgsoft_name(source)
        selection = selections.get(row_id)
        valid_selection = selection and selection.get("source_fingerprint") == source_fingerprint(source)
        remembered = confirmed_record(knowledge, source, database_id, selection.get("source_article") if valid_selection else None)
        current = _current_rule(remembered) if remembered else None
        if remembered and remembered.get("source_identity") and not current:
            row.update(name=None, match_status="unmatched")
            warnings.append(f"Перепроверьте изменённую карточку или единицу 1С: {source_name}")
        if current:
            remembered = current
            quantity = Decimal(remembered.get("service_quantity", "1")) if remembered["service"] else Decimal(str(source["quantity"])) * Decimal(remembered["quantity_factor"])
            if quantity > 0:
                row.update(name=remembered["name"], unit=remembered["unit"], quantity=str(quantity), amount=str(amount),
                           price=str((amount / quantity).quantize(UNIT_PRICE_STEP, rounding=ROUND_HALF_UP)),
                           match_status="remembered", catalog_id=remembered.get("catalog_id"), service=remembered["service"])
                if remembered.get("source_identity"):
                    row.update({key: remembered[key] for key in ("source_identity", "source_article", "source_card", "database_id")})
        if valid_selection:
            quantity = Decimal(selection["quantity"])
            row.update(name=selection["name"], unit=selection["unit"], quantity=str(quantity),
                       price=str((amount / quantity).quantize(UNIT_PRICE_STEP, rounding=ROUND_HALF_UP)),
                       amount=str(amount), match_status="manual", catalog_id=selection.get("catalog_id"),
                       service=selection.get("service", selection["unit"].rstrip(".") == "усл"))
            if row["service"] and selection.get("price") is not None:
                row.update(transformed_numbers(amount, quantity, selection["price"]))
            row.update({key: selection[key] for key in ("source_identity", "source_article", "source_card") if key in selection})
        elif selection:
            warnings.append(f"Повторно проверьте связь после правки чека: {source['name']}")
            row["name"] = None
            row["match_status"] = "unmatched"
        row.update(row_id=row_id, source_index=index, source_name=source_name,
                   source_quantity=str(source["quantity"]), source_price=str(source["price"]), source_amount=str(net[index]))
        if suggest and not row["name"] and not selection:
            product = suggest_catalog(source_name, source.get('article'))
            if product:
                service = product.get("service", product["unit"].rstrip(".") == "усл")
                quantity = Decimal(1) if service else Decimal(row["quantity"])
                row.update(name=product["name"], unit=product["unit"], catalog_id=product["id"], match_status="suggested",
                           service=service,
                           suggestion_score=product["score"], quantity=str(quantity),
                           price=str((amount / quantity).quantize(UNIT_PRICE_STEP, rounding=ROUND_HALF_UP)) if quantity else "0")
                row['suggestion_method']=product.get('match_method')
                if product.get('match_method')=='ai_matching':
                    row.update({key:product[key] for key in ('ai_source_name','ai_source_article','ai_status','ai_risks','ai_basis','ai_file') if key in product})
                warnings.append(f"Подтвердите предложенную номенклатуру 1С: {source_name}")
        if (not vat_in_line_amounts or preserve_receipt_quantities) and not valid_selection:
            # Automatic catalog/learned identity must not reprice the receipt.
            # An explicit operator quantity conversion remains a separate action.
            quantity = Decimal(str(source.get('quantity') or 0))
            automatic_amount=amount if vat_in_line_amounts else net[index].quantize(CENT)
            automatic_price=(automatic_amount/quantity).quantize(UNIT_PRICE_STEP,rounding=ROUND_HALF_UP) if vat_in_line_amounts and quantity else Decimal(str(source.get('price') or 0))
            row.update(quantity=str(quantity), price=str(automatic_price), amount=str(automatic_amount))
        if not row["name"]:
            warnings.append(f"Выберите номенклатуру 1С: {source_name}")
        if Decimal(row["quantity"]) <= 0 or Decimal(row["price"]) <= 0:
            warnings.append(f"Проверьте количество и цену: {source['name']}")
        if row["name"] and not row["unit"]:
            warnings.append(f"Укажите единицу измерения: {row['name']}")
        product = catalog_item(row.get("catalog_id")) if row.get("catalog_id") else (
            catalog_named_item(row["name"], row["unit"]) if row["name"] and not row["service"] else None)
        row["code"] = product.get("code") if product else None
        row['uuid'] = product.get('uuid') if product else None
        if product and product.get("code"):
            row["catalog_id"] = product["id"]
        if row["match_status"] in {"manual", "remembered", "learned"}:
            allowed = _training_targets(source, amount, database_id, row.get("source_article"),
                                        rules=(mapped.get(index) or {}).get("alternatives", []))
        else:
            allowed = []
        if remembered and remembered.get("source_identity") and row["match_status"] in {"manual", "remembered", "learned"}:
            for option in remembered.get("alternatives", []):
                current_option = _current_rule(option)
                if current_option and Decimal(str(source["quantity"])) > 0:
                    target = _rule_target(current_option, Decimal(str(source["quantity"])), amount)
                    if target:
                        allowed.append(target)
        if allowed:
            row["approved_targets"] = _unique_targets(allowed)
        row.pop("alternatives", None)
        items.append(row)
    if not items:
        warnings.append("Нет товарных позиций")
    total = receipt.get("total")
    if total is None or abs(net_total + vat - Decimal(str(total))) > VAT_TOLERANCE:
        warnings.append("Арифметика исходного чека не сходится")
    if total is not None:
        difference = Decimal(str(total)) - sum((Decimal(row["amount"]) for row in items), Decimal(0)) - (vat if not vat_in_line_amounts else Decimal(0))
        if abs(difference) > VAT_TOLERANCE:
            warnings.append("Сумма позиций 1С отличается от итога чека")
        elif difference and items and Decimal(items[-1]["quantity"]) > 0 and vat_in_line_amounts:
            last = items[-1]
            amount = Decimal(last["amount"]) + difference
            last["amount"] = str(amount.quantize(CENT, rounding=ROUND_HALF_UP))
            last["price"] = str((amount / Decimal(last["quantity"])).quantize(UNIT_PRICE_STEP, rounding=ROUND_HALF_UP))
            selection = selections.get(last["row_id"], {})
            if last["match_status"] == "manual" and last["service"] and selection.get("price") is not None:
                last.update(transformed_numbers(amount, last["quantity"], selection["price"]))
    for row in items:
        source = source_rows[row['source_index']]
        # Service totals encoded as quantity × 1 (or 1.16) become one service.
        if ((row.get('service') or source.get('line_kind') == 'service')
                and Decimal(row['source_price']) in {Decimal('1'), Decimal('1.16')}
                and Decimal(row['source_quantity']) > 1):
            row.update(quantity='1', price=format(Decimal(row['amount']).quantize(CENT, rounding=ROUND_HALF_UP), '.2f'),
                       quantity_mode='service', service_quantity_collapsed=True)
            for target in row.get('approved_targets', []):
                if target.get('service'):
                    target.update(quantity='1', quantity_mode='service')
        for target in row.get("approved_targets", []):
            target["amount"] = row["amount"]
            target["price"] = str((Decimal(row["amount"]) / Decimal(target["quantity"])).quantize(UNIT_PRICE_STEP, rounding=ROUND_HALF_UP))
    warnings.extend(receipt.get("warnings", []))
    return {"items": items, "total": total, "vat_amount": receipt.get("vat_amount"), 'vat_in_line_amounts':vat_in_line_amounts,
            "warnings": list(dict.fromkeys(warnings)), "status": "review" if warnings else "applied"}


def invoice_document(receipt, generation):
    try:
        issuer = json.loads(ISSUER_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(
            f"Не найден файл реквизитов счёта: {ISSUER_PATH}. "
            "Скопируйте invoice_issuer.json в папку data проекта."
        ) from None
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Файл реквизитов счёта недоступен или повреждён: {ISSUER_PATH} ({error})") from None
    errors = list(generation["warnings"])
    try:
        date = datetime.strptime(receipt.get("date") or "", "%d.%m.%Y").strftime("%d.%m.%Y")
    except ValueError:
        date = receipt.get("date") or ""
        errors.append("Исправьте дату чека перед печатью счёта")
    if not receipt.get("order_number"):
        errors.append("Укажите номер заказа перед печатью счёта")
    document = {"number": f"OCR-{receipt.get('order_number') or ''}", "date": date,
            "issuer": issuer, "items": generation["items"], "total": generation["total"],
            "vat_amount": generation["vat_amount"], 'vat_in_line_amounts':generation.get('vat_in_line_amounts',True),
            'price_decimal_places':generation.get('price_decimal_places'), "errors": list(dict.fromkeys(errors)), "ready": not errors}
    document["print_form"] = print_fields(document)
    document["token"] = hashlib.sha256(json.dumps(document, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return document
