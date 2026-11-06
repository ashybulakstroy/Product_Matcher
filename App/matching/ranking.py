"""Price tie-breaking, stock partitioning and result presentation data."""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from App.matching.contracts import extract_query


def _price(value):
    """Nonnegative finite unit price; missing/invalid catalog prices are unknown."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).replace("\u00a0", "").replace(" ", "").replace(",", "."))
    except InvalidOperation:
        return None
    return number if number.is_finite() and number >= 0 else None


def _item_price(item):
    for key in ("price", "unit_price", "accounting_price"):
        value = _price(item.get(key))
        if value is not None:
            return value
    return None


def rank_result(items, checked_products, stock_products, spec, top_n=5, min_stock=0):
    algorithm_id = spec.id
    stock_by_id = {}
    for stock_item in stock_products:
        stock_by_id.setdefault(str(stock_item.get("id") or stock_item.get("sku") or ""), stock_item)
    full_candidate_lists = []
    for entry, checked in zip(items, checked_products):
        query_price = _item_price(checked)
        entry["checked_price"] = float(query_price) if query_price is not None else None
        for candidate in entry["candidates"]:
            stock_match = stock_by_id.get(str(candidate["stock_id"]))
            stock_price = _item_price(stock_match or {})
            candidate["stock_price"] = float(stock_price) if stock_price is not None else None
            difference = abs(stock_price - query_price) if stock_price is not None and query_price is not None else None
            candidate["price_difference"] = float(difference) if difference is not None else None
            candidate["stock_price_display"] = str(stock_price.quantize(Decimal("1"), rounding=ROUND_HALF_UP)) if stock_price is not None else "—"
            percent = difference / query_price * 100 if difference is not None and query_price else (Decimal(0) if difference == 0 else None)
            candidate["price_difference_pct"] = float(percent) if percent is not None else None
            candidate["price_difference_display"] = str(percent.quantize(Decimal("1"), rounding=ROUND_HALF_UP)) + "%" if percent is not None else "—"
            if stock_match is not None:
                candidate["stock_qty"] = (
                    stock_match.get("quantity")
                    or stock_match.get("stock_quantity")
                    or stock_match.get("qty")
                )

        # Price only breaks ties in the displayed name-similarity score.
        entry["candidates"].sort(key=lambda candidate: (
            -candidate["similarity_pct"],
            candidate["price_difference"] is None if query_price is not None else False,
            abs(_item_price(stock_by_id.get(str(candidate["stock_id"]), {})) - query_price)
                if query_price is not None and candidate["price_difference"] is not None else Decimal(0),
            str(candidate["stock_id"]),
        ))
        all_candidates = list(entry["candidates"])
        if min_stock:
            full_main_candidates = [
                candidate
                for candidate in all_candidates
                if float(candidate.get("stock_qty") or 0) >= float(min_stock)
            ]
        else:
            full_main_candidates = list(all_candidates)

        main_candidates = full_main_candidates[:top_n]

        cutoff_score = None
        if main_candidates:
            cutoff_score = min(candidate.get("similarity_pct", 0) for candidate in main_candidates)

        excluded_candidates = []
        for candidate in all_candidates:
            qty = float(candidate.get("stock_qty") or 0)
            if qty < float(min_stock) if min_stock else False:
                if cutoff_score is None or float(candidate.get("similarity_pct") or 0) >= float(cutoff_score):
                    excluded_candidates.append(candidate)

        full_excluded_candidates = list(excluded_candidates)
        entry["candidates"] = main_candidates
        entry["excluded_candidates"] = full_excluded_candidates[:top_n]

        entry["algorithm_id"] = algorithm_id
        entry["algorithm_name"] = spec.name
        entry["query"] = extract_query(checked_products)
        entry["best_candidate"] = main_candidates[0] if main_candidates else None
        entry["low_stock_candidate"] = excluded_candidates[0] if excluded_candidates else None
        full_candidate_lists.append((full_main_candidates, full_excluded_candidates))

    first_best = None
    for entry in items:
        if entry.get("best_candidate"):
            first_best = entry["best_candidate"]
            break

    query_value = extract_query(checked_products)
    result = {
        "status": "ok",
        "top_n": top_n,
        "algorithm_id": algorithm_id,
        "algorithm_name": spec.name,
        "algorithm_title": spec.title,
        "description": spec.description,
        "query": query_value,
        "items": items,
        "best_candidate": first_best,
        "low_stock_candidate": next((candidate for entry in items for candidate in entry.get("excluded_candidates", []) if candidate), None),
    }
    full_items = []
    for entry, (full_main, full_excluded) in zip(items, full_candidate_lists):
        full_entry = dict(entry)
        full_entry["candidates"] = full_main
        full_entry["excluded_candidates"] = full_excluded
        full_items.append(full_entry)

    full_result = {**result, "items": full_items}
    return result, full_result
