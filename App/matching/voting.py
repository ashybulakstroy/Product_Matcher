"""Aggregate *already ranked* Product Matcher results into one vote per algorithm.

No OCR, catalog search, data loading, I/O or score calibration is performed here.
Designed for App.matching.engine.match() -> results_by_algorithm.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any
from App.matching.ranking import _item_price, price_distance


def _text_identifier(value: Any) -> str:
    """Keep 1C IDs as strings: '00000123' MUST NOT become '123'."""
    if value is None or isinstance(value, bool):
        return ""
    return str(value).strip()


def _query_identity(checked: Mapping[str, Any]) -> tuple[str, str]:
    checked_id = _text_identifier(checked.get("id") if checked.get("id") is not None else checked.get("sku"))
    checked_name = str(
        checked.get("name") or checked.get("product_name") or checked.get("checked_name") or ""
    )
    return checked_id, checked_name


def _position_rows(result: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    """Use position numbers, not names, so repeated names never merge."""
    by_position: dict[int, Mapping[str, Any]] = {}
    items = result.get("items")
    if not isinstance(items, (list, tuple)):
        return by_position
    for index, row in enumerate(items, start=1):
        if not isinstance(row, Mapping):
            continue
        position = row.get("position_no", index)
        # bool is an int in Python; don't treat true as position 1.
        if not isinstance(position, int) or isinstance(position, bool) or position <= 0:
            continue
        by_position.setdefault(position, row)
    return by_position


def _eligible_candidate(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Take *ranked/stock-filtered* best_candidate only; never low_stock_candidate."""
    candidate = row.get("best_candidate")
    return candidate if isinstance(candidate, Mapping) else None


def aggregate_votes(
    checked_products: Sequence[Mapping[str, Any]],
    results_by_algorithm: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build vote tables for each checked product from engine's completed results.

    Contract:
      - one *successful* algorithm casts <= 1 vote for each Torgsoft position;
      - candidates are identical iff their exact string stock_id matches;
      - only the already stock-filtered items[*].best_candidate participates;
      - per-algorithm percentages are retained, NEVER averaged;
      - equal votes use price when present; equal votes and prices remain a tie;
      - selected_stock_id=None until the user explicitly makes a choice.

    No mutation of caller-owned inputs or result objects occurs.
    """
    started = perf_counter()
    checked = list(checked_products)
    # Defensive de-duplication in case an API accidentally repeats algorithm IDs.
    algorithms: list[tuple[str, dict[int, Mapping[str, Any]]]] = []
    seen_algorithm_ids: set[str] = set()
    for result in results_by_algorithm:
        if not isinstance(result, Mapping):
            continue
        algorithm_id = _text_identifier(result.get("algorithm_id"))
        if not algorithm_id or algorithm_id in seen_algorithm_ids:
            continue
        seen_algorithm_ids.add(algorithm_id)
        if result.get("status") != "ok":
            continue
        algorithms.append((algorithm_id, _position_rows(result)))

    positions: list[dict[str, Any]] = []
    for position_no, query in enumerate(checked, start=1):
        if not isinstance(query, Mapping):
            query = {}
        checked_id, checked_name = _query_identity(query)
        by_product_id: dict[str, dict[str, Any]] = {}
        voted_algorithms = 0

        for algorithm_id, rows in algorithms:
            row = rows.get(position_no)
            if row is None:
                continue
            candidate = _eligible_candidate(row)
            if candidate is None:
                continue
            stock_id = _text_identifier(candidate.get("stock_id"))
            if not stock_id:
                continue
            voted_algorithms += 1
            if stock_id not in by_product_id:
                by_product_id[stock_id] = {
                    "stock_id": stock_id,
                    "stock_name": str(candidate.get("stock_name") or ""),
                    "stock_qty": candidate.get("stock_qty"),
                    "stock_price": candidate.get("stock_price"),
                    "stock_price_display": candidate.get("stock_price_display"),
                    "price_difference_display": candidate.get("price_difference_display"),
                    "price_difference_pct": candidate.get("price_difference_pct"),
                    "votes": 0,
                    "voters": [],
                    "scores_by_algorithm": {},
                    "reasons_by_algorithm": {},
                }
            item = by_product_id[stock_id]
            item["votes"] += 1
            item["voters"].append(algorithm_id)
            # No cross-algorithm score aggregation: the scales are not calibrated.
            item["scores_by_algorithm"][algorithm_id] = candidate.get("similarity_pct")
            if candidate.get("match_reasons"):
                item["reasons_by_algorithm"][algorithm_id] = list(candidate["match_reasons"])
            # Fields may be absent in one algorithm's candidate but present in another.
            for key in ("stock_name", "stock_qty", "stock_price", "stock_price_display",
                        "price_difference_display", "price_difference_pct"):
                if item[key] in (None, "") and candidate.get(key) not in (None, ""):
                    item[key] = candidate[key]

        query_price = _item_price(query)
        def price_key(candidate):
            distance = price_distance(candidate.get('stock_price'), query_price)
            return (distance is None, distance if distance is not None else 0)
        candidates = sorted(by_product_id.values(), key=lambda c: (-c["votes"], price_key(c), c["stock_id"]))
        current_rank = 0
        previous_votes: int | None = None
        for index, candidate in enumerate(candidates, start=1):
            if candidate["votes"] != previous_votes:
                current_rank = index
                previous_votes = candidate["votes"]
            candidate["rank"] = current_rank
            candidate["vote_pct"] = round(100.0 * candidate["votes"] / voted_algorithms, 2) if voted_algorithms else 0.0

        top_votes = candidates[0]["votes"] if candidates else 0
        leader_ids = [c["stock_id"] for c in candidates if c["votes"] == top_votes]
        if query_price is not None and leader_ids:
            best_price = price_key(candidates[0])
            leader_ids = [c['stock_id'] for c in candidates if c['votes'] == top_votes and price_key(c) == best_price]
        is_tie = len(leader_ids) > 1
        positions.append({
            "position_no": position_no,
            "checked_id": checked_id,
            "checked_name": checked_name,
            "voting_algorithms": voted_algorithms,
            "unique_candidates": len(candidates),
            "leader_count": len(leader_ids),
            "recommended_stock_id": leader_ids[0] if len(leader_ids) == 1 else None,
            "selected_stock_id": None,
            "is_tie": is_tie,
            "leader_stock_ids": leader_ids,
            "candidates": candidates,
        })
    return {"positions": positions, "elapsed_ms": round((perf_counter() - started) * 1000, 3)}


__all__ = ["aggregate_votes"]
