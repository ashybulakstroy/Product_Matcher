"""Synthetic-only tests. Run from the repository root with pytest -q App/tests/test_voting.py."""

from copy import deepcopy

from App.matching.voting import aggregate_votes


CHECKED = [{"id": "TS-1", "name": "Саморез кровельный 4,8х35"}]


def run(algorithm_id, proposals, *, status="ok"):
    """proposals: list of (position number, stock id, name, score, quantity)."""
    rows = []
    for position, stock_id, name, score, quantity in proposals:
        candidate = (
            None if stock_id is None else {
                "stock_id": stock_id,
                "stock_name": name,
                "similarity_pct": score,
                "stock_qty": quantity,
                "stock_price": 35,
            }
        )
        rows.append({"position_no": position, "best_candidate": candidate,
                     "low_stock_candidate": {"stock_id": "unavailable"}})
    return {"algorithm_id": algorithm_id, "status": status, "items": rows,
            "best_candidate": rows[0]["best_candidate"] if rows else None}


def test_three_votes_one_product():
    results = [run(f"a{i}", [(1, "00000123", "Саморез", 92 - i, 120)]) for i in range(3)]
    vote = aggregate_votes(CHECKED, results)["positions"][0]
    assert vote["voting_algorithms"] == 3
    assert vote["recommended_stock_id"] == "00000123"
    assert vote["candidates"][0]["votes"] == 3
    assert vote["candidates"][0]["vote_pct"] == 100.0
    assert vote["selected_stock_id"] is None
    assert vote["candidates"][0]["scores_by_algorithm"] == {"a0": 92, "a1": 91, "a2": 90}


def test_all_different():
    results = [run(f"a{i}", [(1, str(i), str(i), 99, 1)]) for i in range(3)]
    vote = aggregate_votes(CHECKED, results)["positions"][0]
    assert vote["is_tie"] is True
    assert vote["recommended_stock_id"] is None
    assert vote["leader_stock_ids"] == ["0", "1", "2"]
    assert [row["vote_pct"] for row in vote["candidates"]] == [33.33] * 3


def test_two_way_tie_and_deterministic_order():
    votes = ["b", "a", "b", "a", "c"]
    results = [run(f"alg{i}", [(1, target, target, 10, 3)]) for i, target in enumerate(votes)]
    result = aggregate_votes(CHECKED, results)["positions"][0]
    assert [(c["stock_id"], c["votes"], c["rank"]) for c in result["candidates"]] == [
        ("a", 2, 1), ("b", 2, 1), ("c", 1, 3)
    ]
    assert result["leader_stock_ids"] == ["a", "b"]


def test_error_and_unavailable_do_not_vote():
    results = [run("a", [(1, "1", "first", 50, 1)]),
               run("b", [(1, "2", "second", 99, 1)], status="error"),
               run("c", [(1, "3", "third", 99, 1)], status="unavailable")]
    vote = aggregate_votes(CHECKED, results)["positions"][0]
    assert vote["voting_algorithms"] == 1
    assert vote["recommended_stock_id"] == "1"


def test_no_candidate_exact_and_zero_division():
    result = aggregate_votes(CHECKED, [run("exact", [(1, None, "", 0, 1)])])["positions"][0]
    assert result["voting_algorithms"] == 0
    assert result["candidates"] == []
    assert result["leader_stock_ids"] == []
    assert result["recommended_stock_id"] is None
    assert result["is_tie"] is False


def test_no_algorithms():
    result = aggregate_votes(CHECKED, [])["positions"][0]
    assert result["voting_algorithms"] == 0
    assert result["unique_candidates"] == 0


def test_different_scores_preserved_not_averaged():
    vote = aggregate_votes(CHECKED, [run("one", [(1, "x", "x", 30.0, 3)]),
                                     run("two", [(1, "x", "x", 95.3, 3)])])["positions"][0]
    candidate = vote["candidates"][0]
    assert candidate["scores_by_algorithm"] == {"one": 30.0, "two": 95.3}
    assert "average" not in candidate


def test_multiple_positions_do_not_mix_votes_even_with_same_name():
    checked = [{"id": "T1", "name": "Болт"}, {"id": "T2", "name": "Болт"}]
    result = aggregate_votes(checked, [
        run("a", [(2, "0002", "B", 80, 3), (1, "0001", "A", 80, 3)]),
        run("b", [(1, "0001", "A", 90, 3), (2, "0003", "C", 90, 3)]),
    ])["positions"]
    assert result[0]["recommended_stock_id"] == "0001"
    assert result[0]["candidates"][0]["votes"] == 2
    assert result[1]["is_tie"] is True
    assert [c["stock_id"] for c in result[1]["candidates"]] == ["0002", "0003"]


def test_low_stock_candidate_never_votes():
    result = run("a", [(1, None, "", 0, 0)])
    result["items"][0]["low_stock_candidate"] = {"stock_id": "00000009"}
    assert aggregate_votes(CHECKED, [result])["positions"][0]["voting_algorithms"] == 0


def test_string_ids_preserve_leading_zeroes_and_distinguish():
    results = [run("a", [(1, "00000001", "A", 80, 3)]), run("b", [(1, "1", "B", 80, 3)])]
    vote = aggregate_votes(CHECKED, results)["positions"][0]
    assert vote["unique_candidates"] == 2
    assert vote["leader_stock_ids"] == ["00000001", "1"]


def test_parallel_and_sequential_result_order_not_relevant_for_count():
    a = run("a", [(1, "X", "X", 80, 3)])
    b = run("b", [(1, "X", "X", 82, 3)])
    c = run("c", [(1, "Y", "Y", 99, 3)])
    one = aggregate_votes(CHECKED, [a, b, c])["positions"][0]
    two = aggregate_votes(CHECKED, [c, a, b])["positions"][0]
    assert one["recommended_stock_id"] == two["recommended_stock_id"] == "X"
    assert {c["stock_id"]: c["votes"] for c in one["candidates"]} == {
        c["stock_id"]: c["votes"] for c in two["candidates"]}


def test_duplicate_algorithm_id_is_only_one_vote():
    first = run("same", [(1, "a", "A", 80, 3)])
    repeated = run("same", [(1, "b", "B", 90, 3)])
    vote = aggregate_votes(CHECKED, [first, repeated])["positions"][0]
    assert vote["voting_algorithms"] == 1
    assert vote["recommended_stock_id"] == "a"


def test_input_is_not_mutated():
    results = [run("one", [(1, "001", "A", 50, 3)])]
    original = deepcopy(results)
    aggregate_votes(CHECKED, results)
    assert results == original


def test_empty_and_missing_position_do_not_fall_back_to_global_best():
    item = run("single", [(1, "001", "A", 50, 3)])
    checked = [{"id": "x", "name": "a"}, {"id": "y", "name": "b"}]
    votes = aggregate_votes(checked, [item])["positions"]
    assert votes[0]["recommended_stock_id"] == "001"
    assert votes[1]["voting_algorithms"] == 0


def test_fifth_added_algorithm_works_without_changes_to_aggregator():
    results = [run(f"algorithm_{i}", [(1, "0001", "A", 60, 1)]) for i in range(5)]
    assert aggregate_votes(CHECKED, results)["positions"][0]["voting_algorithms"] == 5
