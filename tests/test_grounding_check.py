import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.grounding_check import check_grounding, extract_claims


def test_extract_percentage():
    claims = extract_claims("The model achieved a MAPE of 15.3%.")
    assert any(c["kind"] == "percent" and abs(c["value"] - 15.3) < 1e-6 for c in claims)


def test_extract_currency():
    claims = extract_claims("Estimated cost: $206,144.55.")
    assert any(c["kind"] == "currency" and abs(c["value"] - 206144.55) < 1e-2 for c in claims)


def test_trailing_sentence_period_is_not_part_of_the_number():
    claims = extract_claims("The holdout MASE was 0.632.")
    assert [c["raw"] for c in claims] == ["0.632"]


def test_markdown_list_markers_are_not_claims():
    claims = extract_claims("1. First point\n2) Second point\n- 3 models were compared")
    assert [c["raw"] for c in claims] == ["3"]


def test_grounded_number_matches_within_tolerance():
    result = check_grounding("The holdout MASE was 0.63.", {"mase": 0.632})
    assert result["grounded_ratio"] == 1.0


def test_ungrounded_number_is_flagged():
    result = check_grounding("The holdout MASE was 4.2.", {"mase": 0.632})
    flagged = [c for c in result["claims"] if not c["grounded"]]
    assert len(flagged) == 1


def test_no_numbers_is_not_reported_as_grounded():
    result = check_grounding("This report has no numbers in it.", {"mase": 0.5})
    assert result["total_claims"] == 0
    assert result["verifiable"] is False
    assert result["grounded_ratio"] == 0.0


def test_small_scale_hallucination_is_flagged():
    result = check_grounding(
        "The MASE was actually 0.15 this time, way better than reported.",
        {"best_ml_mase_holdout": 0.632},
    )
    flagged = [c for c in result["claims"] if not c["grounded"]]
    assert len(flagged) == 1


def test_matched_fact_key_identifies_which_fact_grounded_the_claim():
    result = check_grounding(
        "The holdout MASE was 0.63.",
        {"best_ml_mase_holdout": 0.632, "sarima_mase_holdout": 1.019},
    )
    assert result["claims"][0]["matched_fact_key"] == "best_ml_mase_holdout"


def test_matched_fact_key_is_none_when_ungrounded():
    result = check_grounding("The holdout MASE was 4.2.", {"mase": 0.632})
    assert result["claims"][0]["matched_fact_key"] is None


def test_small_scale_rounding_still_grounds():
    result = check_grounding("Recall came in around 0.86.", {"control_limit_recall": 0.858})
    assert result["grounded_ratio"] == 1.0


def test_percent_claim_matches_fraction_fact():
    result = check_grounding("Precision reached 86.7%.", {"control_limit_precision": 0.867})
    assert result["grounded_ratio"] == 1.0


def test_percent_claim_matches_percent_scale_fact():
    result = check_grounding("The holdout MAPE was 15.23%.", {"best_ml_mape_holdout": 15.23})
    assert result["grounded_ratio"] == 1.0


def test_currency_claim_only_matches_currency_facts():
    facts = {"best_ml_mape_holdout": 206144.0, "best_ml_estimated_cost_usd": 12.0}
    result = check_grounding("The estimated cost was $206,144.", facts)
    assert result["claims"][0]["grounded"] is False


def test_currency_claim_matches_currency_fact():
    result = check_grounding("The estimated cost was $206,144.", {"best_ml_estimated_cost_usd": 206144.55})
    assert result["grounded_ratio"] == 1.0


def test_plain_number_does_not_match_currency_fact():
    result = check_grounding("There were 206144 units.", {"best_ml_estimated_cost_usd": 206144.0})
    assert result["claims"][0]["grounded"] is False
