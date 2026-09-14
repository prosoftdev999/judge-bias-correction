"""
Sealed verifier for judge-bias-correction.

Grades the agent's /app/output.json against the ground-truth generator
parameters baked into this image at build time (tests/true_params.json,
never shipped to the agent's environment image). Every check reads the
agent's declared numbers/labels -- none of it accepts a bare exit code or
agent-asserted claim as evidence.
"""
import json
import math

import pytest

OUTPUT_PATH = "/app/output.json"
TRUE_PATH = "/tests/true_params.json"

TRUE = json.load(open(TRUE_PATH))
MODEL_NAMES = set(TRUE["true_skill"].keys())

POSITION_BIAS_TOL = 0.08       # calibrated: ~3 fitted SE; rejects tie-as-left misspecification (+0.30 off)
VERBOSITY_COEF_TOL = 0.07      # calibrated: ~3.5 fitted SE; covers legit tie-drop refits (+0.037 off)
SPEARMAN_MIN = 0.97            # ~= at most one adjacent-pair swap out of 9 (tightly packed skills)


def spearman_rho(rank_a, rank_b):
    """rank_a, rank_b: lists of the same items, best-to-worst order."""
    pos_b = {item: i for i, item in enumerate(rank_b)}
    n = len(rank_a)
    d2 = sum((i - pos_b[item]) ** 2 for i, item in enumerate(rank_a))
    return 1 - 6 * d2 / (n * (n ** 2 - 1))


@pytest.fixture(scope="module")
def output():
    try:
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
    except FileNotFoundError:
        pytest.fail(f"{OUTPUT_PATH} does not exist -- no artifact was produced")
    except json.JSONDecodeError as e:
        pytest.fail(f"{OUTPUT_PATH} is not valid JSON: {e}")
    return data


def test_output_has_required_keys(output):
    required = {
        "recovered_ranking_best_to_worst",
        "position_bias_estimate",
        "verbosity_coef_estimate",
        "self_enhancement_model",
    }
    missing = required - set(output.keys())
    assert not missing, f"output.json missing required keys: {missing}"


def test_ranking_is_well_formed(output):
    ranking = output["recovered_ranking_best_to_worst"]
    assert isinstance(ranking, list), "recovered_ranking_best_to_worst must be a list"
    assert len(ranking) == len(MODEL_NAMES), (
        f"expected {len(MODEL_NAMES)} models in the ranking, got {len(ranking)}"
    )
    assert set(ranking) == MODEL_NAMES, (
        f"ranking must contain exactly the {len(MODEL_NAMES)} candidate models; "
        f"got {set(ranking)} vs expected {MODEL_NAMES}"
    )


def test_ranking_recovers_true_skill_order(output):
    ranking = output["recovered_ranking_best_to_worst"]
    assert set(ranking) == MODEL_NAMES, "ranking must be well-formed (see other failure)"
    true_ranking = TRUE["true_ranking_best_to_worst"]
    rho = spearman_rho(ranking, true_ranking)
    assert rho >= SPEARMAN_MIN, (
        f"recovered ranking {ranking} does not match the bias-free true skill "
        f"order {true_ranking} closely enough (spearman rho={rho:.3f} < {SPEARMAN_MIN}). "
        f"A naive, unadjusted analysis of the judge transcripts alone produces a "
        f"noticeably different (wrong) order -- position/verbosity/self-enhancement "
        f"bias must actually be removed, not just acknowledged."
    )


def test_position_bias_estimate(output):
    val = output["position_bias_estimate"]
    assert isinstance(val, (int, float)) and not isinstance(val, bool), (
        "position_bias_estimate must be numeric"
    )
    assert math.isfinite(val), "position_bias_estimate must be finite"
    true_val = TRUE["position_bias"]
    assert abs(val - true_val) <= POSITION_BIAS_TOL, (
        f"position_bias_estimate={val} not within {POSITION_BIAS_TOL} of true value {true_val}"
    )


def test_verbosity_coef_estimate(output):
    val = output["verbosity_coef_estimate"]
    assert isinstance(val, (int, float)) and not isinstance(val, bool), (
        "verbosity_coef_estimate must be numeric"
    )
    assert math.isfinite(val), "verbosity_coef_estimate must be finite"
    true_val = TRUE["verbosity_coef"]
    assert abs(val - true_val) <= VERBOSITY_COEF_TOL, (
        f"verbosity_coef_estimate={val} not within {VERBOSITY_COEF_TOL} of true value {true_val}"
    )


def test_self_enhancement_model_identified(output):
    val = output["self_enhancement_model"]
    true_val = TRUE["self_enhancement_model"]
    assert val == true_val, (
        f"self_enhancement_model={val!r}, expected {true_val!r}. "
        f"Vendor-match alone is not enough evidence -- "
        f"{TRUE['distractor_same_vendor_models']!r} both share the judge's vendor "
        f"too but are NOT self-enhanced, and reporting either is a distinct wrong answer."
    )
