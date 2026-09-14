"""Reference solution for the judge-bias-correction task."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

DATA_DIR = Path("/app/data")
OUTPUT_PATH = Path("/app/output.json")

VALID_JUDGE_VERDICTS = {"win_left", "win_right", "tie"}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def drop_inconsistent_swap_groups(rows):
    """Every real matchup is judged twice, once in each left/right order,
    sharing a swap_id -- so the two rows in a swap_id group must have
    opposite (model_left, model_right) tuples. A logging defect corrupts a
    small number of groups by flipping one row's model labels, leaving both
    rows of that group with the SAME tuple instead. There's no way to tell
    which row (if either) is right, so both must be dropped. This is
    invisible to a per-column scan (every field is individually well-formed)
    -- only grouping by swap_id and checking the pairing surfaces it."""
    groups = {}
    for row in rows:
        groups.setdefault(row.get("swap_id"), []).append(row)
    bad_swap_ids = set()
    for swap_id, group in groups.items():
        tuples = {(r.get("model_left"), r.get("model_right")) for r in group}
        if len(group) != 2 or len(tuples) != 2:
            bad_swap_ids.add(swap_id)
    return [row for row in rows if row.get("swap_id") not in bad_swap_ids]


def primary_session_tag(rows):
    """judge_transcripts.jsonl mixes two session_tag values into one id
    namespace: the deployed production judge and a smaller, independently
    run audit sample with its own, different bias profile. Neither value is
    self-descriptive -- only relative volume distinguishes them, so the
    majority tag is the primary evaluation judge."""
    counts = {}
    for row in rows:
        tag = row.get("session_tag")
        counts[tag] = counts.get(tag, 0) + 1
    return max(counts, key=counts.get)


def clean_judge_rows(rows, primary_tag):
    """Keep only records from the primary evaluation judge (see
    primary_session_tag), dropping any with a malformed verdict or a
    missing/non-numeric token count on either side. Pooling both sessions
    into one fit blends two different position/verbosity coefficients into
    a meaningless average, so the primary-only records must be isolated
    before fitting, not just the malformed rows dropped. A small fraction
    of records also have logging defects (bad verdict string, missing
    token count) that must be excluded rather than coerced."""
    clean = []
    for row in rows:
        verdict = row.get("judge_verdict")
        left_tok = row.get("response_left_tokens")
        right_tok = row.get("response_right_tokens")
        if row.get("session_tag") != primary_tag:
            continue
        if verdict not in VALID_JUDGE_VERDICTS:
            continue
        if not isinstance(left_tok, (int, float)) or isinstance(left_tok, bool):
            continue
        if not isinstance(right_tok, (int, float)) or isinstance(right_tok, bool):
            continue
        clean.append(row)
    return clean


def fit_weighted_logit(X, y, w, ridge=1e-6):
    """Fit weighted binary logistic regression and return coefficients + SEs."""

    def objective(beta):
        z = X @ beta
        loss = np.sum(w * (np.logaddexp(0.0, z) - y * z))
        loss += 0.5 * ridge * float(beta @ beta)
        return float(loss)

    def gradient(beta):
        z = X @ beta
        # Numerically stable sigmoid.
        p = np.empty_like(z)
        positive = z >= 0
        p[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        ez = np.exp(z[~positive])
        p[~positive] = ez / (1.0 + ez)
        return X.T @ (w * (p - y)) + ridge * beta

    start = np.zeros(X.shape[1], dtype=float)
    result = minimize(
        objective,
        start,
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")

    beta = result.x
    z = X @ beta
    p = np.empty_like(z)
    positive = z >= 0
    p[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    p[~positive] = ez / (1.0 + ez)

    fisher_weights = w * p * (1.0 - p)
    hessian = X.T @ (X * fisher_weights[:, None]) + ridge * np.eye(X.shape[1])
    covariance = np.linalg.pinv(hessian)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return beta, standard_errors


def append_outcome(X_rows, y_rows, w_rows, features, verdict, left_label, right_label):
    """Encode wins normally and ties as two half-weight pseudo-observations."""
    if verdict == left_label:
        X_rows.append(features)
        y_rows.append(1.0)
        w_rows.append(1.0)
    elif verdict == right_label:
        X_rows.append(features)
        y_rows.append(0.0)
        w_rows.append(1.0)
    elif verdict == "tie":
        X_rows.extend((features.copy(), features.copy()))
        y_rows.extend((1.0, 0.0))
        w_rows.extend((0.5, 0.5))
    else:
        raise ValueError(f"unexpected verdict: {verdict!r}")


def build_judge_design(rows, model_names, reference_model, length_mean, length_sd):
    others = [m for m in model_names if m != reference_model]
    skill_col = {m: i for i, m in enumerate(others)}

    # Columns: left-position intercept, z-scored token difference, skill dummies.
    n_extra = 2
    n_columns = n_extra + len(others)
    X_rows, y_rows, w_rows = [], [], []

    for row in rows:
        features = np.zeros(n_columns, dtype=float)
        features[0] = 1.0
        left_z = (float(row["response_left_tokens"]) - length_mean) / length_sd
        right_z = (float(row["response_right_tokens"]) - length_mean) / length_sd
        features[1] = left_z - right_z

        left_model = row["model_left"]
        right_model = row["model_right"]
        if left_model in skill_col:
            features[n_extra + skill_col[left_model]] += 1.0
        if right_model in skill_col:
            features[n_extra + skill_col[right_model]] -= 1.0

        append_outcome(
            X_rows,
            y_rows,
            w_rows,
            features,
            row["judge_verdict"],
            "win_left",
            "win_right",
        )

    return (
        np.asarray(X_rows, dtype=float),
        np.asarray(y_rows, dtype=float),
        np.asarray(w_rows, dtype=float),
        others,
    )


def build_human_design(rows, model_names, reference_model):
    others = [m for m in model_names if m != reference_model]
    skill_col = {m: i for i, m in enumerate(others)}

    # Include an intercept as a calibration sanity term.
    n_columns = 1 + len(others)
    X_rows, y_rows, w_rows = [], [], []

    for row in rows:
        features = np.zeros(n_columns, dtype=float)
        features[0] = 1.0

        model_a = row["model_a"]
        model_b = row["model_b"]
        if model_a in skill_col:
            features[1 + skill_col[model_a]] += 1.0
        if model_b in skill_col:
            features[1 + skill_col[model_b]] -= 1.0

        append_outcome(
            X_rows,
            y_rows,
            w_rows,
            features,
            row["human_verdict"],
            "win_a",
            "win_b",
        )

    return (
        np.asarray(X_rows, dtype=float),
        np.asarray(y_rows, dtype=float),
        np.asarray(w_rows, dtype=float),
        others,
    )


def main():
    metadata = load_json(DATA_DIR / "models_metadata.json")
    judge_rows_raw = load_jsonl(DATA_DIR / "judge_transcripts.jsonl")
    human_rows = load_jsonl(DATA_DIR / "human_calibration.jsonl")
    judge_rows_raw = drop_inconsistent_swap_groups(judge_rows_raw)
    primary_tag = primary_session_tag(judge_rows_raw)
    judge_rows = clean_judge_rows(judge_rows_raw, primary_tag)

    model_names = [entry["name"] for entry in metadata["models"]]
    vendors = {entry["name"]: entry["vendor"] for entry in metadata["models"]}
    judge_vendor = metadata["judge_vendor"]

    length_mean = float(metadata["length_normalization"]["mean"])
    length_sd = float(metadata["length_normalization"]["sd"])
    if length_sd <= 0:
        raise ValueError("length_normalization.sd must be positive")

    # Any model can be the pinned baseline for identifiability, EXCEPT one
    # sharing the judge's vendor: pinning a same-vendor model's skill to 0
    # would silently drop it from (or bias) the self-enhancement candidate
    # comparison below, since that model could no longer show a nonzero
    # judge-vs-human skill gap relative to itself.
    non_judge_vendor_models = [m for m in model_names if vendors[m] != judge_vendor]
    reference_model = sorted(non_judge_vendor_models)[-1]

    Xj, yj, wj, judge_others = build_judge_design(
        judge_rows, model_names, reference_model, length_mean, length_sd
    )
    judge_beta, judge_se = fit_weighted_logit(Xj, yj, wj)

    position_bias = float(judge_beta[0])
    verbosity_coef = float(judge_beta[1])

    judge_skill = {reference_model: 0.0}
    judge_skill_se = {reference_model: 0.0}
    for i, model in enumerate(judge_others):
        judge_skill[model] = float(judge_beta[2 + i])
        judge_skill_se[model] = float(judge_se[2 + i])

    Xh, yh, wh, human_others = build_human_design(human_rows, model_names, reference_model)
    human_beta, human_se = fit_weighted_logit(Xh, yh, wh)

    human_skill = {reference_model: 0.0}
    human_skill_se = {reference_model: 0.0}
    for i, model in enumerate(human_others):
        human_skill[model] = float(human_beta[1 + i])
        human_skill_se[model] = float(human_se[1 + i])

    # Compare judge-implied skill with the independent human-calibrated skill
    # only among models from the judge's vendor. A large positive standardized
    # gap is evidence of an unexplained judge-side preference.
    candidates = [
        model
        for model in model_names
        if vendors[model] == judge_vendor and model != reference_model
    ]

    best_model = None
    best_z = float("-inf")
    for model in candidates:
        gap = judge_skill[model] - human_skill[model]
        gap_se = float(
            np.sqrt(judge_skill_se[model] ** 2 + human_skill_se[model] ** 2)
        )
        z_score = gap / gap_se if gap_se > 0 else float("-inf")
        if z_score > best_z:
            best_z = z_score
            best_model = model

    self_enhancement_model = best_model if best_z > 3.0 else None
    ranking = sorted(model_names, key=lambda model: human_skill[model], reverse=True)

    output = {
        "recovered_ranking_best_to_worst": ranking,
        "position_bias_estimate": round(position_bias, 4),
        "verbosity_coef_estimate": round(verbosity_coef, 4),
        "self_enhancement_model": self_enhancement_model,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
