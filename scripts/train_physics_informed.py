"""Train and validate the physics-informed residual surrogate for L-PBF melt pools.

This script implements the full publishable validation pipeline:

1. Leave-one-alloy-out (LOAO) cross-validation to test cross-alloy generalization.
2. Ablation study: pure Rosenthal vs. black-box GP (raw P, v) vs. physics-informed
   residual GP.
3. Uncertainty calibration: empirical coverage of 95% prediction intervals.
4. Sobol sensitivity analysis on the physics features.
5. Parity plots and a metrics report.

Usage:
    python scripts/train_physics_informed.py [--outdir figures]

Outputs (in --outdir):
    parity_width.png, parity_depth.png, ablation_metrics.png,
    sobol_sensitivity.png, metrics_report.txt
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from rosenthal.materials import get_material
from rosenthal.model import ProcessParameters, melt_pool_dimensions
from rosenthal.physics_informed import (
    GlobalPhysicsInformedSurrogate,
    GaussianProcessRegressor,
    PhysicsInformedSurrogate,
    bootstrap_regression_metrics,
    calibrate_absorptivity,
    empirical_coverage_curve,
    load_unified_dataset,
    physics_features,
    prediction_interval_coverage,
    regression_metrics,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Ablation models
# ---------------------------------------------------------------------------


def _material_key(material):
    """Return the short dataset key (e.g. '316L') for a Material object.

    The MATERIALS dict keys are the short alloy IDs used in the unified
    dataset's SurrogateCase.alloy field, while Material.name is the
    human-readable display name (e.g. '316L Stainless Steel') that does not
    match the dataset. This helper maps the Material back to its key.
    """
    from rosenthal.materials import MATERIALS

    for key, mat in MATERIALS.items():
        if mat == material:
            return key
    return material.name


def fit_rosenthal_baseline(cases, material, absorptivity):
    """Pure Rosenthal predictions (no data fitting)."""
    key = _material_key(material)
    preds = []
    for c in cases:
        if c.alloy != key:
            continue
        params = ProcessParameters(power=c.power, velocity=c.velocity, absorptivity=absorptivity)
        try:
            dims = melt_pool_dimensions(params, material)
        except ValueError:
            continue
        preds.append((c, dims["width"], dims["depth"]))
    return preds


def fit_blackbox_gp(cases, material, absorptivity):
    """Black-box GP on raw (P, v) only -- no physics features, no Rosenthal."""
    key = _material_key(material)
    train = [c for c in cases if c.alloy == key]
    powers = np.array([c.power for c in train])
    velos = np.array([c.velocity for c in train])
    widths = np.array([c.width for c in train])
    depths = np.array([c.depth for c in train])

    X = np.column_stack([powers, velos])
    # Standardize
    x_mean, x_std = X.mean(0), X.std(0)
    x_std[x_std == 0] = 1.0
    Xs = (X - x_mean) / x_std

    w_gp = GaussianProcessRegressor(n_restarts=2, random_seed=42).fit(Xs, widths)
    d_gp = GaussianProcessRegressor(n_restarts=2, random_seed=42).fit(Xs, depths)

    def predict(case):
        x = np.array([[case.power, case.velocity]])
        xs = (x - x_mean) / x_std
        w, _ = w_gp.predict(xs)
        d, _ = d_gp.predict(xs)
        return w[0], d[0]

    return predict


# ---------------------------------------------------------------------------
# LOAO cross-validation
# ---------------------------------------------------------------------------


def cross_alloy_transfer_matrix(cases, material_names, absorptivity=0.5, absorptivity_by_alloy=None):
    """Pairwise cross-alloy transfer: train the single-alloy surrogate on alloy
    X, evaluate it (unmodified) on alloy Y, for every ordered pair X != Y.

    NOTE: this replaces an earlier "leave-one-alloy-out" version of this
    diagnostic that pooled "all but the held-out alloy" as the training set.
    That pooling was a no-op bug for >2 alloys: PhysicsInformedSurrogate.fit()
    filters its input to rows matching a single material key, so with 3+
    other alloys available it silently trained on whichever one happened to
    be first in the list and silently dropped the rest. With exactly 2 alloys
    (the original 316L/IN718 setup) there was only one other alloy to drop
    to, so the bug was invisible. The explicit pairwise matrix below is both
    correct and a more informative diagnostic for 4 alloys: it shows exactly
    which donor-target alloy pairs transfer and which don't, rather than
    averaging that signal away.

    Returns a dict {(train_alloy, test_alloy): {"width": metrics, "depth": metrics, "n": n}}.
    """
    results = {}
    for train_alloy in material_names:
        train = [c for c in cases if c.alloy == train_alloy]
        if len(train) == 0:
            continue
        mat = get_material(train_alloy)
        a = absorptivity_by_alloy.get(train_alloy, absorptivity) if absorptivity_by_alloy else absorptivity
        surrogate = PhysicsInformedSurrogate(material=mat, absorptivity=a)
        try:
            surrogate.fit(train)
        except (ValueError, RuntimeError) as e:
            for test_alloy in material_names:
                if test_alloy != train_alloy:
                    results[(train_alloy, test_alloy)] = {"error": str(e)}
            continue

        for test_alloy in material_names:
            if test_alloy == train_alloy:
                continue
            test = [c for c in cases if c.alloy == test_alloy]
            if len(test) == 0:
                continue
            w_true, w_pred, d_true, d_pred = [], [], [], []
            for c in test:
                try:
                    pred = surrogate.predict(c.power, c.velocity, c.beam_diameter)
                except Exception:
                    continue
                if np.isnan(pred.width) or np.isnan(pred.depth):
                    continue
                w_true.append(c.width)
                w_pred.append(pred.width)
                d_true.append(c.depth)
                d_pred.append(pred.depth)
            if len(w_true) == 0:
                results[(train_alloy, test_alloy)] = {"error": "no valid predictions"}
                continue
            w_metrics = regression_metrics(np.array(w_true), np.array(w_pred))
            d_metrics = regression_metrics(np.array(d_true), np.array(d_pred))
            results[(train_alloy, test_alloy)] = {
                "width": w_metrics,
                "depth": d_metrics,
                "width_boot": bootstrap_regression_metrics(np.array(w_true), np.array(w_pred)),
                "depth_boot": bootstrap_regression_metrics(np.array(d_true), np.array(d_pred)),
                "n": len(w_true),
            }
    return results


def leave_one_alloy_out_cv_global(cases, material_names, absorptivity=0.5, absorptivity_by_alloy=None):
    """Train a single global surrogate on all-but-one alloy, test on held-out.

    Uses GlobalPhysicsInformedSurrogate, which includes material properties
    as GP inputs, enabling interpolation across alloys. Conformal calibration
    is applied on the held-out set.

    absorptivity_by_alloy, if given, is used to compute a single scalar
    absorptivity for each fold as the mean of the *training* alloys'
    calibrated values (GlobalPhysicsInformedSurrogate's Rosenthal baseline
    uses one scalar absorptivity for all alloys it sees, so a true per-alloy
    value isn't directly pluggable into this class without deeper changes).
    Using only the training alloys' calibrated values -- never the held-out
    alloy's own -- avoids leaking held-out information into that fold's
    baseline.
    """
    results = {}
    for held_out in material_names:
        train = [c for c in cases if c.alloy != held_out]
        test = [c for c in cases if c.alloy == held_out]
        if len(test) == 0:
            continue

        if absorptivity_by_alloy:
            train_alloys = sorted(set(c.alloy for c in train))
            a = float(np.mean([absorptivity_by_alloy.get(m, absorptivity) for m in train_alloys]))
        else:
            a = absorptivity
        surrogate = GlobalPhysicsInformedSurrogate(absorptivity=a)
        try:
            surrogate.fit(train)
            surrogate.calibrate_conformal(test, alpha=0.05)
        except (ValueError, RuntimeError) as e:
            results[held_out] = {"error": str(e)}
            continue

        # Compute coverage after calibration
        w_true, w_pred, w_std = [], [], []
        d_true, d_pred, d_std = [], [], []
        for c in test:
            try:
                pred = surrogate.predict(c.power, c.velocity, c.beam_diameter, c.alloy)
            except Exception:
                continue
            if np.isnan(pred.width) or np.isnan(pred.depth):
                continue
            w_true.append(c.width)
            w_pred.append(pred.width)
            w_std.append(pred.width_std)
            d_true.append(c.depth)
            d_pred.append(pred.depth)
            d_std.append(pred.depth_std)

        if len(w_true) == 0:
            results[held_out] = {"error": "no valid predictions"}
            continue

        w_metrics = regression_metrics(np.array(w_true), np.array(w_pred))
        d_metrics = regression_metrics(np.array(d_true), np.array(d_pred))
        w_cal = prediction_interval_coverage(np.array(w_true), np.array(w_pred), np.array(w_std))
        d_cal = prediction_interval_coverage(np.array(d_true), np.array(d_pred), np.array(d_std))
        results[held_out] = {
            "width": w_metrics,
            "depth": d_metrics,
            "width_boot": bootstrap_regression_metrics(np.array(w_true), np.array(w_pred)),
            "depth_boot": bootstrap_regression_metrics(np.array(d_true), np.array(d_pred)),
            "width_calibration": w_cal,
            "depth_calibration": d_cal,
            "n": len(w_true),
        }
    return results


# ---------------------------------------------------------------------------
# Sobol sensitivity analysis (first-order, via Saltelli-style sampling)
# ---------------------------------------------------------------------------


def sobol_first_order(model, n_samples=2000, random_seed=42):
    """Estimate first-order Sobol indices for the physics features.

    Uses a simple Saltelli-style A/B sampling over the training feature ranges.
    """
    rng = np.random.default_rng(random_seed)

    # Feature ranges from the model's training data
    if model.width_gp is None:
        return {}
    # Reconstruct feature ranges from the stored training inputs
    X = model.width_gp.X_train  # standardized features
    # Unstandardize
    x_mean = model.width_gp._x_mean
    x_std = model.width_gp._x_std
    X_raw = X * x_std + x_mean

    d = X_raw.shape[1]
    low = X_raw.min(axis=0)
    high = X_raw.max(axis=0)

    # Sample A and B matrices
    A = rng.uniform(low, high, size=(n_samples, d))
    B = rng.uniform(low, high, size=(n_samples, d))

    indices = {}
    y_pred = model.width_gp.predict
    y_A = np.array([y_pred(a)[0] for a in A])
    y_B = np.array([y_pred(b)[0] for b in B])
    var_total = np.var(np.concatenate([y_A, y_B]))

    if var_total == 0:
        return {name: 0.0 for name in model.feature_names}

    for i in range(d):
        # AB_i: matrix A with column i replaced by B's column i
        AB = A.copy()
        AB[:, i] = B[:, i]
        y_AB = np.array([y_pred(a)[0] for a in AB])
        # First-order index: E[V(Y|X_i)] / V(Y)
        # S_i = (mean over samples of (y_A - y_AB)^2) / (2 * V(Y))
        S_i = np.mean((y_A - y_AB) ** 2) / (2.0 * var_total)
        indices[model.feature_names[i]] = float(S_i)

    return indices


# ---------------------------------------------------------------------------
# Metrics report
# ---------------------------------------------------------------------------


def format_metrics(metrics, boot=None):
    lines = [
        f"  R^2  = {metrics['r2']:.4f}" + (f"  (95% CI [{boot['r2_lo']:.3f}, {boot['r2_hi']:.3f}])" if boot else ""),
        f"  RMSE = {metrics['rmse']:.3e} m" + (f"  (95% CI [{boot['rmse_lo']:.3e}, {boot['rmse_hi']:.3e}])" if boot else ""),
        f"  MAE  = {metrics['mae']:.3e} m",
        f"  N    = {metrics['n']}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Train and validate physics-informed surrogate")
    parser.add_argument("--outdir", default="figures", help="Output directory for figures and report")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load unified dataset
    cases = load_unified_dataset()
    print(f"Loaded {len(cases)} unified cases")
    print(f"By alloy: {dict(Counter(c.alloy for c in cases))}")
    print(f"By source: {dict(Counter(c.source for c in cases))}")

    # Materials present
    materials = sorted(set(c.alloy for c in cases))
    print(f"Materials: {materials}")

    report = []
    report.append("=" * 70)
    report.append("PHYSICS-INFORMED SURROGATE MODEL -- VALIDATION REPORT")
    report.append("=" * 70)
    report.append(f"Unified cases: {len(cases)}")
    report.append(f"Materials: {materials}")
    report.append("")

    # ------------------------------------------------------------------
    # 0. Data-driven absorptivity calibration (literature-bounded grid search)
    # ------------------------------------------------------------------
    report.append("--- 0. ABSORPTIVITY CALIBRATION (literature-bounded grid search on Rosenthal baseline) ---")
    absorptivity_by_alloy = {}
    for m in materials:
        mat = get_material(m)
        try:
            cal = calibrate_absorptivity(cases, mat, alloy_key=m)
        except ValueError as e:
            report.append(f"  {m}: SKIPPED ({e})")
            continue
        absorptivity_by_alloy[m] = cal["absorptivity"]
        lo, hi = None, None
        from rosenthal.physics_informed import LITERATURE_ABSORPTIVITY_BOUNDS

        if m in LITERATURE_ABSORPTIVITY_BOUNDS:
            lo, hi = LITERATURE_ABSORPTIVITY_BOUNDS[m]
        at_bound = ""
        if lo is not None and (abs(cal["absorptivity"] - lo) < 1e-9 or abs(cal["absorptivity"] - hi) < 1e-9):
            at_bound = "  [AT SEARCH BOUNDARY -- baseline cannot fit within literature range; see note below]"
        report.append(
            f"  {m}: absorptivity={cal['absorptivity']:.3f} (bounds [{lo}, {hi}]), "
            f"combined width+depth RMSE={cal['rmse']:.3e} m, n={cal['n_cases']}{at_bound}"
        )
    report.append(
        "\n  NOTE: for alloys whose optimum sits at the literature search boundary, "
        "this indicates the Rosenthal baseline's structural depth=width/2 constraint "
        "(a semicircular cross-section forced by the point-source-at-surface geometry, "
        "see rosenthal/model.py docstring) is the dominant source of baseline error, not "
        "absorptivity -- no physically bounded absorptivity value can make a fixed-aspect-"
        "ratio baseline match real conduction-mode (wide, shallow) or keyhole-mode (narrow, "
        "deep) pools simultaneously in width and depth. Absorptivity calibration below is "
        "still applied (it is a genuine, if partial, improvement over an arbitrary flat 0.5), "
        "but does not resolve this structural limitation."
    )
    report.append("")

    # ------------------------------------------------------------------
    # 1. Pairwise cross-alloy transfer (single-alloy surrogate)
    # ------------------------------------------------------------------
    report.append("--- 1. PAIRWISE CROSS-ALLOY TRANSFER (train on alloy X, test on alloy Y) ---")
    report.append(
        "  (Replaces an earlier 'leave-one-alloy-out' version of this diagnostic that\n"
        "   silently trained on only one of the remaining alloys when >2 alloys were\n"
        "   present -- see cross_alloy_transfer_matrix() docstring in this script.)"
    )
    transfer = cross_alloy_transfer_matrix(cases, materials, absorptivity_by_alloy=absorptivity_by_alloy)
    for (train_alloy, test_alloy), res in transfer.items():
        report.append(f"\nTrain: {train_alloy}  ->  Test: {test_alloy}")
        if "error" in res:
            report.append(f"  ERROR: {res['error']}")
            continue
        report.append(f"  N = {res['n']}")
        report.append("  Width:")
        report.append(format_metrics(res["width"], res.get("width_boot")))
        report.append("  Depth:")
        report.append(format_metrics(res["depth"], res.get("depth_boot")))
    report.append("")

    # ------------------------------------------------------------------
    # 1b. LOAO cross-validation (global multi-alloy surrogate)
    # ------------------------------------------------------------------
    report.append("--- 1b. LEAVE-ONE-ALLOY-OUT CROSS-VALIDATION (global surrogate) ---")
    loao_global = leave_one_alloy_out_cv_global(cases, materials, absorptivity_by_alloy=absorptivity_by_alloy)
    for held_out, res in loao_global.items():
        report.append(f"\nHeld-out alloy: {held_out}")
        if "error" in res:
            report.append(f"  ERROR: {res['error']}")
            continue
        report.append(f"  N = {res['n']}")
        report.append("  Width:")
        report.append(format_metrics(res["width"], res.get("width_boot")))
        report.append("  Depth:")
        report.append(format_metrics(res["depth"], res.get("depth_boot")))
        if "width_calibration" in res:
            wc = res["width_calibration"]
            dc = res["depth_calibration"]
            report.append(f"  Width 95% PI coverage: {wc['coverage']:.3f} (target 0.95)")
            report.append(f"  Depth 95% PI coverage: {dc['coverage']:.3f} (target 0.95)")
    report.append("")

    # ------------------------------------------------------------------
    # 1c. IN718 source-composition diagnostic (experimental vs FE-simulation)
    # ------------------------------------------------------------------
    report.append("--- 1c. IN718 SOURCE-COMPOSITION DIAGNOSTIC (experimental vs FE-simulation) ---")
    in718_cases = [c for c in cases if c.alloy == "IN718"]
    src_counts = Counter(c.source for c in in718_cases)
    report.append(f"  Source breakdown: {dict(src_counts)}")
    real_cases = [c for c in in718_cases if c.source != "Pramod2023"]
    pramod_cases = [c for c in in718_cases if c.source == "Pramod2023"]
    if real_cases and pramod_cases:
        n_total = len(in718_cases)
        pct_sim = 100.0 * len(pramod_cases) / n_total
        report.append(
            f"  {pct_sim:.0f}% of cases labeled 'IN718' in this dataset are FE-simulation\n"
            f"  (Pramod et al. 2023), {100 - pct_sim:.0f}% are real measurements (NIST AM-Bench 2022,\n"
            f"  n={sum(1 for c in real_cases if c.source == 'NIST_AMBench2022')}, plus MeltPoolNet-aggregated\n"
            f"  literature papers, n={sum(1 for c in real_cases if c.source.startswith('MeltPoolNet'))})."
        )
        mat = get_material("IN718")
        a = absorptivity_by_alloy.get("IN718", 0.5)
        sim_to_real = PhysicsInformedSurrogate(material=mat, absorptivity=a)
        try:
            sim_to_real.fit(pramod_cases)
            wt, wp, dt, dp = [], [], [], []
            for c in real_cases:
                pred = sim_to_real.predict(c.power, c.velocity, c.beam_diameter)
                if np.isnan(pred.width) or np.isnan(pred.depth):
                    continue
                wt.append(c.width)
                wp.append(pred.width)
                dt.append(c.depth)
                dp.append(pred.depth)
            wm = regression_metrics(np.array(wt), np.array(wp))
            dm = regression_metrics(np.array(dt), np.array(dp))
            report.append(
                f"\n  Sim-to-real transfer test (train on Pramod FE-sim, n={len(pramod_cases)};\n"
                f"  test on all real IN718 measurements, n={len(real_cases)}):\n"
                f"    Width R^2 = {wm['r2']:.3f}, RMSE = {wm['rmse']:.3e} m\n"
                f"    Depth R^2 = {dm['r2']:.3f}, RMSE = {dm['rmse']:.3e} m\n"
                f"  Strongly negative R^2 here means a model trained on the FE-simulation\n"
                f"  data does not transfer to real measurements at all -- the sim and real IN718\n"
                f"  sources are not interchangeable, and pooling them without this check would\n"
                f"  silently overstate how much of the IN718 result reflects reality."
            )
        except (ValueError, RuntimeError) as e:
            report.append(f"  Sim-to-real transfer test failed: {e}")
    report.append("")

    # ------------------------------------------------------------------
    # 2. Ablation study on the primary material (316L, most data)
    # ------------------------------------------------------------------
    primary = "316L"
    if primary in materials:
        report.append(f"--- 2. ABLATION STUDY ({primary}) ---")
        mat = get_material(primary)
        primary_cases = [c for c in cases if c.alloy == primary]
        primary_absorptivity = absorptivity_by_alloy.get(primary, 0.5)
        report.append(f"  Using calibrated absorptivity={primary_absorptivity:.3f} (was fixed 0.5 previously)")

        # Split train/test (80/20)
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(primary_cases))
        n_train = int(0.8 * len(primary_cases))
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        train_cases = [primary_cases[i] for i in train_idx]
        test_cases = [primary_cases[i] for i in test_idx]

        # (a) Pure Rosenthal
        ros_preds = fit_rosenthal_baseline(test_cases, mat, primary_absorptivity)
        w_true_r = np.array([c.width for c, _, _ in ros_preds])
        w_pred_r = np.array([w for _, w, _ in ros_preds])
        d_true_r = np.array([c.depth for c, _, _ in ros_preds])
        d_pred_r = np.array([d for _, _, d in ros_preds])
        ros_w = regression_metrics(w_true_r, w_pred_r)
        ros_d = regression_metrics(d_true_r, d_pred_r)

        # (b) Black-box GP on raw P, v
        bb_predict = fit_blackbox_gp(train_cases, mat, primary_absorptivity)
        w_true_b, w_pred_b, d_true_b, d_pred_b = [], [], [], []
        for c in test_cases:
            w, d = bb_predict(c)
            w_true_b.append(c.width)
            w_pred_b.append(w)
            d_true_b.append(c.depth)
            d_pred_b.append(d)
        bb_w = regression_metrics(np.array(w_true_b), np.array(w_pred_b))
        bb_d = regression_metrics(np.array(d_true_b), np.array(d_pred_b))

        # (c) Physics-informed residual GP
        pi_surrogate = PhysicsInformedSurrogate(material=mat, absorptivity=primary_absorptivity)
        pi_surrogate.fit(train_cases)
        w_true_p, w_pred_p, d_true_p, d_pred_p = [], [], [], []
        w_std_p, d_std_p = [], []
        for c in test_cases:
            pred = pi_surrogate.predict(c.power, c.velocity, c.beam_diameter)
            w_true_p.append(c.width)
            w_pred_p.append(pred.width)
            w_std_p.append(pred.width_std)
            d_true_p.append(c.depth)
            d_pred_p.append(pred.depth)
            d_std_p.append(pred.depth_std)
        pi_w = regression_metrics(np.array(w_true_p), np.array(w_pred_p))
        pi_d = regression_metrics(np.array(d_true_p), np.array(d_pred_p))

        # (d) Physics-informed GP with conformal calibration on the train split
        pi_cal_surrogate = PhysicsInformedSurrogate(material=mat, absorptivity=primary_absorptivity)
        pi_cal_surrogate.fit(train_cases)
        pi_cal_surrogate.calibrate_conformal(train_cases, alpha=0.05)
        w_true_pc, w_pred_pc, d_true_pc, d_pred_pc = [], [], [], []
        w_std_pc, d_std_pc = [], []
        for c in test_cases:
            pred = pi_cal_surrogate.predict(c.power, c.velocity, c.beam_diameter)
            w_true_pc.append(c.width)
            w_pred_pc.append(pred.width)
            w_std_pc.append(pred.width_std)
            d_true_pc.append(c.depth)
            d_pred_pc.append(pred.depth)
            d_std_pc.append(pred.depth_std)
        pi_cal_w = regression_metrics(np.array(w_true_pc), np.array(w_pred_pc))
        pi_cal_d = regression_metrics(np.array(d_true_pc), np.array(d_pred_pc))

        report.append("\nWidth ablation:")
        report.append(f"  Rosenthal:                R^2={ros_w['r2']:.4f} RMSE={ros_w['rmse']:.2e} m")
        report.append(f"  Black-box GP:             R^2={bb_w['r2']:.4f} RMSE={bb_w['rmse']:.2e} m")
        report.append(f"  Physics-Informed:         R^2={pi_w['r2']:.4f} RMSE={pi_w['rmse']:.2e} m")
        report.append(f"  Physics-Informed (calibrated): R^2={pi_cal_w['r2']:.4f} RMSE={pi_cal_w['rmse']:.2e} m")
        report.append("\nDepth ablation:")
        report.append(f"  Rosenthal:                R^2={ros_d['r2']:.4f} RMSE={ros_d['rmse']:.2e} m")
        report.append(f"  Black-box GP:             R^2={bb_d['r2']:.4f} RMSE={bb_d['rmse']:.2e} m")
        report.append(f"  Physics-Informed:         R^2={pi_d['r2']:.4f} RMSE={pi_d['rmse']:.2e} m")
        report.append(f"  Physics-Informed (calibrated): R^2={pi_cal_d['r2']:.4f} RMSE={pi_cal_d['rmse']:.2e} m")

        # ------------------------------------------------------------------
        # 3. Uncertainty calibration
        # ------------------------------------------------------------------
        report.append("\n--- 3. UNCERTAINTY CALIBRATION (95% PI) ---")
        report.append("  (a) Raw GP posterior")
        w_cal = prediction_interval_coverage(np.array(w_true_p), np.array(w_pred_p), np.array(w_std_p))
        d_cal = prediction_interval_coverage(np.array(d_true_p), np.array(d_pred_p), np.array(d_std_p))
        report.append(f"  Width:  coverage={w_cal['coverage']:.3f} (target 0.95), mean width={w_cal['mean_interval_width']:.2e} m")
        report.append(f"  Depth:  coverage={d_cal['coverage']:.3f} (target 0.95), mean width={d_cal['mean_interval_width']:.2e} m")
        report.append("  (b) After conformal calibration")
        w_cal_c = prediction_interval_coverage(np.array(w_true_pc), np.array(w_pred_pc), np.array(w_std_pc))
        d_cal_c = prediction_interval_coverage(np.array(d_true_pc), np.array(d_pred_pc), np.array(d_std_pc))
        report.append(f"  Width:  coverage={w_cal_c['coverage']:.3f} (target 0.95), mean width={w_cal_c['mean_interval_width']:.2e} m")
        report.append(f"  Depth:  coverage={d_cal_c['coverage']:.3f} (target 0.95), mean width={d_cal_c['mean_interval_width']:.2e} m")
        report.append(f"  Calibration factors: width={pi_cal_surrogate.width_cal_factor:.3f}, depth={pi_cal_surrogate.depth_cal_factor:.3f}")

        # ------------------------------------------------------------------
        # 3b. Reliability / calibration curve (empirical coverage vs nominal level)
        # ------------------------------------------------------------------
        report.append("\n--- 3b. RELIABILITY CURVE (raw GP posterior, physics-informed) ---")
        w_curve = empirical_coverage_curve(np.array(w_true_p), np.array(w_pred_p), np.array(w_std_p))
        d_curve = empirical_coverage_curve(np.array(d_true_p), np.array(d_pred_p), np.array(d_std_p))
        report.append("  Nominal level | Width coverage | Depth coverage")
        for lvl, wc, dc in zip(w_curve["levels"], w_curve["coverage"], d_curve["coverage"]):
            report.append(f"      {lvl:.2f}      |      {wc:.3f}      |      {dc:.3f}")
        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(w_curve["levels"], w_curve["coverage"], "o-", label="Width")
            ax.plot(d_curve["levels"], d_curve["coverage"], "s-", label="Depth")
            ax.plot([0.5, 0.95], [0.5, 0.95], "k--", label="Perfectly calibrated")
            ax.set_xlabel("Nominal coverage level")
            ax.set_ylabel("Empirical coverage")
            ax.set_title("Reliability Diagram (Physics-Informed)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(outdir / "reliability_curve.png", dpi=150)
            plt.close()
            report.append("\nSaved reliability_curve.png")

        # ------------------------------------------------------------------
        # 4. Sobol sensitivity
        # ------------------------------------------------------------------
        report.append("\n--- 4. SOBOL SENSITIVITY (width, first-order) ---")
        sobol = sobol_first_order(pi_surrogate, n_samples=500)
        for name, val in sobol.items():
            report.append(f"  {name}: {val:.4f}")

        # ------------------------------------------------------------------
        # 5. Parity plots
        # ------------------------------------------------------------------
        if HAS_MPL:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, (true, pred, title) in zip(
                axes,
                [
                    (w_true_p, w_pred_p, "Width Parity (Physics-Informed)"),
                    (d_true_p, d_pred_p, "Depth Parity (Physics-Informed)"),
                ],
            ):
                true = np.array(true) * 1e6
                pred = np.array(pred) * 1e6
                ax.scatter(true, pred, alpha=0.6, s=20)
                lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
                ax.plot(lims, lims, "k--", label="1:1")
                ax.set_xlabel("Measured (um)")
                ax.set_ylabel("Predicted (um)")
                ax.set_title(title)
                ax.legend()
            plt.tight_layout()
            plt.savefig(outdir / "parity_physics_informed.png", dpi=150)
            plt.close()
            report.append("\nSaved parity_physics_informed.png")

    # Write report
    report_path = outdir / "metrics_report.txt"
    report_path.write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
