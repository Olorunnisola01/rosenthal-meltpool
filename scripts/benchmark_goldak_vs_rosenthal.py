"""Scoped benchmark: Rosenthal vs. Goldak baseline fit quality, per alloy.

This is deliberately NOT full leave-one-alloy-out cross-validation with a GP
residual on top of Goldak -- that remains out of scope for compute-cost
reasons (see TODO.md). This script instead runs all three Goldak shape-
calibration strategies (one-step geometric, depth-only least-squares, joint
(a,b) least-squares) against the SAME fixed evaluation subset per alloy, so
the comparison across strategies -- and against Rosenthal -- is apples-to-
apples rather than confounded by different random subsets across separate
runs (an issue in an earlier version of this script). Uses
`width_depth_fast()` (fixed-quadrature, ~25x faster than the adaptive
solver, validated in tests/test_physics_informed.py) so a larger, more
statistically stable subset is affordable.

Usage:
    python scripts/benchmark_goldak_vs_rosenthal.py [--n-per-alloy 30]
"""

import argparse

import numpy as np

from rosenthal.goldak import (
    GoldakParameters,
    calibrate_goldak_depth_only,
    calibrate_goldak_shape,
    calibrate_goldak_shape_lsq,
    width_depth_fast,
)
from rosenthal.materials import get_material
from rosenthal.model import ProcessParameters, melt_pool_dimensions as rosenthal_dimensions
from rosenthal.physics_informed import (
    calibrate_absorptivity,
    load_unified_dataset,
    regression_metrics,
)


def _eval_goldak(subset, shape, absorptivity, mat):
    w_true, w_pred, d_true, d_pred = [], [], [], []
    for c in subset:
        params = GoldakParameters(
            power=c.power,
            velocity=c.velocity,
            absorptivity=absorptivity,
            a=shape["a"],
            b=shape["b"],
            c_front=shape["c_front"],
            c_rear=shape["c_rear"],
        )
        try:
            w, d = width_depth_fast(params, mat)
        except ValueError:
            continue
        w_true.append(c.width)
        w_pred.append(w)
        d_true.append(c.depth)
        d_pred.append(d)
    return regression_metrics(np.array(w_true), np.array(w_pred)), regression_metrics(np.array(d_true), np.array(d_pred))


def _kfold_goldak(alloy_cases, mat, absorptivity, k=4, seed=42):
    """Proper k-fold CV: calibrate on k-1 folds, evaluate on the held-out
    fold, repeat so every case is evaluated exactly once out-of-fold.
    Unlike the in-sample comparison in main(), calibration and evaluation
    never draw from the same cases within a fold -- see round 15 in
    TODO.md for why this was added (the original single-split comparison
    had 100% calibration/evaluation overlap for AlSi10Mg, which only has
    20 total cases).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(alloy_cases))
    folds = np.array_split(idx, min(k, len(alloy_cases)))

    w_true_all, w_pred_all, d_true_all, d_pred_all = [], [], [], []
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        if len(train_idx) < 4:
            continue
        train = [alloy_cases[j] for j in train_idx]
        test = [alloy_cases[j] for j in test_idx]
        try:
            shape = calibrate_goldak_shape_lsq(train, mat, absorptivity=absorptivity, n_calibration_cases=min(18, len(train)))
        except ValueError:
            continue
        for c in test:
            params = GoldakParameters(
                power=c.power, velocity=c.velocity, absorptivity=absorptivity,
                a=shape["a"], b=shape["b"], c_front=shape["c_front"], c_rear=shape["c_rear"],
            )
            try:
                w, d = width_depth_fast(params, mat)
            except ValueError:
                continue
            w_true_all.append(c.width); w_pred_all.append(w)
            d_true_all.append(c.depth); d_pred_all.append(d)

    w_m = regression_metrics(np.array(w_true_all), np.array(w_pred_all))
    d_m = regression_metrics(np.array(d_true_all), np.array(d_pred_all))
    return w_m, d_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-alloy", type=int, default=30)
    parser.add_argument("--kfold", action="store_true", help="Run proper k-fold CV (no calibration/eval overlap) instead of the single in-sample-risk comparison")
    args = parser.parse_args()

    if args.kfold:
        cases = load_unified_dataset()
        materials = sorted(set(c.alloy for c in cases))
        print("=" * 70)
        print("ROSENTHAL vs GOLDAK (JOINT a,b): PROPER 4-FOLD CV, NO CALIBRATION/EVAL OVERLAP")
        print("=" * 70)
        for alloy in materials:
            alloy_cases = [c for c in cases if c.alloy == alloy and not np.isnan(c.width) and not np.isnan(c.depth)]
            mat = get_material(alloy)
            cal = calibrate_absorptivity(cases, mat, alloy_key=alloy)
            absorptivity = cal["absorptivity"]

            ros_w_true, ros_w_pred, ros_d_true, ros_d_pred = [], [], [], []
            for c in alloy_cases:
                try:
                    d = rosenthal_dimensions(ProcessParameters(power=c.power, velocity=c.velocity, absorptivity=absorptivity), mat)
                except ValueError:
                    continue
                ros_w_true.append(c.width); ros_w_pred.append(d["width"])
                ros_d_true.append(c.depth); ros_d_pred.append(d["depth"])
            ros_w_m = regression_metrics(np.array(ros_w_true), np.array(ros_w_pred))
            ros_d_m = regression_metrics(np.array(ros_d_true), np.array(ros_d_pred))

            gold_w_m, gold_d_m = _kfold_goldak(alloy_cases, mat, absorptivity)

            print(f"\n=== {alloy} (n={len(alloy_cases)}, absorptivity={absorptivity:.3f}) ===")
            print(f"  Rosenthal:        width R2={ros_w_m['r2']:+.3f} | depth R2={ros_d_m['r2']:+.3f}")
            print(f"  Goldak (4-fold CV, genuinely held-out): width R2={gold_w_m['r2']:+.3f} | depth R2={gold_d_m['r2']:+.3f}")
        return

    cases = load_unified_dataset()
    materials = sorted(set(c.alloy for c in cases))

    print("=" * 70)
    print("ROSENTHAL vs GOLDAK BASELINE BENCHMARK -- all calibration strategies, same subset")
    print("=" * 70)

    for alloy in materials:
        alloy_cases = [c for c in cases if c.alloy == alloy and not np.isnan(c.width) and not np.isnan(c.depth)]
        mat = get_material(alloy)

        cal = calibrate_absorptivity(cases, mat, alloy_key=alloy)
        absorptivity = cal["absorptivity"]

        rng = np.random.default_rng(42)
        idx = rng.permutation(len(alloy_cases))[: min(args.n_per_alloy, len(alloy_cases))]
        subset = [alloy_cases[i] for i in idx]

        # Rosenthal
        ros_w_true, ros_w_pred, ros_d_true, ros_d_pred = [], [], [], []
        for c in subset:
            params = ProcessParameters(power=c.power, velocity=c.velocity, absorptivity=absorptivity)
            try:
                d = rosenthal_dimensions(params, mat)
            except ValueError:
                continue
            ros_w_true.append(c.width)
            ros_w_pred.append(d["width"])
            ros_d_true.append(c.depth)
            ros_d_pred.append(d["depth"])
        ros_w_m = regression_metrics(np.array(ros_w_true), np.array(ros_w_pred))
        ros_d_m = regression_metrics(np.array(ros_d_true), np.array(ros_d_pred))

        # Goldak, three calibration strategies (fit against alloy_cases, evaluated on `subset`)
        geom = calibrate_goldak_shape(alloy_cases, mat, absorptivity=absorptivity)
        depth_only = calibrate_goldak_depth_only(alloy_cases, mat, absorptivity=absorptivity, n_calibration_cases=20)
        joint = calibrate_goldak_shape_lsq(alloy_cases, mat, absorptivity=absorptivity, n_calibration_cases=20)

        geom_w, geom_d = _eval_goldak(subset, geom, absorptivity, mat)
        depth_w, depth_d = _eval_goldak(subset, depth_only, absorptivity, mat)
        joint_w, joint_d = _eval_goldak(subset, joint, absorptivity, mat)

        meas_ratio = np.mean([c.depth / c.width for c in subset if c.width > 0])

        print(f"\n=== {alloy} (absorptivity={absorptivity:.3f}, eval n={len(subset)}) ===")
        print(f"  Measured mean depth/width ratio: {meas_ratio:.3f}")
        print(f"  Rosenthal:            width R2={ros_w_m['r2']:+.3f} RMSE={ros_w_m['rmse']:.2e} | depth R2={ros_d_m['r2']:+.3f} RMSE={ros_d_m['rmse']:.2e}  (ratio fixed 0.500)")
        print(f"  Goldak (geometric):   width R2={geom_w['r2']:+.3f} RMSE={geom_w['rmse']:.2e} | depth R2={geom_d['r2']:+.3f} RMSE={geom_d['rmse']:.2e}")
        print(f"  Goldak (depth-only):  width R2={depth_w['r2']:+.3f} RMSE={depth_w['rmse']:.2e} | depth R2={depth_d['r2']:+.3f} RMSE={depth_d['rmse']:.2e}  (b={depth_only['b']*1e6:.1f}um, success={depth_only['success']})")
        print(f"  Goldak (joint a,b):   width R2={joint_w['r2']:+.3f} RMSE={joint_w['rmse']:.2e} | depth R2={joint_d['r2']:+.3f} RMSE={joint_d['rmse']:.2e}  (a={joint['a']*1e6:.1f}um b={joint['b']*1e6:.1f}um, success={joint['success']})")

        best_depth_r2 = max(ros_d_m["r2"], geom_d["r2"], depth_d["r2"], joint_d["r2"])
        winner = {ros_d_m["r2"]: "Rosenthal", geom_d["r2"]: "Goldak-geometric", depth_d["r2"]: "Goldak-depth-only", joint_d["r2"]: "Goldak-joint"}[best_depth_r2]
        print(f"  --> Best depth R^2 for {alloy}: {winner} ({best_depth_r2:+.3f})")


def kfold_mode_stratified(cases_all, materials, k=3):
    """Re-audit of round 8's mode-stratified comparison under proper
    disjoint-fold CV (round 16's fix, applied to the mode-stratified
    numbers flagged as not-yet-re-verified in docs/paper2_draft.tex).
    """
    from rosenthal.physics_informed import KEYHOLE_DW_RATIO_THRESHOLD

    print("=" * 70)
    print("MODE-STRATIFIED GOLDAK: PROPER K-FOLD CV RE-AUDIT (round 16 follow-up)")
    print("=" * 70)
    for alloy in materials:
        alloy_cases = [c for c in cases_all if c.alloy == alloy and not np.isnan(c.width) and not np.isnan(c.depth) and c.width > 0]
        mat = get_material(alloy)
        cal = calibrate_absorptivity(cases_all, mat, alloy_key=alloy)
        absorptivity = cal["absorptivity"]

        ratios = [c.depth / c.width for c in alloy_cases]
        keyhole = [c for c, r in zip(alloy_cases, ratios) if r >= KEYHOLE_DW_RATIO_THRESHOLD]
        if len(keyhole) < 2 * k:
            print(f"\n{alloy}: keyhole n={len(keyhole)}, too few for {k}-fold CV, skipping")
            continue

        ros_w_true, ros_w_pred, ros_d_true, ros_d_pred = [], [], [], []
        for c in keyhole:
            try:
                d = rosenthal_dimensions(ProcessParameters(power=c.power, velocity=c.velocity, absorptivity=absorptivity), mat)
            except ValueError:
                continue
            ros_w_true.append(c.width); ros_w_pred.append(d["width"])
            ros_d_true.append(c.depth); ros_d_pred.append(d["depth"])
        ros_d_m = regression_metrics(np.array(ros_d_true), np.array(ros_d_pred))

        gold_w_m, gold_d_m = _kfold_goldak(keyhole, mat, absorptivity, k=k)
        print(f"\n{alloy} (keyhole n={len(keyhole)}): Rosenthal depth R2={ros_d_m['r2']:+.3f} | Goldak ({k}-fold CV) depth R2={gold_d_m['r2']:+.3f}")




def kfold_mode_aware_predictor(cases_all, materials, k=4):
    """Re-audit of round 9's mode-aware PREDICTOR (normalized-enthalpy
    threshold classifier + per-mode Goldak shape, using only process
    parameters at prediction time) under proper disjoint-fold CV. Both the
    threshold and the two per-mode shapes are fit on the training folds
    only and applied to the untouched held-out fold -- the full pipeline,
    not just the shape-fitting step.
    """
    from rosenthal.goldak import calibrate_goldak_mode_aware, predict_mode_aware

    print("=" * 70)
    print("MODE-AWARE PREDICTOR: PROPER K-FOLD CV RE-AUDIT (round 17 follow-up)")
    print("=" * 70)
    for alloy in materials:
        alloy_cases = [c for c in cases_all if c.alloy == alloy and not np.isnan(c.width) and not np.isnan(c.depth) and c.width > 0]
        mat = get_material(alloy)
        cal = calibrate_absorptivity(cases_all, mat, alloy_key=alloy)
        absorptivity = cal["absorptivity"]

        if len(alloy_cases) < 4 * k:
            print(f"\n{alloy}: n={len(alloy_cases)}, too few for {k}-fold CV of the full pipeline, skipping")
            continue

        rng = np.random.default_rng(11)
        idx = rng.permutation(len(alloy_cases))
        folds = np.array_split(idx, k)

        ros_d_true, ros_d_pred = [], []
        gold_d_true, gold_d_pred = [], []
        mode_correct, mode_total = 0, 0
        for i, test_idx in enumerate(folds):
            train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
            train = [alloy_cases[j] for j in train_idx]
            test = [alloy_cases[j] for j in test_idx]
            try:
                calib = calibrate_goldak_mode_aware(train, mat, absorptivity=absorptivity, min_cases_per_mode=6)
            except ValueError:
                continue
            for c in test:
                try:
                    d = rosenthal_dimensions(ProcessParameters(power=c.power, velocity=c.velocity, absorptivity=absorptivity), mat)
                    ros_d_true.append(c.depth); ros_d_pred.append(d["depth"])
                except ValueError:
                    pass
                try:
                    pred = predict_mode_aware(c.power, c.velocity, c.beam_diameter, mat, absorptivity, calib)
                except ValueError:
                    continue
                gold_d_true.append(c.depth); gold_d_pred.append(pred["depth"])
                true_mode = "keyhole" if (c.depth / c.width) >= 0.8 else "conduction"
                mode_total += 1
                if pred["mode"] == true_mode:
                    mode_correct += 1

        ros_d_m = regression_metrics(np.array(ros_d_true), np.array(ros_d_pred))
        gold_d_m = regression_metrics(np.array(gold_d_true), np.array(gold_d_pred))
        acc = mode_correct / mode_total if mode_total else float("nan")
        print(f"\n{alloy} (n={len(alloy_cases)}): Rosenthal depth R2={ros_d_m['r2']:+.3f} | Mode-aware predictor ({k}-fold CV) depth R2={gold_d_m['r2']:+.3f}, mode accuracy={acc:.2f}")


if __name__ == "__main__":
    import sys

    if "--mode-audit" in sys.argv:
        _cases = load_unified_dataset()
        _materials = sorted(set(c.alloy for c in _cases))
        kfold_mode_stratified(_cases, _materials)
    elif "--predictor-audit" in sys.argv:
        _cases = load_unified_dataset()
        _materials = sorted(set(c.alloy for c in _cases))
        kfold_mode_aware_predictor(_cases, _materials)
    else:
        main()
