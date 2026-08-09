"""Full GP-residual LOAO comparison: Rosenthal baseline vs. Goldak baseline.

For each alloy, calibrate absorptivity and Goldak shape (joint least-squares),
then run the same pairwise cross-alloy transfer diagnostic used in
scripts/train_physics_informed.py (train PhysicsInformedSurrogate on alloy X,
test on alloy Y) twice: once with the Rosenthal baseline (as today), once with
the Goldak baseline substituted in via PhysicsInformedSurrogate(goldak_shape=...).

Usage:
    python scripts/benchmark_goldak_gp_loao.py
"""

import numpy as np

from rosenthal.goldak import calibrate_goldak_shape_lsq
from rosenthal.materials import get_material
from rosenthal.physics_informed import (
    PhysicsInformedSurrogate,
    calibrate_absorptivity,
    load_unified_dataset,
    regression_metrics,
)


def _transfer(cases, materials, absorptivity_by_alloy, goldak_shape_by_alloy=None):
    results = {}
    for train_alloy in materials:
        train = [c for c in cases if c.alloy == train_alloy]
        mat = get_material(train_alloy)
        a = absorptivity_by_alloy[train_alloy]
        shape = goldak_shape_by_alloy[train_alloy] if goldak_shape_by_alloy else None
        surrogate = PhysicsInformedSurrogate(material=mat, absorptivity=a, goldak_shape=shape)
        try:
            surrogate.fit(train)
        except (ValueError, RuntimeError) as e:
            continue
        for test_alloy in materials:
            if test_alloy == train_alloy:
                continue
            test = [c for c in cases if c.alloy == test_alloy]
            wt, wp, dt, dp = [], [], [], []
            for c in test:
                try:
                    pred = surrogate.predict(c.power, c.velocity, c.beam_diameter)
                except Exception:
                    continue
                if np.isnan(pred.width) or np.isnan(pred.depth):
                    continue
                wt.append(c.width)
                wp.append(pred.width)
                dt.append(c.depth)
                dp.append(pred.depth)
            if not wt:
                continue
            results[(train_alloy, test_alloy)] = {
                "width": regression_metrics(np.array(wt), np.array(wp)),
                "depth": regression_metrics(np.array(dt), np.array(dp)),
            }
    return results


def main():
    cases = load_unified_dataset(balance_in718_sources=True)
    materials = sorted(set(c.alloy for c in cases))

    absorptivity_by_alloy = {}
    goldak_shape_by_alloy = {}
    for m in materials:
        mat = get_material(m)
        cal = calibrate_absorptivity(cases, mat, alloy_key=m)
        absorptivity_by_alloy[m] = cal["absorptivity"]
        alloy_cases = [c for c in cases if c.alloy == m]
        goldak_shape_by_alloy[m] = calibrate_goldak_shape_lsq(alloy_cases, mat, absorptivity=cal["absorptivity"], n_calibration_cases=18)
        print(f"{m}: absorptivity={cal['absorptivity']:.3f} goldak a={goldak_shape_by_alloy[m]['a']*1e6:.1f}um b={goldak_shape_by_alloy[m]['b']*1e6:.1f}um")

    print("\nRunning Rosenthal-baseline transfer matrix...")
    ros_results = _transfer(cases, materials, absorptivity_by_alloy)
    print("Running Goldak-baseline transfer matrix...")
    gold_results = _transfer(cases, materials, absorptivity_by_alloy, goldak_shape_by_alloy)

    print("\n" + "=" * 90)
    print(f"{'train->test':<20}{'Rosenthal width R2':<20}{'Goldak width R2':<20}{'Rosenthal depth R2':<20}{'Goldak depth R2':<20}")
    print("=" * 90)
    ros_depth_wins, gold_depth_wins = 0, 0
    ros_width_wins, gold_width_wins = 0, 0
    for key in ros_results:
        if key not in gold_results:
            continue
        rw = ros_results[key]["width"]["r2"]
        gw = gold_results[key]["width"]["r2"]
        rd = ros_results[key]["depth"]["r2"]
        gd = gold_results[key]["depth"]["r2"]
        pair = f"{key[0]}->{key[1]}"
        print(f"{pair:<20}{rw:<20.3f}{gw:<20.3f}{rd:<20.3f}{gd:<20.3f}")
        if gd > rd:
            gold_depth_wins += 1
        else:
            ros_depth_wins += 1
        if gw > rw:
            gold_width_wins += 1
        else:
            ros_width_wins += 1

    print(f"\nDepth: Goldak wins {gold_depth_wins}/{gold_depth_wins+ros_depth_wins} pairs")
    print(f"Width: Goldak wins {gold_width_wins}/{gold_width_wins+ros_width_wins} pairs")


if __name__ == "__main__":
    main()
