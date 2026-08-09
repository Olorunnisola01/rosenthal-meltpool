"""Unit tests for the physics-informed residual surrogate module."""

import math

import numpy as np
import pytest

from rosenthal.materials import get_material
from rosenthal.physics_informed import (
    LITERATURE_ABSORPTIVITY_BOUNDS,
    GlobalPhysicsInformedSurrogate,
    GaussianProcessRegressor,
    PhysicsInformedSurrogate,
    SurrogateCase,
    areal_energy_density,
    bootstrap_regression_metrics,
    calibrate_absorptivity,
    classify_mode,
    conformal_calibration_factor,
    empirical_coverage_curve,
    linear_energy_density,
    load_unified_dataset,
    normalized_enthalpy,
    peclet_number,
    physics_features,
    prediction_interval_coverage,
    regression_metrics,
)


# ---------------------------------------------------------------------------
# Physics features
# ---------------------------------------------------------------------------


class TestPhysicsFeatures:
    def test_normalized_enthalpy_positive(self):
        mat = get_material("316L")
        val = normalized_enthalpy(200.0, 1.0, 100e-6, mat, 0.5)
        assert val > 0
        assert math.isfinite(val)

    def test_normalized_enthalpy_increases_with_power(self):
        mat = get_material("316L")
        low = normalized_enthalpy(100.0, 1.0, 100e-6, mat, 0.5)
        high = normalized_enthalpy(400.0, 1.0, 100e-6, mat, 0.5)
        assert high > low

    def test_normalized_enthalpy_decreases_with_velocity(self):
        mat = get_material("316L")
        slow = normalized_enthalpy(200.0, 0.5, 100e-6, mat, 0.5)
        fast = normalized_enthalpy(200.0, 2.0, 100e-6, mat, 0.5)
        assert slow > fast

    def test_peclet_number(self):
        mat = get_material("316L")
        pe = peclet_number(1.0, 100e-6, mat)
        assert pe > 0
        # Pe = v*d/(2*alpha)
        expected = 1.0 * 100e-6 / (2.0 * mat.alpha)
        assert pe == pytest.approx(expected)

    def test_energy_densities(self):
        assert linear_energy_density(200.0, 1.0) == pytest.approx(200.0)
        assert areal_energy_density(200.0, 1.0, 100e-6) == pytest.approx(200.0 / 100e-6)

    def test_physics_features_dict(self):
        mat = get_material("316L")
        feats = physics_features(200.0, 1.0, 100e-6, mat, 0.5)
        assert set(feats) == {
            "normalized_enthalpy",
            "peclet_number",
            "linear_energy_density",
            "areal_energy_density",
        }
        for v in feats.values():
            assert math.isfinite(v)


# ---------------------------------------------------------------------------
# Mode classification
# ---------------------------------------------------------------------------


class TestClassifyMode:
    def test_conduction(self):
        assert classify_mode(0.5) == "conduction"

    def test_keyhole(self):
        assert classify_mode(1.0) == "keyhole"

    def test_boundary(self):
        assert classify_mode(0.8) == "keyhole"

    def test_nan(self):
        assert classify_mode(float("nan")) == "no_melt"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_regression_metrics_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        m = regression_metrics(y, y)
        assert m["r2"] == pytest.approx(1.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["n"] == 3

    def test_regression_metrics_handles_nan(self):
        y_true = np.array([1.0, 2.0, float("nan")])
        y_pred = np.array([1.0, 2.0, 3.0])
        m = regression_metrics(y_true, y_pred)
        assert m["n"] == 2

    def test_pi_coverage_perfect(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        y_std = np.array([0.1, 0.1, 0.1])
        c = prediction_interval_coverage(y_true, y_pred, y_std)
        assert c["coverage"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Gaussian process
# ---------------------------------------------------------------------------


class TestGaussianProcess:
    def test_gp_fits_linear_function(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, size=(30, 2))
        y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5
        gp = GaussianProcessRegressor(n_restarts=2, random_seed=0).fit(X, y)
        mean, std = gp.predict(X[:5])
        assert mean.shape == (5,)
        assert std.shape == (5,)
        assert np.all(std >= 0)
        # GP should fit training points closely
        assert np.allclose(mean, y[:5], atol=0.5)

    def test_gp_requires_fit(self):
        gp = GaussianProcessRegressor()
        with pytest.raises(RuntimeError):
            gp.predict(np.array([[0.0, 0.0]]))


# ---------------------------------------------------------------------------
# Unified dataset
# ---------------------------------------------------------------------------


class TestUnifiedDataset:
    def test_load_all(self):
        cases = load_unified_dataset()
        assert len(cases) > 0
        alloys = {c.alloy for c in cases}
        assert "316L" in alloys
        assert "IN718" in alloys

    def test_load_316l_only(self):
        cases = load_unified_dataset(
            include_in718_nist=False, include_in718_pramod=False, include_in718_meltpoolnet=False, include_in718_chen2020=False, include_ti64=False, include_alsi10mg=False
        )
        assert len(cases) > 0
        assert all(c.alloy == "316L" for c in cases)

    def test_load_in718_only(self):
        cases = load_unified_dataset(include_316l=False, include_ti64=False, include_alsi10mg=False)
        assert len(cases) > 0
        assert all(c.alloy == "IN718" for c in cases)

    def test_load_ti64_only(self):
        cases = load_unified_dataset(
            include_316l=False, include_in718_nist=False, include_in718_pramod=False, include_in718_meltpoolnet=False, include_in718_chen2020=False, include_alsi10mg=False
        )
        assert len(cases) > 0
        assert all(c.alloy == "Ti-6Al-4V" for c in cases)

    def test_load_alsi10mg_only(self):
        cases = load_unified_dataset(
            include_316l=False, include_in718_nist=False, include_in718_pramod=False, include_in718_meltpoolnet=False, include_in718_chen2020=False, include_ti64=False
        )
        assert len(cases) > 0
        assert all(c.alloy == "AlSi10Mg" for c in cases)

    def test_load_all_four_alloys(self):
        cases = load_unified_dataset()
        alloys = {c.alloy for c in cases}
        assert alloys == {"316L", "IN718", "Ti-6Al-4V", "AlSi10Mg"}


# ---------------------------------------------------------------------------
# Surrogate
# ---------------------------------------------------------------------------


class TestSurrogate:
    def _make_cases(self, n=20):
        """Synthetic cases for a quick surrogate fit test."""
        mat = get_material("316L")
        cases = []
        for i in range(n):
            p = 100.0 + 300.0 * (i / n)
            v = 0.5 + 1.5 * (i / n)
            cases.append(
                SurrogateCase(
                    alloy="316L",
                    source="synthetic",
                    case_id=f"s{i}",
                    power=p,
                    velocity=v,
                    beam_diameter=100e-6,
                    layer_thickness=0.0,
                    width=150e-6 + 50e-6 * (i / n),
                    depth=80e-6 + 40e-6 * (i / n),
                    length=float("nan"),
                )
            )
        return cases

    def test_fit_and_predict(self):
        cases = self._make_cases()
        mat = get_material("316L")
        surr = PhysicsInformedSurrogate(material=mat, absorptivity=0.5, n_restarts=1)
        surr.fit(cases)
        pred = surr.predict(200.0, 1.0, 100e-6)
        assert pred.width > 0
        assert pred.depth > 0
        assert pred.width_std >= 0
        assert pred.depth_std >= 0
        assert pred.mode in ("conduction", "keyhole")

    def test_predict_requires_fit(self):
        mat = get_material("316L")
        surr = PhysicsInformedSurrogate(material=mat)
        with pytest.raises(RuntimeError):
            surr.predict(200.0, 1.0, 100e-6)

    def test_no_melt_returns_nan(self):
        cases = self._make_cases()
        mat = get_material("316L")
        surr = PhysicsInformedSurrogate(material=mat, absorptivity=0.5, n_restarts=1)
        surr.fit(cases)
        # Extremely low power / high velocity -> no melt pool
        pred = surr.predict(1.0, 10.0, 100e-6)
        assert pred.mode == "no_melt"
        assert math.isnan(pred.width)

    def test_calibrate_conformal_updates_factors(self):
        cases = self._make_cases()
        mat = get_material("316L")
        surr = PhysicsInformedSurrogate(material=mat, absorptivity=0.5, n_restarts=1)
        surr.fit(cases)
        # Before calibration, factors are 1.0
        assert surr.width_cal_factor == pytest.approx(1.0)
        assert surr.depth_cal_factor == pytest.approx(1.0)
        # Calibrate on the same cases (in-sample, but tests the mechanism).
        # The factor may be < 1 (over-conservative GP) or > 1 (under-conservative);
        # the key property is that it is positive, finite, and re-scales the std.
        surr.calibrate_conformal(cases, alpha=0.05)
        assert surr.width_cal_factor > 0
        assert math.isfinite(surr.width_cal_factor)
        assert surr.depth_cal_factor > 0
        assert math.isfinite(surr.depth_cal_factor)
        # Prediction std should be rescaled by the factor
        pred = surr.predict(200.0, 1.0, 100e-6)
        assert pred.width_std >= 0
        assert pred.depth_std >= 0

    def test_calibrate_conformal_split_disjoint(self):
        cases = self._make_cases(n=40)
        mat = get_material("316L")
        surr = PhysicsInformedSurrogate(material=mat, absorptivity=0.5, n_restarts=1)
        surr.fit(cases)
        # Split into calibration and evaluation halves (disjoint)
        half = len(cases) // 2
        cal_cases = cases[:half]
        eval_cases = cases[half:]
        result = surr.calibrate_conformal_split(cal_cases, eval_cases, alpha=0.05)
        assert "width" in result and "depth" in result
        # Factors are positive and finite
        assert result["width"]["cal_factor"] > 0
        assert math.isfinite(result["width"]["cal_factor"])
        assert result["depth"]["cal_factor"] > 0
        assert math.isfinite(result["depth"]["cal_factor"])
        # Coverage is a probability in [0, 1]
        assert 0.0 <= result["width"]["coverage"] <= 1.0
        assert 0.0 <= result["depth"]["coverage"] <= 1.0
        # The surrogate's stored factors should match the returned ones
        assert surr.width_cal_factor == pytest.approx(result["width"]["cal_factor"])
        assert surr.depth_cal_factor == pytest.approx(result["depth"]["cal_factor"])


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------


class TestEmpiricalCoverageCurve:
    def test_perfect_calibration(self):
        # Perfect predictions with small, consistent std -> ~100% coverage at all levels
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        y_std = np.array([0.05, 0.05, 0.05, 0.05])
        curve = empirical_coverage_curve(y_true, y_pred, y_std)
        assert "levels" in curve
        assert "coverage" in curve
        assert len(curve["levels"]) == len(curve["coverage"])
        # At the 95% level, coverage should be ~1.0 (all points within 1.96*std)
        assert curve["coverage"][-1] == pytest.approx(1.0)

    def test_monotonic_in_level(self):
        # Coverage should be non-decreasing as the nominal level increases
        rng = np.random.default_rng(0)
        y_true = rng.normal(0, 1, 200)
        y_pred = y_true + rng.normal(0, 0.5, 200)
        y_std = np.full(200, 0.5)
        curve = empirical_coverage_curve(y_true, y_pred, y_std)
        assert np.all(np.diff(curve["coverage"]) >= -1e-6)

    def test_handles_nan(self):
        y_true = np.array([1.0, float("nan"), 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        y_std = np.array([0.1, 0.1, 0.1, 0.1])
        curve = empirical_coverage_curve(y_true, y_pred, y_std)
        assert len(curve["coverage"]) == len(curve["levels"])


class TestConformalCalibration:
    def test_perfect_predictions_give_small_factor(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        y_std = np.array([0.1, 0.1, 0.1, 0.1])
        c = conformal_calibration_factor(y_true, y_pred, y_std, alpha=0.05)
        # All residuals are 0 -> quantile is 0 -> guard returns 1.0
        assert c == pytest.approx(1.0)

    def test_underconfident_std_gives_large_factor(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5])
        y_std = np.array([0.01, 0.01, 0.01, 0.01])
        c = conformal_calibration_factor(y_true, y_pred, y_std, alpha=0.05)
        # Residuals are 0.5 / (1.96*0.01) ~ 25.5 -> factor should be large
        assert c > 1.0

    def test_handles_nan(self):
        y_true = np.array([1.0, float("nan"), 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        y_std = np.array([0.1, 0.1, 0.1])
        c = conformal_calibration_factor(y_true, y_pred, y_std)
        assert c >= 1.0

    def test_empty_returns_one(self):
        c = conformal_calibration_factor(np.array([]), np.array([]), np.array([]))
        assert c == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Global multi-alloy surrogate
# ---------------------------------------------------------------------------


class TestGlobalSurrogate:
    def _make_cases(self, n_per_alloy=15):
        """Synthetic cases across two alloys for a quick global fit test."""
        cases = []
        for alloy, p0, p1, v0, v1 in [
            ("316L", 100.0, 400.0, 0.5, 2.0),
            ("IN718", 120.0, 420.0, 0.4, 1.8),
        ]:
            for i in range(n_per_alloy):
                p = p0 + (p1 - p0) * (i / n_per_alloy)
                v = v0 + (v1 - v0) * (i / n_per_alloy)
                cases.append(
                    SurrogateCase(
                        alloy=alloy,
                        source="synthetic",
                        case_id=f"{alloy}_{i}",
                        power=p,
                        velocity=v,
                        beam_diameter=100e-6,
                        layer_thickness=0.0,
                        width=150e-6 + 50e-6 * (i / n_per_alloy),
                        depth=80e-6 + 40e-6 * (i / n_per_alloy),
                        length=float("nan"),
                    )
                )
        return cases

    def test_fit_and_predict(self):
        cases = self._make_cases()
        surr = GlobalPhysicsInformedSurrogate(absorptivity=0.5, n_restarts=1)
        surr.fit(cases)
        # 10 features: 4 physics + 5 material properties + source indicator
        assert len(surr.feature_names) == 10
        pred = surr.predict(200.0, 1.0, 100e-6, "316L")
        assert pred.width > 0
        assert pred.depth > 0
        assert pred.width_std >= 0
        assert pred.depth_std >= 0
        assert pred.mode in ("conduction", "keyhole")

    def test_predict_requires_fit(self):
        surr = GlobalPhysicsInformedSurrogate()
        with pytest.raises(RuntimeError):
            surr.predict(200.0, 1.0, 100e-6, "316L")

    def test_cross_alloy_prediction(self):
        cases = self._make_cases()
        surr = GlobalPhysicsInformedSurrogate(absorptivity=0.5, n_restarts=1)
        surr.fit(cases)
        # Predict on the other alloy (IN718) -- should still work
        pred = surr.predict(200.0, 1.0, 100e-6, "IN718")
        assert pred.width > 0
        assert pred.depth > 0

    def test_no_melt_returns_nan(self):
        cases = self._make_cases()
        surr = GlobalPhysicsInformedSurrogate(absorptivity=0.5, n_restarts=1)
        surr.fit(cases)
        pred = surr.predict(1.0, 10.0, 100e-6, "316L")
        assert pred.mode == "no_melt"
        assert math.isnan(pred.width)

    def test_calibrate_conformal(self):
        cases = self._make_cases()
        surr = GlobalPhysicsInformedSurrogate(absorptivity=0.5, n_restarts=1)
        surr.fit(cases)
        assert surr.width_cal_factor == pytest.approx(1.0)
        surr.calibrate_conformal(cases, alpha=0.05)
        # The factor may be < 1 (over-conservative GP) or > 1 (under-conservative);
        # the key property is that it is positive, finite, and re-scales the std.
        assert surr.width_cal_factor > 0
        assert math.isfinite(surr.width_cal_factor)
        assert surr.depth_cal_factor > 0
        assert math.isfinite(surr.depth_cal_factor)


# ---------------------------------------------------------------------------
# Absorptivity calibration
# ---------------------------------------------------------------------------


class TestAbsorptivityCalibration:
    def test_bounds_respected(self):
        cases = load_unified_dataset(include_in718_nist=False, include_in718_pramod=False, include_in718_meltpoolnet=False, include_in718_chen2020=False, include_ti64=False, include_alsi10mg=False)
        mat = get_material("316L")
        result = calibrate_absorptivity(cases, mat, alloy_key="316L", n_grid=11)
        lo, hi = LITERATURE_ABSORPTIVITY_BOUNDS["316L"]
        assert lo <= result["absorptivity"] <= hi
        assert result["rmse"] >= 0
        assert result["n_cases"] > 0

    def test_unknown_alloy_requires_explicit_bounds(self):
        cases = load_unified_dataset(include_in718_nist=False, include_in718_pramod=False, include_in718_meltpoolnet=False, include_in718_chen2020=False, include_ti64=False, include_alsi10mg=False)
        mat = get_material("316L")
        with pytest.raises(ValueError):
            calibrate_absorptivity(cases, mat, alloy_key="UnknownAlloy", n_grid=5)

    def test_explicit_bounds_override(self):
        cases = load_unified_dataset(include_in718_nist=False, include_in718_pramod=False, include_in718_meltpoolnet=False, include_in718_chen2020=False, include_ti64=False, include_alsi10mg=False)
        mat = get_material("316L")
        result = calibrate_absorptivity(cases, mat, alloy_key="316L", bounds=(0.4, 0.6), n_grid=5)
        assert 0.4 <= result["absorptivity"] <= 0.6


# ---------------------------------------------------------------------------
# Bootstrap regression metrics
# ---------------------------------------------------------------------------


class TestBootstrapRegressionMetrics:
    def test_perfect_predictions_give_r2_near_one(self):
        y_true = np.linspace(1.0, 10.0, 30)
        result = bootstrap_regression_metrics(y_true, y_true.copy(), n_boot=200)
        assert result["r2_lo"] > 0.99
        assert result["rmse_hi"] < 1e-6

    def test_ci_bounds_ordered(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(size=40)
        y_pred = y_true + rng.normal(scale=0.5, size=40)
        result = bootstrap_regression_metrics(y_true, y_pred, n_boot=300)
        assert result["r2_lo"] <= result["r2_hi"]
        assert result["rmse_lo"] <= result["rmse_hi"]

    def test_too_few_points_returns_nan(self):
        result = bootstrap_regression_metrics(np.array([1.0]), np.array([1.0]), n_boot=50)
        assert math.isnan(result["r2_lo"])


# ---------------------------------------------------------------------------
# Goldak double-ellipsoidal moving heat source
# ---------------------------------------------------------------------------


class TestGoldak:
    def test_point_source_limit_matches_rosenthal(self):
        from rosenthal.goldak import GoldakParameters, temperature as goldak_temperature
        from rosenthal.model import ProcessParameters, temperature as rosenthal_temperature

        mat = get_material("316L")
        gp = GoldakParameters(power=200.0, velocity=1.0, absorptivity=0.4, a=1e-6, b=1e-6, c_front=1e-6, c_rear=1e-6)
        rp = ProcessParameters(power=200.0, velocity=1.0, absorptivity=0.4)
        for x, y, z in [(50e-6, 30e-6, 20e-6), (-80e-6, 0.0, 0.0), (0.0, 60e-6, 0.0), (0.0, 0.0, 50e-6)]:
            tg = goldak_temperature(x, y, z, gp, mat)
            tr = rosenthal_temperature(x, y, z, rp, mat)
            assert tg == pytest.approx(tr, rel=0.01)

    def test_aspect_ratio_not_fixed_at_half(self):
        from rosenthal.goldak import GoldakParameters, melt_pool_dimensions

        mat = get_material("316L")
        wide = GoldakParameters(power=200.0, velocity=1.0, absorptivity=0.4, a=60e-6, b=20e-6, c_front=40e-6, c_rear=160e-6)
        deep = GoldakParameters(power=200.0, velocity=1.0, absorptivity=0.4, a=20e-6, b=60e-6, c_front=40e-6, c_rear=160e-6)
        d_wide = melt_pool_dimensions(wide, mat)
        d_deep = melt_pool_dimensions(deep, mat)
        assert d_wide["depth"] / d_wide["width"] != pytest.approx(0.5, abs=0.02)
        assert d_deep["depth"] / d_deep["width"] > d_wide["depth"] / d_wide["width"]

    def test_f_front_continuity_default(self):
        from rosenthal.goldak import GoldakParameters

        gp = GoldakParameters(power=200.0, velocity=1.0, absorptivity=0.4, a=1e-5, b=1e-5, c_front=1e-5, c_rear=4e-5)
        assert gp.f_front_value == pytest.approx(2 * 1e-5 / (1e-5 + 4e-5))
        assert gp.f_front_value + gp.f_rear_value == pytest.approx(2.0)

    def test_f_front_explicit_override(self):
        from rosenthal.goldak import GoldakParameters

        gp = GoldakParameters(
            power=200.0, velocity=1.0, absorptivity=0.4, a=1e-5, b=1e-5, c_front=1e-5, c_rear=4e-5, f_front=0.6
        )
        assert gp.f_front_value == pytest.approx(0.6)
        assert gp.f_rear_value == pytest.approx(1.4)

    def test_no_melt_raises(self):
        from rosenthal.goldak import GoldakParameters, melt_pool_dimensions

        mat = get_material("316L")
        gp = GoldakParameters(power=0.5, velocity=10.0, absorptivity=0.1, a=30e-6, b=30e-6, c_front=20e-6, c_rear=80e-6)
        with pytest.raises(ValueError):
            melt_pool_dimensions(gp, mat)


class TestGoldakCalibration:
    def test_calibrate_goldak_shape_from_synthetic_cases(self):
        from rosenthal.goldak import calibrate_goldak_shape
        from rosenthal.physics_informed import SurrogateCase

        cases = [
            SurrogateCase(
                alloy="316L",
                source="synthetic",
                case_id=f"s{i}",
                power=200.0,
                velocity=1.0,
                beam_diameter=100e-6,
                layer_thickness=0.0,
                width=150e-6 + i * 1e-6,
                depth=70e-6 + i * 1e-6,
                length=float("nan"),
            )
            for i in range(10)
        ]
        mat = get_material("316L")
        result = calibrate_goldak_shape(cases, mat, absorptivity=0.5)
        assert result["a"] == pytest.approx(np.mean([c.width for c in cases]) / 2.0)
        assert result["b"] == pytest.approx(np.mean([c.depth for c in cases]))
        assert result["c_rear"] == pytest.approx(4.0 * result["c_front"])
        assert result["n_cases"] == 10


class TestGoldakFastPath:
    def test_temperature_batch_matches_adaptive(self):
        from rosenthal.goldak import GoldakParameters, temperature, temperature_batch

        mat = get_material("316L")
        gp = GoldakParameters(power=200.0, velocity=1.0, absorptivity=0.4, a=60e-6, b=30e-6, c_front=40e-6, c_rear=160e-6)
        pts = [(50e-6, 30e-6, 20e-6), (-80e-6, 0.0, 0.0), (0.0, 60e-6, 0.0), (0.0, 0.0, 50e-6)]
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        zs = np.array([p[2] for p in pts])
        adaptive = np.array([temperature(x, y, z, gp, mat) for x, y, z in pts])
        batch = temperature_batch(xs, ys, zs, gp, mat)
        assert batch == pytest.approx(adaptive, rel=1e-3)

    def test_width_depth_fast_matches_adaptive_dimensions(self):
        from rosenthal.goldak import GoldakParameters, melt_pool_dimensions, width_depth_fast

        mat = get_material("316L")
        gp = GoldakParameters(power=200.0, velocity=1.0, absorptivity=0.4, a=60e-6, b=30e-6, c_front=40e-6, c_rear=160e-6)
        w_fast, d_fast = width_depth_fast(gp, mat)
        d_adaptive = melt_pool_dimensions(gp, mat)
        assert w_fast == pytest.approx(d_adaptive["width"], rel=1e-2)
        assert d_fast == pytest.approx(d_adaptive["depth"], rel=1e-2)

    def test_calibrate_goldak_depth_only_fixes_a(self):
        from rosenthal.goldak import calibrate_goldak_depth_only, calibrate_goldak_shape
        from rosenthal.physics_informed import SurrogateCase

        mat = get_material("316L")
        cases = [
            SurrogateCase(
                alloy="316L",
                source="synthetic",
                case_id=f"s{i}",
                power=150.0 + 10 * i,
                velocity=0.8 + 0.05 * i,
                beam_diameter=100e-6,
                layer_thickness=0.0,
                width=140e-6 + 2e-6 * i,
                depth=60e-6 + 1e-6 * i,
                length=float("nan"),
            )
            for i in range(12)
        ]
        geom = calibrate_goldak_shape(cases, mat, absorptivity=0.5)
        result = calibrate_goldak_depth_only(cases, mat, absorptivity=0.5, n_calibration_cases=8)
        assert result["a"] == pytest.approx(geom["a"])
        assert result["c_front"] == pytest.approx(geom["a"])
        assert result["c_rear"] == pytest.approx(4.0 * geom["a"])
        assert result["b"] > 0


class TestGoldakModeAware:
    def _make_bimodal_cases(self, mat_name="316L"):
        from rosenthal.physics_informed import SurrogateCase

        cases = []
        # Conduction-mode-like: low power/high velocity, shallow wide pools.
        for i in range(12):
            cases.append(
                SurrogateCase(
                    alloy=mat_name,
                    source="synthetic",
                    case_id=f"cond{i}",
                    power=100.0 + 5 * i,
                    velocity=1.5 + 0.05 * i,
                    beam_diameter=100e-6,
                    layer_thickness=0.0,
                    width=150e-6 + 2e-6 * i,
                    depth=50e-6 + 1e-6 * i,
                    length=float("nan"),
                )
            )
        # Keyhole-mode-like: high power/low velocity, deep narrow pools.
        for i in range(12):
            cases.append(
                SurrogateCase(
                    alloy=mat_name,
                    source="synthetic",
                    case_id=f"key{i}",
                    power=350.0 + 5 * i,
                    velocity=0.3 + 0.02 * i,
                    beam_diameter=100e-6,
                    layer_thickness=0.0,
                    width=120e-6 + 2e-6 * i,
                    depth=140e-6 + 2e-6 * i,
                    length=float("nan"),
                )
            )
        return cases

    def test_mode_aware_true_with_enough_data(self):
        from rosenthal.goldak import calibrate_goldak_mode_aware

        mat = get_material("316L")
        cases = self._make_bimodal_cases()
        result = calibrate_goldak_mode_aware(cases, mat, absorptivity=0.5, min_cases_per_mode=8)
        assert result["mode_aware"] is True
        assert result["n_conduction"] >= 8
        assert result["n_keyhole"] >= 8
        assert 0.0 <= result["train_accuracy"] <= 1.0

    def test_mode_aware_falls_back_with_too_little_data(self):
        from rosenthal.goldak import calibrate_goldak_mode_aware

        mat = get_material("316L")
        cases = self._make_bimodal_cases()[:5]  # too few for either mode
        result = calibrate_goldak_mode_aware(cases, mat, absorptivity=0.5, min_cases_per_mode=8)
        assert result["mode_aware"] is False
        assert result["shape_conduction"] == result["shape_keyhole"]

    def test_predict_mode_aware_classifies_and_predicts(self):
        from rosenthal.goldak import calibrate_goldak_mode_aware, predict_mode_aware

        mat = get_material("316L")
        cases = self._make_bimodal_cases()
        cal = calibrate_goldak_mode_aware(cases, mat, absorptivity=0.5, min_cases_per_mode=8)

        # A representative (high-energy) point should classify and predict without error.
        pred = predict_mode_aware(370.0, 0.32, 100e-6, mat, 0.5, cal)
        assert pred["mode"] in ("conduction", "keyhole")
        assert pred["width"] > 0
        assert pred["depth"] > 0

    def test_fit_mode_threshold_reasonable_accuracy(self):
        from rosenthal.goldak import fit_mode_threshold

        mat = get_material("316L")
        cases = self._make_bimodal_cases()
        result = fit_mode_threshold(cases, mat, absorptivity=0.5)
        # These synthetic cases are cleanly separated by power/velocity, so
        # normalized enthalpy should separate them well.
        assert result["train_accuracy"] > 0.8


class TestGoldakModeAwareSurrogateIntegration:
    def test_surrogate_with_goldak_mode_aware_baseline(self):
        from rosenthal.goldak import calibrate_goldak_mode_aware
        from rosenthal.physics_informed import SurrogateCase

        mat = get_material("316L")
        cases = []
        for i in range(12):
            cases.append(
                SurrogateCase(
                    alloy="316L",
                    source="synthetic",
                    case_id=f"cond{i}",
                    power=100.0 + 5 * i,
                    velocity=1.5 + 0.05 * i,
                    beam_diameter=100e-6,
                    layer_thickness=0.0,
                    width=150e-6 + 2e-6 * i,
                    depth=50e-6 + 1e-6 * i,
                    length=float("nan"),
                )
            )
        for i in range(12):
            cases.append(
                SurrogateCase(
                    alloy="316L",
                    source="synthetic",
                    case_id=f"key{i}",
                    power=350.0 + 5 * i,
                    velocity=0.3 + 0.02 * i,
                    beam_diameter=100e-6,
                    layer_thickness=0.0,
                    width=120e-6 + 2e-6 * i,
                    depth=140e-6 + 2e-6 * i,
                    length=float("nan"),
                )
            )
        mc = calibrate_goldak_mode_aware(cases, mat, absorptivity=0.5, min_cases_per_mode=8)
        assert mc["mode_aware"] is True

        surrogate = PhysicsInformedSurrogate(material=mat, absorptivity=0.5, goldak_mode_aware=mc, n_restarts=1)
        surrogate.fit(cases)
        pred = surrogate.predict(370.0, 0.32, 100e-6)
        assert not math.isnan(pred.width)
        assert not math.isnan(pred.depth)
        assert pred.width > 0
        assert pred.depth > 0

    def test_rosenthal_baseline_requires_beam_diameter_for_mode_aware(self):
        mat = get_material("316L")
        surrogate = PhysicsInformedSurrogate(material=mat, absorptivity=0.5, goldak_mode_aware={"threshold": 10.0})
        with pytest.raises(ValueError):
            surrogate._rosenthal_baseline(200.0, 1.0)


class TestChen2020InconelPreheat:
    def test_load_default_preheat(self):
        from rosenthal.data.chen2020_in718_preheat import load_chen2020_in718

        cases = load_chen2020_in718()
        assert len(cases) == 16
        assert all(c.preheat_c == 100 for c in cases)
        assert all(c.measured_width > 0 and c.measured_depth > 0 for c in cases)
        assert all(c.beam_diameter == pytest.approx(100e-6) for c in cases)

    def test_load_invalid_preheat_raises(self):
        from rosenthal.data.chen2020_in718_preheat import load_chen2020_in718

        with pytest.raises(ValueError):
            load_chen2020_in718(preheat_c=999)

    def test_load_other_preheat_levels(self):
        from rosenthal.data.chen2020_in718_preheat import load_chen2020_in718

        for t in (200, 300, 400, 500):
            cases = load_chen2020_in718(preheat_c=t)
            assert len(cases) == 16
            assert all(c.preheat_c == t for c in cases)

    def test_wired_into_unified_dataset(self):
        cases = load_unified_dataset()
        chen = [c for c in cases if c.source == "Chen2020_preheat100C"]
        assert len(chen) == 16
        assert all(c.alloy == "IN718" for c in chen)

    def test_in718_real_majority_with_balancing(self):
        from rosenthal.physics_informed import _is_experimental_source

        cases = load_unified_dataset(balance_in718_sources=True)
        in718 = [c for c in cases if c.alloy == "IN718"]
        real = sum(1 for c in in718 if _is_experimental_source(c.source))
        assert real > len(in718) - real
