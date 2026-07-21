"""Generate the example figures used in the Project 1 documentation PDF.

Mirrors the Streamlit app's plan-view plotting logic exactly (same clipping,
same auto-fit zoom, same fixed rendered figure size) so the "snapshots" in the
documentation are faithful to what the deployed calculator actually shows.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rosenthal import ProcessParameters, get_material, melt_pool_dimensions, temperature

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    }
)


def plan_view_figure(material_name: str, power: float, velocity: float, absorptivity: float, t0: float, out_name: str, title: str):
    material = get_material(material_name)
    params = ProcessParameters(power=power, velocity=velocity, absorptivity=absorptivity, t0=t0)
    dims = melt_pool_dimensions(params, material)

    delta = material.t_melt - t0
    vmax_display = t0 + 2.2 * delta
    levels = np.linspace(t0, vmax_display, 60)

    half_w = dims["width"] / 2 * 1.6
    x_back = dims["length_back"] * 0.35
    x_front = dims["length_front"] * 1.8

    x = np.linspace(-x_back, x_front, 220)
    y = np.linspace(-half_w, half_w, 220)
    A, B = np.meshgrid(x, y)
    T = np.vectorize(lambda a, b: temperature(a, b, 0.0, params, material) if (a, b) != (0.0, 0.0) else np.nan)(A, B)
    T = np.clip(T, t0, vmax_display)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    cf = ax.contourf(x * 1e6, y * 1e6, T, levels=levels, cmap="inferno", extend="max")
    ax.contour(x * 1e6, y * 1e6, T, levels=[material.t_melt], colors="#00e5ff", linewidths=2.2)
    ax.plot(0, 0, marker="*", color="white", markersize=12, markeredgecolor="black", markeredgewidth=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x, scan direction (µm)")
    ax.set_ylabel("y (µm)")
    fig.colorbar(cf, ax=ax, label="Temperature (K)", shrink=0.7)
    fig.suptitle(title, fontsize=10, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}  dims(um)={ {k: round(v*1e6,1) for k,v in dims.items()} }")
    return dims


if __name__ == "__main__":
    # Example 1: default/typical operating point, Ti-6Al-4V
    plan_view_figure(
        "Ti-6Al-4V", power=200.0, velocity=0.8, absorptivity=0.4, t0=300.0,
        out_name="example1_ti64_default.png",
        title="Ti-6Al-4V: 200 W, 800 mm/s, absorptivity 0.4",
    )

    # Example 2: matches Zhang et al. (2024) validation case N01 (316L)
    plan_view_figure(
        "316L", power=260.0, velocity=0.52, absorptivity=0.5, t0=300.0,
        out_name="example2_316L_validation_N01.png",
        title="316L: 260 W, 520 mm/s, absorptivity 0.5 (Zhang et al. N01)",
    )

    # Example 3: high-conductivity material, wide shallow pool
    plan_view_figure(
        "AlSi10Mg", power=200.0, velocity=0.8, absorptivity=0.4, t0=300.0,
        out_name="example3_alsi10mg_default.png",
        title="AlSi10Mg: 200 W, 800 mm/s, absorptivity 0.4",
    )
