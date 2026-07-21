"""Generate process-map figures (width, depth, aspect ratio vs power & velocity)
for every preset material. These are the Project 2 figures referenced in Paper 2.

Run: python scripts/generate_process_maps.py
Output: figures/process_map_<material>.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rosenthal import MATERIALS, get_material, sweep

POWERS = np.linspace(50, 400, 45)
VELOCITIES_MM_S = np.linspace(200, 2000, 45)
ABSORPTIVITY = 0.4
T0 = 300.0

OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


def plot_material(material_name: str) -> None:
    material = get_material(material_name)
    velocities = VELOCITIES_MM_S / 1000.0
    result = sweep(material, POWERS, velocities, absorptivity=ABSORPTIVITY, t0=T0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"{material.name} — process map (absorptivity={ABSORPTIVITY}, T0={T0:.0f} K)", fontweight="bold")

    panels = [
        ("width", result["width"] * 1e6, "Width (µm)"),
        ("depth", result["depth"] * 1e6, "Depth (µm)"),
        ("aspect_ratio", result["aspect_ratio"], "Depth / width ratio"),
    ]

    for ax, (_, data, label) in zip(axes, panels):
        im = ax.pcolormesh(POWERS, VELOCITIES_MM_S, data, shading="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("Power (W)")
        ax.set_ylabel("Velocity (mm/s)")
        ax.set_title(label)

    # Overlay the conduction/keyhole boundary (aspect ratio = 1) on all panels.
    aspect = result["aspect_ratio"]
    for ax in axes:
        ax.contour(POWERS, VELOCITIES_MM_S, aspect, levels=[1.0], colors="red", linewidths=1.5)

    fig.tight_layout()
    out_path = OUT_DIR / f"process_map_{material_name.replace('-', '').replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    for name in MATERIALS:
        plot_material(name)


if __name__ == "__main__":
    main()
