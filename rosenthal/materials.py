"""Thermophysical property presets for common L-PBF alloys.

Properties are temperature-averaged (constant-property) approximations, which is the
standard simplifying assumption required by the Rosenthal analytical solution (it has
no closed form for temperature-dependent k, rho, or Cp). Values are drawn from Mills,
K.C. (2002), "Recommended Values of Thermophysical Properties for Selected Commercial
Alloys," Woodhead Publishing -- the compilation most widely cited in the AM melt-pool
modelling literature for exactly these four alloys.

Solidus/liquidus ranges are real; a single T_melt is taken as the liquidus (the
temperature above which the material is fully molten) since that is the boundary the
Rosenthal isotherm is normally compared against in single-track validation studies.

Before using these in a real analysis, cross-check against the specific composition
and temperature range in your own source -- these are starting points, not certified
values for a specific certified feedstock lot.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """Constant thermophysical properties for one alloy.

    Attributes:
        name: Human-readable alloy name.
        k: Thermal conductivity (W / m / K).
        rho: Density (kg / m^3).
        cp: Specific heat capacity (J / kg / K).
        t_melt: Liquidus temperature (K), used as the melt-pool boundary isotherm.
        source: Citation for the property values.
    """

    name: str
    k: float
    rho: float
    cp: float
    t_melt: float
    source: str

    @property
    def alpha(self) -> float:
        """Thermal diffusivity (m^2 / s)."""
        return self.k / (self.rho * self.cp)


MATERIALS: dict[str, Material] = {
    "Ti-6Al-4V": Material(
        name="Ti-6Al-4V",
        k=6.7,
        rho=4430.0,
        cp=526.0,
        t_melt=1928.0,
        source="Mills (2002), Table for Ti-6Al-4V",
    ),
    "316L": Material(
        name="316L Stainless Steel",
        k=21.5,
        rho=8000.0,
        cp=500.0,
        t_melt=1710.0,
        source="Mills (2002), Table for 316L stainless steel",
    ),
    "AlSi10Mg": Material(
        name="AlSi10Mg",
        k=113.0,
        rho=2670.0,
        cp=900.0,
        t_melt=870.0,
        source="Mills (2002), Table for Al-Si casting alloys",
    ),
    "IN718": Material(
        name="Inconel 718",
        k=11.4,
        rho=8190.0,
        cp=435.0,
        t_melt=1609.0,
        source="Mills (2002), Table for Inconel 718",
    ),
}


def get_material(name: str) -> Material:
    """Look up a preset material by name.

    Raises:
        KeyError: if `name` is not one of the preset materials in MATERIALS.
    """
    try:
        return MATERIALS[name]
    except KeyError as exc:
        available = ", ".join(MATERIALS)
        raise KeyError(f"Unknown material {name!r}. Available: {available}") from exc
