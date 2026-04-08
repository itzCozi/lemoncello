"""
Limoncello Recipe Optimizer
============================
Finds the globally optimal limoncello recipe using psychophysical sensory
models, chemical extraction kinetics, and consumer preference curves.

Sensory models:
  - Two-compartment first-order kinetics for lemon oil extraction
  - Sigmoidal limonin diffusion for pith bitterness
  - Stevens' power law for ethanol burn perception
  - TRPV1 receptor cold-gating for temperature-dependent burn reduction
  - Weber-Fechner law for sweetness perception
  - Michaelis-Menten saturation for lemon oil flavor intensity

Run:
    python example.py 2>&1 | Tee-Object output.txt
"""

import sys
import json
import numpy as np
from math import exp, log

sys.path.insert(0, ".")
from framework import RecipeOptimizer, Variable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OIL_PER_LEMON = 0.50       # mL essential oil per well-zested lemon
SERVING_TEMP_C = 0          # served from the freezer (0C)
TARGET_ABV = 0.28           # traditional limoncello
TARGET_SWEET = 5.2          # perceived sweetness target (0-10 scale)
TARGET_BRIX = 27            # traditional range 25-30

# ---------------------------------------------------------------------------
# 1. Define variables
# ---------------------------------------------------------------------------
variables = [
    Variable("lemons", 4, 18, integer=True, unit="count"),
    Variable("spirit_abv", 0.40, 0.96, unit="fraction"),
    Variable("spirit_vol", 300, 1500, unit="mL", rounding=5),
    Variable("days", 2, 45, unit="days", rounding=1),
    Variable("sugar_g", 200, 1500, unit="g", rounding=5),
    Variable("water_ml", 200, 2500, unit="mL", rounding=5),
]

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------
WEIGHTS = {
    "taste":        0.30,
    "drinkability": 0.30,
    "ease":         0.20,
    "strength":     0.10,
    "authenticity": 0.10,
}


# ---------------------------------------------------------------------------
# 2. Scoring function
# ---------------------------------------------------------------------------
def score(v: dict) -> tuple[float, dict]:
    lemons = v["lemons"]
    abv    = v["spirit_abv"]
    vol    = v["spirit_vol"]
    days   = v["days"]
    sugar  = v["sugar_g"]
    water  = v["water_ml"]

    # Bounds guard: Nelder-Mead can escape variable bounds.
    for var in variables:
        if v[var.name] < var.low - 1 or v[var.name] > var.high + 1:
            return 0.0, {k: 0.0 for k in WEIGHTS}

    # ===================================================================
    # PHYSICAL CALCULATIONS
    # ===================================================================

    # -- Oil extraction: two-compartment first-order kinetics --
    #
    # Lemon zest has two oil reservoirs:
    #   Surface glands (60%): burst on zesting, dissolve fast
    #   Deep flavedo (40%): solvent must diffuse in, dissolve slow
    #
    # Rate constants scale with ABV^1.5 because d-limonene (the dominant
    # monoterpene, ~95% of lemon oil) is lipophilic and its solubility
    # in ethanol-water mixtures rises nonlinearly with ethanol fraction.
    rate = (abv / 0.96) ** 1.5
    k_fast = 0.80 * rate          # surface oil: ~1 day half-life at 96%
    k_slow = 0.08 * rate          # deep oil: ~9 day half-life at 96%
    extraction = (0.60 * (1 - exp(-k_fast * max(days, 0)))
               + 0.40 * (1 - exp(-k_slow * max(days, 0))))
    total_oil = lemons * OIL_PER_LEMON * extraction   # mL

    # -- Volumes and concentrations --
    sugar_vol  = sugar * 0.63                           # mL when dissolved
    total_vol  = vol + water + sugar_vol                # mL total
    total_vol  = max(total_vol, 1)                      # guard
    oil_conc   = total_oil / total_vol * 1000           # mL/L
    final_abv  = (abv * vol) / total_vol

    # -- Brix --
    # Approximate Brix from the mass ratio of sugar to total mixture.
    spirit_density = abv * 0.789 + (1 - abv)  # g/mL weighted average
    total_mass = vol * spirit_density + water + sugar
    brix = sugar / max(total_mass, 1) * 100

    # -- Pith bitterness: limonin diffusion --
    #
    # Even careful zesting includes trace pith. Limonin is a triterpenoid
    # dilactone that diffuses from pith fragments into the spirit. Modeled
    # as a sigmoid: negligible at short times, rising toward a ceiling.
    # Higher ABV accelerates limonin extraction (better solvent for
    # triterpenoids), shifting the midpoint earlier.
    t_mid = 35 - 18 * abv   # days to 50% bitterness extraction
    bitterness = 1 / (1 + exp(-0.18 * (days - t_mid)))

    # -- Complexity --
    #
    # Longer infusion extracts minor terpenes (beta-pinene, gamma-terpinene,
    # citral, geraniol) beyond the dominant d-limonene. These add aromatic
    # depth. Logarithmic diminishing returns.
    complexity = min(10, 4 + 2.5 * log(1 + max(days, 0) / 5))

    # ===================================================================
    # TASTE (weight 0.30)
    # ===================================================================
    #
    # Weighted sum of four sensory components.

    # Oil intensity: Michaelis-Menten receptor saturation.
    # Km = 0.5 mL/L means half-max perceived intensity at 0.5 mL/L.
    # Above ~2 mL/L, diminishing returns (saturated olfactory receptors).
    oil_intensity = 10 * oil_conc / (oil_conc + 0.5)

    # Sweetness: Weber-Fechner law.
    # Perceived sweetness = k * ln(1 + Brix).
    # k = 1.56 calibrated so that 27 Brix maps to ~5.2 perceived sweetness.
    perceived_sweet = 1.56 * log(1 + max(brix, 0))

    # Acid suppression of sweetness: citric acid from lemon oil causes
    # modest cross-modal suppression of perceived sweetness.
    acid_factor = max(0.7, 1 - 0.08 * oil_conc)
    perceived_sweet *= acid_factor

    # Preference peak for sweetness: Gaussian centered at TARGET_SWEET.
    sweet_pref = 10 * exp(-0.5 * ((perceived_sweet - TARGET_SWEET) / 0.9) ** 2)

    # Bitterness cleanliness: 10 = no bitterness, 0 = fully bitter
    clean = 10 * (1 - bitterness)

    taste = (0.40 * oil_intensity
           + 0.25 * sweet_pref
           + 0.20 * clean
           + 0.15 * complexity)
    taste = float(np.clip(taste, 0, 10))

    # ===================================================================
    # DRINKABILITY (weight 0.30)
    # ===================================================================
    #
    # "Would you pour a second glass?" Geometric mean of five components,
    # so one weak dimension drags the whole score down (as in real life).

    # Burn perception: Stevens' power law + TRPV1 cold-gating.
    #
    # Stevens' power law: psi = k * C^n, n ~= 1.3 for ethanol.
    # TRPV1 ion channels are the primary ethanol/heat nociceptor.
    # At 0C, TRPV1 is well below its activation threshold (~42C for heat,
    # lowered by ethanol to ~34C). Cold also activates TRPM8 (menthol
    # receptor), which has an analgesic effect. Net result: perceived burn
    # drops by ~75% when served from the freezer.
    burn_room = 10 * (max(final_abv, 0) / 0.50) ** 1.3
    cold_factor = 0.25 + 0.75 * (SERVING_TEMP_C / 40) ** 2
    burn = burn_room * max(cold_factor, 0.25)

    # Burn comfort: low burn is good. Gaussian centered at burn=1.5.
    burn_comfort = 10 * exp(-0.15 * (burn - 1.5) ** 2)

    # Sweetness pleasantness
    sweet_score = 10 * exp(-0.18 * (perceived_sweet - 5.5) ** 2)

    # Flavor presence (same as oil intensity)
    flavor_score = oil_intensity

    # Mouthfeel: sugar and alcohol contribute viscosity/body.
    # Saturating function of Brix.
    mouthfeel = 10 * (1 - exp(-0.08 * max(brix, 0)))

    # Cleanliness: linear penalty from bitterness
    cleanliness = 10 * (1 - 0.7 * bitterness)

    # Geometric mean
    components = [burn_comfort, sweet_score, flavor_score, mouthfeel, cleanliness]
    geo_mean = float(np.prod([max(c, 0.001) for c in components]) ** (1 / len(components)))
    drinkability = float(np.clip(geo_mean, 0, 10))

    # ===================================================================
    # EASE (weight 0.20)
    # ===================================================================
    #
    # Vodka (40%) is available at any liquor store. Overproof (60-70%)
    # requires searching. Grain alcohol (95-96%) is specialty or illegal
    # in some US states.
    if abv < 0.50:
        spirit_ease = 10
    elif abv < 0.70:
        spirit_ease = 7
    else:
        spirit_ease = 5

    wait_ease = 10 * exp(-0.015 * max(days, 0))
    quantity_ease = 10 if lemons <= 12 else max(5, 10 - 0.4 * (lemons - 12))

    ease = float(np.clip((spirit_ease + wait_ease + quantity_ease) / 3, 0, 10))

    # ===================================================================
    # STRENGTH (weight 0.10)
    # ===================================================================
    #
    # Traditional limoncello: 28% ABV. Tight Gaussian preference.
    strength = float(10 * exp(-0.5 * ((final_abv - TARGET_ABV) / 0.03) ** 2))

    # ===================================================================
    # AUTHENTICITY (weight 0.10)
    # ===================================================================
    #
    # Does it match what you would find in a bar on the Amalfi Coast?
    # ABV in range, Brix in range, strong lemon oil presence.
    abv_auth  = exp(-0.5 * ((final_abv - 0.30) / 0.05) ** 2)
    brix_auth = exp(-0.5 * ((brix - TARGET_BRIX) / 5) ** 2)
    oil_auth  = min(1.0, oil_conc / 1.5)
    authenticity = float(10 * np.clip(
        (abv_auth * brix_auth * oil_auth) ** (1/3), 0, 1
    ))

    # ===================================================================
    # COMPOSITE
    # ===================================================================
    scores = {
        "taste":        taste,
        "drinkability": drinkability,
        "ease":         ease,
        "strength":     strength,
        "authenticity": authenticity,
    }
    composite = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)

    return composite, scores


# ---------------------------------------------------------------------------
# 3. Sanity checks
# ---------------------------------------------------------------------------
def sanity_check():
    """Test a few known recipes to verify the scoring function produces
    varied, reasonable scores before handing it to the optimizer."""
    test_recipes = [
        # Reasonable baseline
        {"lemons": 8, "spirit_abv": 0.95, "spirit_vol": 750, "days": 7,
         "sugar_g": 700, "water_ml": 1000},
        # Vodka-based, short infusion
        {"lemons": 6, "spirit_abv": 0.40, "spirit_vol": 1200, "days": 4,
         "sugar_g": 600, "water_ml": 400},
        # Overextracted: too many lemons, too long, lots of bitterness
        {"lemons": 16, "spirit_abv": 0.96, "spirit_vol": 1200, "days": 35,
         "sugar_g": 1200, "water_ml": 2000},
        # Near-optimal guess
        {"lemons": 10, "spirit_abv": 0.90, "spirit_vol": 900, "days": 10,
         "sugar_g": 800, "water_ml": 1400},
    ]
    print("Sanity check (should produce varied, reasonable scores):\n")
    for r in test_recipes:
        sc, subs = score(r)
        parts = "  ".join(f"{k}={v:.1f}" for k, v in subs.items())
        print(f"  {sc:.2f}   {parts}")
        print(f"         {r}")
    print()


# ---------------------------------------------------------------------------
# 4. Diagnostics
# ---------------------------------------------------------------------------
def print_diagnostics(v: dict):
    """Print the physical properties of a recipe for verification."""
    abv    = v["spirit_abv"]
    vol    = v["spirit_vol"]
    days   = v["days"]
    sugar  = v["sugar_g"]
    water  = v["water_ml"]
    lemons = v["lemons"]

    rate = (abv / 0.96) ** 1.5
    k_fast = 0.80 * rate
    k_slow = 0.08 * rate
    extraction = 0.60 * (1 - exp(-k_fast * days)) + 0.40 * (1 - exp(-k_slow * days))
    total_oil = lemons * OIL_PER_LEMON * extraction
    total_vol = vol + water + sugar * 0.63
    oil_conc = total_oil / total_vol * 1000
    final_abv = abv * vol / total_vol
    spirit_density = abv * 0.789 + (1 - abv)
    total_mass = vol * spirit_density + water + sugar
    brix = sugar / total_mass * 100
    t_mid = 35 - 18 * abv
    bitterness = 1 / (1 + exp(-0.18 * (days - t_mid)))
    burn_room = 10 * (final_abv / 0.50) ** 1.3
    burn = burn_room * 0.25
    perceived_sweet = 1.56 * log(1 + brix) * max(0.7, 1 - 0.08 * oil_conc)

    print(f"\n  WINNER DIAGNOSTICS")
    print(f"    Oil extraction:    {extraction*100:.1f}%")
    print(f"    Total oil:         {total_oil:.2f} mL")
    print(f"    Oil concentration: {oil_conc:.2f} mL/L")
    print(f"    Total volume:      {total_vol:.0f} mL")
    print(f"    Final ABV:         {final_abv*100:.1f}%")
    print(f"    Brix:              {brix:.1f}")
    print(f"    Bitterness:        {bitterness:.3f} (midpoint day {t_mid:.0f})")
    print(f"    Burn at 0C:        {burn:.2f}/10")
    print(f"    Perceived sweet:   {perceived_sweet:.2f}/10")


# ---------------------------------------------------------------------------
# 5. Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sanity_check()

    opt = RecipeOptimizer("Limoncello", variables, score)
    result = opt.run(de_restarts=15, de_popsize=50, de_maxiter=3000)

    print_diagnostics(result["rounded"])

    # Sensitivity analysis
    print(f"\n  SENSITIVITY (+/-5% perturbation)")
    sens = opt.sensitivity(result["rounded"])
    for name, info in sens.items():
        print(f"    {name:12s}: {info['sensitivity']:+.4f}  "
              f"(score {info['score_at_low']:.3f} to {info['score_at_high']:.3f})")

    # Save result
    output = {
        "recipe": result["rounded"],
        "composite_score": round(result["composite"], 6),
        "criterion_scores": {k: round(v, 4) for k, v in result["scores"].items()},
        "stats": result["stats"],
    }
    with open("result.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Result saved to result.json")
