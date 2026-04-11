"""
Limoncello Recipe Optimizer
============================
Finds the globally optimal limoncello recipe using psychophysical sensory
models, chemical extraction kinetics, and consumer preference curves.

Sensory models:
  - Two-compartment first-order kinetics for lemon oil extraction
  - Sigmoidal d-limonene solubility model (Li & Tamura, 2010)
  - Arrhenius temperature dependence on extraction rates
  - Surface-area-dependent extraction from zest fineness
  - Six lemon variety profiles from published compositional data
    (Flamini et al., 2007; Ferrara et al., 2020; Sawamura et al., 2004)
  - Freshness-dependent oil volatility decay (first-order degradation)
  - Ethanol evaporation loss during infusion (temperature-dependent)
  - Monoterpene oxidation at extended infusion times (J. Agric. Food Chem.)
  - Sigmoidal limonin diffusion for pith bitterness (variety-scaled pith)
  - Sweetness-bitterness masking (Breslin & Beauchamp, 1997)
  - Louche quality model for oil-in-water emulsion aesthetics
  - Three-channel sensory model: orthonasal aroma, retronasal flavor, mouthfeel
  - Volatile partitioning: Henry's law for headspace aroma at serving temp
  - Oil-in-water emulsion stability (Stokes' law, ABV-dependent)
  - Stevens' power law + TRPV1/TRPM8 cold-gating for ethanol burn
  - Congener-mediated harshness from spirit quality
  - Post-mix ester formation and mellowing kinetics
  - Weber-Fechner law for sweetness perception
  - Viscosity model: sugar + ethanol + emulsion contributions

Run:
    python example.py 2>&1 | Tee-Object output.txt
"""

import os
import sys
import json

# Limit internal threading -- we use explicit process-level parallelism
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
from math import exp, log

sys.path.insert(0, ".")
from framework import RecipeOptimizer, Variable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_ABV = 0.28           # traditional limoncello
TARGET_SWEET = 5.2          # perceived sweetness target (0-10 scale)
TARGET_BRIX = 27            # traditional range 25-30
REF_TEMP_C = 20             # reference temperature for Arrhenius (room temp)

# ---------------------------------------------------------------------------
# Lemon variety profiles
# ---------------------------------------------------------------------------
# Published compositional data from J. Food Sci., Flavour & Fragrance J.,
# and Italian agricultural research (CRA-ACM). Values are per-lemon averages
# for well-grown fruit at commercial maturity.
#
# Varieties encoded as integers for the optimizer:
#   1 = Eureka/Lisbon    (US supermarket standard, year-round)
#   2 = Femminello/Sfusato (Amalfi coast, traditional limoncello lemon)
#   3 = Meyer             (sweeter, thinner peel, mandarin hybrid)
#   4 = Primofiore/Verdello (Italian seasonal, high acidity)
#   5 = Interdonato       (Sicilian hybrid, citron x lemon)
#   6 = Verna             (Spanish cultivar, widely grown in Europe)

VARIETY_PROFILES = {
    1: {  # Eureka / Lisbon
        "name": "Eureka/Lisbon",
        "oil_ml": 0.40,           # mL essential oil per lemon (moderate peel)
        "limonene_frac": 0.66,    # d-limonene fraction (Flamini et al., 2007: 59-73%)
        "citral_frac": 0.03,      # citral (geranial+neral): key lemon aroma
        "minor_terpenes": 0.31,   # beta-pinene ~12%, gamma-terpinene ~9%, myrcene ~2%, etc.
        "pith_thickness": 1.0,    # relative pith thickness (baseline)
        "acidity": 5.5,           # % citric acid in juice (not used in oil)
        "peel_brix": 2.0,         # sugars in peel (slight sweetness boost)
    },
    2: {  # Femminello / Sfusato Amalfitano
        "name": "Femminello",
        "oil_ml": 0.60,           # thicker flavedo, more oil glands (Ferrara et al., 2020)
        "limonene_frac": 0.58,    # lower limonene, more oxygenated terpenes
        "citral_frac": 0.05,      # higher citral = strongest lemon identity
        "minor_terpenes": 0.37,   # rich terpene diversity: pinene, terpinene, linalool
        "pith_thickness": 1.3,    # thicker pith (more careful zesting needed)
        "acidity": 6.0,
        "peel_brix": 1.8,
    },
    3: {  # Meyer
        "name": "Meyer",
        "oil_ml": 0.30,           # thinner peel, less oil (Sawamura et al., 2004)
        "limonene_frac": 0.62,    # 55-68% limonene
        "citral_frac": 0.015,     # lower citral = softer, less "lemony" character
        "minor_terpenes": 0.365,  # high linalool (3-8%) from mandarin heritage, floral notes
        "pith_thickness": 0.5,    # very thin pith (less bitterness risk)
        "acidity": 4.0,           # noticeably less acidic
        "peel_brix": 3.5,         # sweeter peel
    },
    4: {  # Primofiore / Verdello
        "name": "Primofiore",
        "oil_ml": 0.50,
        "limonene_frac": 0.65,    # 60-72% limonene
        "citral_frac": 0.04,      # moderate-high citral
        "minor_terpenes": 0.31,   # pinene-dominant minor fraction
        "pith_thickness": 1.1,
        "acidity": 7.0,           # high acidity, very sharp
        "peel_brix": 1.5,
    },
    5: {  # Interdonato (Sicilian hybrid, citron x lemon)
        "name": "Interdonato",
        "oil_ml": 0.45,           # moderate oil yield (Ferrara et al., 2020)
        "limonene_frac": 0.60,    # 55-65% limonene
        "citral_frac": 0.03,      # moderate citral
        "minor_terpenes": 0.37,   # high beta-pinene (10-18%), good complexity
        "pith_thickness": 0.8,    # thinner pith than Eureka, easy to zest
        "acidity": 5.0,           # mild acidity
        "peel_brix": 2.2,
    },
    6: {  # Verna (Spanish cultivar, widely grown)
        "name": "Verna",
        "oil_ml": 0.42,           # moderate oil, thin-to-medium peel
        "limonene_frac": 0.64,    # 60-70% limonene
        "citral_frac": 0.045,     # good citral level
        "minor_terpenes": 0.315,  # balanced minor terpene profile
        "pith_thickness": 0.9,
        "acidity": 5.8,
        "peel_brix": 1.9,
    },
}

def get_variety(val):
    """Clamp and round to nearest valid variety key."""
    return int(max(1, min(6, round(val))))

# ---------------------------------------------------------------------------
# 1. Define variables
# ---------------------------------------------------------------------------
variables = [
    # Core recipe
    Variable("lemons", 4, 18, integer=True, unit="count"),
    Variable("lemon_variety", 1, 6, integer=True, unit="1=Eureka,2=Femminello,3=Meyer,4=Primofiore,5=Interdonato,6=Verna"),
    Variable("freshness_days", 1, 14, unit="days since harvest", rounding=1),
    Variable("spirit_abv", 0.40, 0.96, unit="fraction"),
    Variable("spirit_vol", 300, 1500, unit="mL", rounding=5),
    Variable("days", 2, 45, unit="days", rounding=1),
    Variable("sugar_g", 200, 1500, unit="g", rounding=5),
    Variable("water_ml", 200, 2500, unit="mL", rounding=5),
    # Technique & process
    Variable("zest_fineness", 1, 10, unit="1=peeler,10=microplane", rounding=1),
    Variable("infusion_temp_c", 4, 30, unit="C", rounding=1),
    Variable("vodka_quality", 1, 10, unit="1=cheap,10=premium", rounding=1),
    Variable("rest_days", 0, 30, unit="days", rounding=1),
    Variable("serving_temp_c", -18, 10, unit="C", rounding=1),
]

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------
WEIGHTS = {
    "taste":        0.35,
    "drinkability": 0.35,
    "strength":     0.15,
    "authenticity": 0.15,
}

EVAL_COUNT = 0


# ---------------------------------------------------------------------------
# 2. Scoring function
# ---------------------------------------------------------------------------
def score(v: dict) -> tuple[float, dict]:
    global EVAL_COUNT
    EVAL_COUNT += 1
    lemons      = v["lemons"]
    variety_raw = v["lemon_variety"]
    fresh_days  = v["freshness_days"]
    abv         = v["spirit_abv"]
    vol         = v["spirit_vol"]
    days        = v["days"]
    sugar       = v["sugar_g"]
    water       = v["water_ml"]
    zest_fine   = v["zest_fineness"]
    inf_temp    = v["infusion_temp_c"]
    vod_quality = v["vodka_quality"]
    rest        = v["rest_days"]
    serv_temp   = v["serving_temp_c"]

    # Bounds guard: Nelder-Mead can escape variable bounds.
    for var in variables:
        if v[var.name] < var.low - 1 or v[var.name] > var.high + 1:
            return 0.0, {k: 0.0 for k in WEIGHTS}

    # Resolve lemon variety to profile
    variety = get_variety(variety_raw)
    lp = VARIETY_PROFILES[variety]

    # ===================================================================
    # PHYSICAL CALCULATIONS
    # ===================================================================

    # -- Oil yield per lemon, adjusted for freshness --
    #
    # Essential oil degrades post-harvest via oxidation and evaporation.
    # Monoterpenes (especially d-limonene) oxidize to carvone and limonene
    # oxide. First-order decay with ~20-day half-life at room temp.
    # Fresh lemons (1-2 days) retain ~100%; 14-day-old lemons retain ~60%.
    freshness_factor = exp(-0.035 * max(fresh_days - 1, 0))
    oil_per_lemon = lp["oil_ml"] * freshness_factor

    # -- Oil extraction: two-compartment first-order kinetics --
    #
    # Rate constants scale with:
    #   ABV solubility: sigmoidal model from published partition coefficients
    #     (Li & Tamura, 2010; Rao & McClements, 2012). d-Limonene is nearly
    #     insoluble below 20% ABV, good solubility around 50-60%, fully
    #     miscible above ~90%. Steep transition centered at ~48% ABV.
    #   Arrhenius temp factor: ~2x rate per 10C above 20C reference
    #   Zest surface area: finer zest = more exposed oil glands
    abv_rate = 1 / (1 + exp(-15 * (abv - 0.48)))
    temp_factor = 2 ** ((inf_temp - REF_TEMP_C) / 10)
    zest_area = 0.3 + 0.12 * max(zest_fine, 1)
    rate = abv_rate * temp_factor * zest_area

    k_fast = 0.80 * rate
    k_slow = 0.08 * rate
    extraction = (0.60 * (1 - exp(-k_fast * max(days, 0)))
               + 0.40 * (1 - exp(-k_slow * max(days, 0))))
    total_oil = lemons * oil_per_lemon * extraction   # mL

    # -- Ethanol evaporation during infusion --
    #
    # Even in a sealed jar, some ethanol is lost to headspace and micro-leaks.
    # Loss rate increases with temperature (vapor pressure doubles ~every 15C).
    # Well-sealed jar at 20C: ~0.5% ABV loss per week. Warm (30C) or loose
    # lid: up to 2%/week. Cold (4C): negligible. This affects final ABV calc.
    # (Raoult's law for ethanol partial pressure over aqueous solution)
    evap_rate = 0.0007 * 2 ** ((inf_temp - 20) / 15)  # fraction lost per day
    abv_after_infusion = abv * (1 - evap_rate * max(days, 0))
    abv_after_infusion = max(abv_after_infusion, abv * 0.85)  # cap at 15% loss

    # -- Monoterpene oxidation during extended infusion --
    #
    # d-Limonene and other monoterpenes oxidize in ethanol solution to form
    # limonene oxide, carvone, and p-cymene (camphoraceous, stale off-notes).
    # Rate accelerates above 20-25 days and at warm temperatures.
    # (Nguyen et al., 2009, J. Agric. Food Chem.; Clark & Chamblee, 1992)
    # Freshness_factor already captured harvest degradation; this captures
    # in-solution oxidation during the steep itself.
    oxidation_days = max(0, days - 14)  # negligible first 2 weeks
    oxidation_temp = 1 + 0.5 * max(0, (inf_temp - 18) / 12)  # warm accelerates
    terpene_freshness = exp(-0.008 * oxidation_days * oxidation_temp)
    # Scale: 1.0 at 14 days, 0.93 at 21 days (20C), 0.78 at 35 days (20C),
    #        0.65 at 45 days (30C)

    # -- Terpene composition of extracted oil --
    #
    # d-Limonene extracts fastest (most soluble). Minor terpenes (beta-pinene,
    # gamma-terpinene, citral, linalool) extract slower. Citral in particular
    # is the key "lemon identity" compound -- without enough citral, it tastes
    # like generic citrus, not lemon.
    limonene_ml = total_oil * lp["limonene_frac"]
    citral_ml = total_oil * lp["citral_frac"] * min(1, extraction * 1.2)
    minor_ml = total_oil * lp["minor_terpenes"] * min(1, extraction * 0.8)

    # -- Volumes and concentrations --
    sugar_vol  = sugar * 0.63
    total_vol  = vol + water + sugar_vol
    total_vol  = max(total_vol, 1)
    oil_conc   = total_oil / total_vol * 1000           # mL/L
    citral_conc = citral_ml / total_vol * 1000          # mL/L
    minor_conc = minor_ml / total_vol * 1000            # mL/L
    final_abv  = (abv_after_infusion * vol) / total_vol

    # -- Effective oil quality: reduced by in-solution oxidation --
    effective_oil_quality = terpene_freshness  # 0-1, applied to aroma/flavor

    # -- Brix --
    spirit_density = abv * 0.789 + (1 - abv)
    total_mass = vol * spirit_density + water + sugar
    brix = sugar / max(total_mass, 1) * 100

    # -- Oil-in-water emulsion stability --
    #
    # Lemon oil is immiscible in the final drink. It forms a micro-emulsion
    # stabilized by ethanol as co-solvent. Below ~25% ABV, emulsion breaks
    # and oil separates (becomes cloudy/oily). Above ~35%, oil dissolves
    # fully. Sugar increases aqueous density, reducing Stokes settling.
    # Resting time allows emulsion to equilibrate.
    emulsion_abv = max(0, min(1, (final_abv - 0.20) / 0.20))  # 0 at 20%, 1 at 40%
    sugar_stabilize = 0.5 + 0.5 * min(1, brix / 30)           # sugar helps
    rest_stabilize = 1 - 0.3 * exp(-0.2 * max(rest, 0))       # resting helps
    emulsion_stability = min(1, emulsion_abv * sugar_stabilize * rest_stabilize)

    # -- Louche quality (visual aesthetics) --
    #
    # When water/syrup is added, lemon oil forms a micro-emulsion that
    # scatters light (louche). This milky opalescence is a quality signal.
    # Too little oil = transparent/thin (no louche). Too much = heavy
    # turbidity and oil slick. Optimal is a gentle, even opalescence.
    # Peak louche quality at oil_conc ~1.0-2.0 mL/L.
    louche_quality = exp(-0.5 * ((oil_conc - 1.5) / 1.0) ** 2)

    # -- Pith bitterness: limonin from pith exposure --
    #
    # Bitterness comes from limonin in the white pith, NOT from the zest.
    # With careful zesting (peeler or light microplane), almost no pith
    # enters the infusion and bitterness stays near zero even at 30+ days.
    # Only sloppy zesting (deep microplane or including peel chunks)
    # introduces pith. This is why every Nonna recipe can steep for a
    # month with zero bitterness -- they zest cleanly.
    #
    # pith_exposure: 0 for clean strips (fineness 1-3), rises steeply 8-10
    pith_exposure = lp["pith_thickness"] * max(0, (zest_fine - 3) / 7) ** 1.5
    temp_bitter = 1 + 0.02 * max(inf_temp - 20, 0)

    if pith_exposure < 0.01:
        bitterness = 0.0
    else:
        # ABV helps extract limonin from whatever pith is present
        limonin_rate = 0.05 * (1 + 0.5 * abv) * pith_exposure * temp_bitter
        bitterness = pith_exposure * (1 - exp(-limonin_rate * max(days, 0)))

    # -- Post-mix mellowing --
    ester_formation = 1 - exp(-0.14 * max(rest, 0))
    mellow_factor = 1 - 0.25 * exp(-0.12 * max(rest, 0))

    # -- Congener effect from spirit quality --
    burn_quality = 1.36 - 0.06 * max(vod_quality, 1)
    congener_taint = (10 - max(vod_quality, 1)) / 10 * 0.12

    # ===================================================================
    # TASTE (weight 0.30)
    # ===================================================================
    #
    # Three-channel sensory model. Taste is a composite of what you
    # actually perceive in distinct neural pathways:
    #
    #   1. Orthonasal aroma (sniffing the glass -- volatile partitioning)
    #   2. Retronasal flavor (in-mouth -- oil concentration on tongue)
    #   3. Gustatory (sweetness, bitterness, acid -- taste buds only)

    # -- Freshness quality: the single biggest factor --
    #
    # Fresh zest has bright, vivid terpenes. Oxidized oil develops off-notes
    # (carvone, limonene oxide) that muddy the aroma irreversibly. No amount
    # of technique can compensate for stale lemons.
    freshness_quality = 0.3 + 0.7 * freshness_factor

    # --- Channel 1: Orthonasal Aroma ---
    #
    # What you smell before sipping. Governed by Henry's law: volatile
    # terpenes partition between liquid and headspace. Higher serving
    # temp = more volatiles in headspace = stronger aroma.
    # At -18C almost nothing reaches your nose. At 10C, full aromatic bloom.
    henry_temp = max(0.05, 0.05 + 0.95 * (serv_temp + 18) / 28)

    # Citral is the "lemon identity" compound. Without it, aroma is
    # generic-citrus. Michaelis-Menten for olfactory receptor saturation.
    citral_aroma = 10 * citral_conc / (citral_conc + 0.08) * henry_temp

    # Minor terpenes add complexity (beta-pinene = pine, linalool = floral,
    # gamma-terpinene = herbal). Logarithmic: many small contributions.
    minor_aroma = min(4, 2 * log(1 + minor_conc / 0.05)) * henry_temp

    # Limonene contributes generic citrus backdrop (always present)
    limonene_aroma = 6 * min(1, oil_conc / 2.0) * henry_temp

    # Congeners from cheap spirit mask delicate terpene notes
    aroma_clarity = 1 - congener_taint

    # Ester formation adds fruity top-notes (ethyl citrate, ethyl acetate)
    ester_aroma = 2.0 * ester_formation * henry_temp

    orthonasal = min(10, (0.35 * citral_aroma
                        + 0.25 * limonene_aroma
                        + 0.20 * minor_aroma
                        + 0.20 * ester_aroma) * aroma_clarity * freshness_quality
                        * effective_oil_quality)

    # --- Channel 2: Retronasal Flavor ---
    #
    # What you taste after swallowing -- oil released from emulsion droplets
    # hitting the soft palate. Less temperature-dependent than orthonasal
    # (mouth warms the liquid). Depends heavily on oil concentration and
    # emulsion quality (stable emulsion = controlled, even release).
    oil_flavor = 10 * oil_conc / (oil_conc + 0.5)

    # Stable emulsion delivers consistent flavor; broken emulsion gives
    # oily blobs on top and watery liquid below -- unpleasant.
    emulsion_delivery = 0.5 + 0.5 * emulsion_stability

    # Complexity: how many distinct flavor notes you can detect.
    # Time is huge: terpene recombination, ester formation, harsh note
    # dissipation. Real recipes say minimum 2 weeks infusion.
    # Stale lemons produce flat, oxidized terpenes -- less complexity.
    terpene_diversity = min(1, (lp["citral_frac"] + lp["minor_terpenes"]) / 0.12)
    time_develop = 3.5 * log(1 + max(days, 0) / 5)
    maturity = min(1, max(days, 0) / 21)  # reaches 1.0 at 3 weeks
    complexity = min(10, (3 + time_develop + 2 * maturity) * terpene_diversity
                     * freshness_quality * effective_oil_quality)
    complexity += 2.0 * ester_formation  # esters add fruity layer

    retronasal = min(10, (0.45 * oil_flavor * emulsion_delivery
                        + 0.30 * complexity
                        + 0.25 * oil_flavor * (1 - congener_taint)) * freshness_quality)

    # --- Channel 3: Gustatory (taste buds) ---
    #
    # Sweetness, bitterness, acidity -- the only "true" tastes.

    # Sweetness: Weber-Fechner law
    perceived_sweet = 1.56 * log(1 + max(brix, 0))

    # Acid suppression + variety acidity contribution
    # Lemon oil carries citric acid; variety acidity modulates.
    acid_from_oil = oil_conc * 0.06 * (lp["acidity"] / 5.5)
    acid_factor = max(0.65, 1 - acid_from_oil)
    perceived_sweet *= acid_factor

    # Peel sugars (especially Meyer) add slight sweetness
    peel_sweet_boost = lp["peel_brix"] / 100 * lemons * 0.3
    perceived_sweet += peel_sweet_boost / max(total_vol / 1000, 0.5)

    sweet_pref = 10 * exp(-0.5 * ((perceived_sweet - TARGET_SWEET) / 0.9) ** 2)

    # Pith bitterness: even small amounts ruin limoncello. Real makers
    # are obsessive about avoiding any white pith. Exponential penalty.
    # BUT: sweetness masks bitterness perception (Breslin & Beauchamp, 1997;
    # Keast & Breslin, 2003). At typical limoncello sugar levels (25-30 Brix),
    # sucrose suppresses quinine-type bitterness by 30-50%.
    sweet_mask = 1 - 0.4 * min(1, max(brix, 0) / 30)  # 0.6 at 30 Brix
    perceived_bitterness = bitterness * sweet_mask
    clean = 10 * exp(-4 * perceived_bitterness ** 2)

    # Acid balance: some acidity is good (brightness), too much is harsh.
    # Optimal around acidity_factor = 0.85.
    acid_balance = 10 * exp(-8 * (acid_factor - 0.85) ** 2)

    gustatory = 0.40 * sweet_pref + 0.30 * clean + 0.30 * acid_balance

    # --- Combined taste ---
    taste = float(np.clip(0.35 * orthonasal + 0.35 * retronasal + 0.30 * gustatory, 0, 10))

    # ===================================================================
    # DRINKABILITY (weight 0.30)
    # ===================================================================
    #
    # "Would you pour a second glass?" Geometric mean of components,
    # so one weak dimension drags the whole score down (as in real life).

    # Burn perception: Stevens' power law + TRPV1/TRPM8 cold-gating + quality.
    burn_room = 10 * (max(final_abv, 0) / 0.50) ** 1.3
    # TRPM8 cold receptor activation: sigmoidal gating centered at ~8C.
    # Below -5C, ethanol burn is almost fully suppressed (receptor saturated).
    # Above 15C, no cold-suppression at all. Matches psychophysical data
    # from Green (1993) and Lemon et al. (2019).
    cold_factor = 1 / (1 + exp(-0.3 * (serv_temp + 2)))  # 0.05 at -18C, 0.95 at 10C
    burn = burn_room * cold_factor * burn_quality * mellow_factor
    burn_comfort = 10 * exp(-0.15 * (burn - 1.5) ** 2)

    # Sweetness pleasantness
    sweet_score = 10 * exp(-0.18 * (perceived_sweet - 5.5) ** 2)

    # Flavor persistence: retronasal carries into drinkability
    flavor_score = retronasal * 0.9 + orthonasal * 0.1

    # Mouthfeel: proper viscosity model
    #
    # Three contributions to body/viscosity:
    #   1. Sugar syrup: dominates. Exponential rise with Brix.
    #   2. Ethanol: moderate viscosity peak around 40% ABV (max water-ethanol
    #      hydrogen bonding), drops at higher/lower ABV.
    #   3. Emulsion: stable micro-emulsion adds silky texture; broken
    #      emulsion feels oily and thin simultaneously.
    sugar_body = 1 - exp(-0.08 * max(brix, 0))
    etoh_body = exp(-3 * (final_abv - 0.40) ** 2)  # peak at 40% ABV
    emulsion_body = 0.3 * emulsion_stability
    mouthfeel = 10 * min(1, sugar_body * 0.55 + etoh_body * 0.25 + emulsion_body * 0.20)

    # Cleanliness: pith bitterness ruins drinkability fast
    cleanliness = 10 * exp(-5 * perceived_bitterness ** 2) * (1 - congener_taint * 0.3)

    # Visual / presentation: louche quality (opalescence)
    # Drinkability includes the visual experience. A proper limoncello
    # should have a milky opalescent glow, not be transparent or oil-slicked.
    presentation = 6 + 4 * louche_quality

    # Smoothness from resting
    smoothness = 6 + 4 * ester_formation  # 6-10 range

    # Geometric mean (7 components -- one bad dimension drags everything down)
    components = [burn_comfort, sweet_score, flavor_score, mouthfeel,
                  cleanliness, smoothness, presentation]
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

    # Vodka quality: premium costs more, harder to justify
    quality_ease = max(4, 10 - 0.5 * max(vod_quality - 5, 0))

    # Lemon variety availability:
    # Eureka=everywhere, Primofiore=Italian shops, Femminello=specialty,
    # Meyer=seasonal, Interdonato=Italian/specialty, Verna=European markets
    variety_ease = {1: 10, 2: 4, 3: 7, 4: 5, 5: 4, 6: 5}.get(variety, 7)

    # Freshness: 1-3 days = farmers market trip, 4-7 = normal store, 8+ = easy
    if fresh_days <= 3:
        fresh_ease = 6   # need to plan timing
    elif fresh_days <= 7:
        fresh_ease = 9   # normal grocery store
    else:
        fresh_ease = 10  # no effort, but worse oil

    # Total waiting: infusion + resting
    total_wait = max(days, 0) + max(rest, 0)
    wait_ease = 10 * exp(-0.012 * total_wait)

    quantity_ease = 10 if lemons <= 12 else max(5, 10 - 0.4 * (lemons - 12))

    # Infusion temp: room temp is easiest, fridge is fine, warm needs setup
    if 15 <= inf_temp <= 25:
        temp_ease = 10
    elif inf_temp < 15:
        temp_ease = 8   # just put it in the fridge
    else:
        temp_ease = 6   # need warm water bath or similar

    # Zest technique: peeler is easy, microplane takes patience
    zest_ease = max(5, 10 - 0.4 * max(zest_fine - 3, 0))

    ease = float(np.clip(
        (spirit_ease + quality_ease + variety_ease + fresh_ease
         + wait_ease + quantity_ease + temp_ease + zest_ease) / 8,
        0, 10
    ))

    # ===================================================================
    # STRENGTH (weight 0.10)
    # ===================================================================
    #
    # Traditional limoncello: 28% ABV. Real range is 24-35%.
    strength = float(10 * exp(-0.5 * ((final_abv - TARGET_ABV) / 0.06) ** 2))

    # ===================================================================
    # AUTHENTICITY (weight 0.10)
    # ===================================================================
    #
    # Does it match what you would find in a bar on the Amalfi Coast?
    # ABV in range, Brix in range, strong lemon oil, traditional variety.
    abv_auth  = exp(-0.5 * ((final_abv - 0.30) / 0.05) ** 2)
    brix_auth = exp(-0.5 * ((brix - TARGET_BRIX) / 5) ** 2)
    oil_auth  = min(1.0, oil_conc / 1.5)
    # Femminello is the canonical limoncello lemon; Primofiore is acceptable
    # Interdonato has Sicilian heritage; Verna is Spanish (less traditional)
    variety_auth = {1: 0.6, 2: 1.0, 3: 0.3, 4: 0.8, 5: 0.7, 6: 0.5}.get(variety, 0.5)
    authenticity = float(10 * np.clip(
        (abv_auth * brix_auth * oil_auth * variety_auth) ** (1/4), 0, 1
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
        # Giada De Laurentiis (Food Network):
        # 10 lemons, 750mL grain alcohol, 4 days, ~700g sugar in 1L water
        ("Giada De Laurentiis",
         {"lemons": 10, "lemon_variety": 1, "freshness_days": 3,
          "spirit_abv": 0.95, "spirit_vol": 750, "days": 4,
          "sugar_g": 700, "water_ml": 1000, "zest_fineness": 6,
          "infusion_temp_c": 20, "vodka_quality": 6, "rest_days": 4,
          "serving_temp_c": -10}),
        # Serious Eats (vodka method):
        # 12 lemons, 750mL vodka, 7 days, simple syrup (600g sugar, 500mL water)
        ("Serious Eats vodka",
         {"lemons": 12, "lemon_variety": 1, "freshness_days": 4,
          "spirit_abv": 0.40, "spirit_vol": 750, "days": 7,
          "sugar_g": 600, "water_ml": 500, "zest_fineness": 7,
          "infusion_temp_c": 20, "vodka_quality": 6, "rest_days": 0,
          "serving_temp_c": -5}),
        # Traditional Amalfi (Sal De Riso style):
        # 15 Sfusato lemons, 1L 96% alcohol, 10-15 days, 800g sugar, 1.5L water, 30 day rest
        ("Sal De Riso Amalfi",
         {"lemons": 15, "lemon_variety": 2, "freshness_days": 1,
          "spirit_abv": 0.96, "spirit_vol": 1000, "days": 12,
          "sugar_g": 800, "water_ml": 1500, "zest_fineness": 6,
          "infusion_temp_c": 18, "vodka_quality": 7, "rest_days": 30,
          "serving_temp_c": -12}),
        # Bon Appetit "Best" (Everclear method):
        # 8 lemons, 750mL Everclear, 5 days, 500g sugar, 750mL water, 1 week rest
        ("Bon Appetit Everclear",
         {"lemons": 8, "lemon_variety": 1, "freshness_days": 3,
          "spirit_abv": 0.75, "spirit_vol": 750, "days": 5,
          "sugar_g": 500, "water_ml": 750, "zest_fineness": 8,
          "infusion_temp_c": 20, "vodka_quality": 5, "rest_days": 7,
          "serving_temp_c": -8}),
        # Italian Nonna (month-long patience):
        # 12 Femminello, 1L grain alcohol, 30 days steep, 1kg sugar, 1L water, 14 day rest
        ("Italian Nonna classic",
         {"lemons": 12, "lemon_variety": 2, "freshness_days": 1,
          "spirit_abv": 0.96, "spirit_vol": 1000, "days": 30,
          "sugar_g": 1000, "water_ml": 1000, "zest_fineness": 5,
          "infusion_temp_c": 18, "vodka_quality": 7, "rest_days": 14,
          "serving_temp_c": -10}),
        # Meyer lemon variation (California style):
        # 14 Meyer lemons (need more due to less oil), vodka, longer steep
        ("Meyer California",
         {"lemons": 14, "lemon_variety": 3, "freshness_days": 2,
          "spirit_abv": 0.40, "spirit_vol": 750, "days": 14,
          "sugar_g": 550, "water_ml": 600, "zest_fineness": 7,
          "infusion_temp_c": 20, "vodka_quality": 7, "rest_days": 7,
          "serving_temp_c": -8}),
        # Sorrento IGP commercial profile:
        # Strict IGP spec: Sfusato only, 96% alcohol, 25-30% final ABV, 25-32 Brix
        ("Sorrento IGP",
         {"lemons": 18, "lemon_variety": 2, "freshness_days": 1,
          "spirit_abv": 0.96, "spirit_vol": 1200, "days": 15,
          "sugar_g": 900, "water_ml": 1800, "zest_fineness": 5,
          "infusion_temp_c": 16, "vodka_quality": 7, "rest_days": 21,
          "serving_temp_c": -14}),
        # Cold infusion (fridge method, popular on r/limoncello):
        # 10 lemons, grain alcohol, 30+ days cold, less bitterness risk
        ("Reddit cold infusion",
         {"lemons": 10, "lemon_variety": 1, "freshness_days": 3,
          "spirit_abv": 0.96, "spirit_vol": 750, "days": 40,
          "sugar_g": 650, "water_ml": 900, "zest_fineness": 8,
          "infusion_temp_c": 4, "vodka_quality": 6, "rest_days": 14,
          "serving_temp_c": -12}),
        # Sicilian Interdonato style:
        # Mild acidity Interdonato lemons, clean zesting, medium rest
        ("Sicilian Interdonato",
         {"lemons": 12, "lemon_variety": 5, "freshness_days": 2,
          "spirit_abv": 0.96, "spirit_vol": 1000, "days": 10,
          "sugar_g": 750, "water_ml": 1200, "zest_fineness": 5,
          "infusion_temp_c": 20, "vodka_quality": 6, "rest_days": 14,
          "serving_temp_c": -10}),
        # Minimal/quick recipe (worst case test):
        # 4 lemons, vodka, 2 days, minimal rest
        ("Quick minimal",
         {"lemons": 4, "lemon_variety": 1, "freshness_days": 10,
          "spirit_abv": 0.40, "spirit_vol": 375, "days": 2,
          "sugar_g": 200, "water_ml": 200, "zest_fineness": 3,
          "infusion_temp_c": 25, "vodka_quality": 3, "rest_days": 0,
          "serving_temp_c": 5}),
    ]
    print("\n" + "="*65)
    print("  SANITY CHECK")
    print("  " + "-"*61)
    print("  Testing known recipes to verify scoring function...\n")
    for i, (name, r) in enumerate(test_recipes, 1):
        sc, subs = score(r)
        variety = VARIETY_PROFILES[get_variety(r['lemon_variety'])]['name']
        parts = "  ".join(f"{k}={v:.1f}" for k, v in subs.items())
        print(f"  Recipe {i}: {name}")
        print(f"    {variety}, {r['lemons']} lemons, {r['spirit_abv']*100:.0f}% ABV, "
              f"{r['days']}d infuse, {r['rest_days']}d rest")
        print(f"    Score: {sc:.2f}   {parts}")
    print("  " + "-"*61)
    print()


# ---------------------------------------------------------------------------
# 4. Diagnostics
# ---------------------------------------------------------------------------
def print_diagnostics(v: dict):
    """Print the physical properties of a recipe for verification."""
    abv         = v["spirit_abv"]
    vol         = v["spirit_vol"]
    days        = v["days"]
    sugar       = v["sugar_g"]
    water       = v["water_ml"]
    lemons      = v["lemons"]
    variety     = get_variety(v["lemon_variety"])
    lp          = VARIETY_PROFILES[variety]
    fresh_days  = v["freshness_days"]
    zest_fine   = v["zest_fineness"]
    inf_temp    = v["infusion_temp_c"]
    vod_quality = v["vodka_quality"]
    rest        = v["rest_days"]
    serv_temp   = v["serving_temp_c"]

    freshness_factor = exp(-0.035 * max(fresh_days - 1, 0))
    oil_per_lemon = lp["oil_ml"] * freshness_factor

    abv_rate = 1 / (1 + exp(-15 * (abv - 0.48)))
    temp_factor = 2 ** ((inf_temp - REF_TEMP_C) / 10)
    zest_area = 0.3 + 0.12 * max(zest_fine, 1)
    rate = abv_rate * temp_factor * zest_area
    k_fast = 0.80 * rate
    k_slow = 0.08 * rate
    extraction = 0.60 * (1 - exp(-k_fast * days)) + 0.40 * (1 - exp(-k_slow * days))
    total_oil = lemons * oil_per_lemon * extraction
    evap_rate = 0.0007 * 2 ** ((inf_temp - 20) / 15)
    abv_after_infusion = abv * (1 - evap_rate * days)
    abv_after_infusion = max(abv_after_infusion, abv * 0.85)
    oxidation_days = max(0, days - 14)
    oxidation_temp = 1 + 0.5 * max(0, (inf_temp - 18) / 12)
    terpene_freshness = exp(-0.008 * oxidation_days * oxidation_temp)
    total_vol = vol + water + sugar * 0.63
    oil_conc = total_oil / total_vol * 1000
    citral_conc = total_oil * lp["citral_frac"] / total_vol * 1000
    final_abv = abv_after_infusion * vol / total_vol
    spirit_density = abv * 0.789 + (1 - abv)
    total_mass = vol * spirit_density + water + sugar
    brix = sugar / total_mass * 100
    pith_exposure = lp["pith_thickness"] * max(0, (zest_fine - 3) / 7) ** 1.5
    temp_bitter = 1 + 0.02 * max(inf_temp - 20, 0)
    if pith_exposure < 0.01:
        bitterness = 0.0
    else:
        limonin_rate = 0.05 * (1 + 0.5 * abv) * pith_exposure * temp_bitter
        bitterness = pith_exposure * (1 - exp(-limonin_rate * days))
    sweet_mask = 1 - 0.4 * min(1, brix / 30)
    perceived_bitterness = bitterness * sweet_mask
    burn_quality = 1.36 - 0.06 * max(vod_quality, 1)
    mellow_factor = 1 - 0.25 * exp(-0.12 * max(rest, 0))
    burn_room = 10 * (final_abv / 0.50) ** 1.3
    cold_factor = 1 / (1 + exp(-0.3 * (serv_temp + 2)))
    burn = burn_room * cold_factor * burn_quality * mellow_factor
    ester = 1 - exp(-0.14 * max(rest, 0))
    congener_taint = (10 - max(vod_quality, 1)) / 10 * 0.12
    perceived_sweet = 1.56 * log(1 + brix) * max(0.65, 1 - oil_conc * 0.06 * lp["acidity"] / 5.5)
    henry_temp = max(0.05, 0.05 + 0.95 * (serv_temp + 18) / 28)
    emulsion_abv = max(0, min(1, (final_abv - 0.20) / 0.20))
    sugar_stab = 0.5 + 0.5 * min(1, brix / 30)
    rest_stab = 1 - 0.3 * exp(-0.2 * max(rest, 0))
    emulsion = min(1, emulsion_abv * sugar_stab * rest_stab)
    louche = exp(-0.5 * ((oil_conc - 1.5) / 1.0) ** 2)

    print(f"\n  WINNER DIAGNOSTICS")
    print(f"  {'-'*61}")
    print(f"    Lemon variety:     {lp['name']}")
    print(f"    Oil per lemon:     {oil_per_lemon:.2f} mL (fresh factor {freshness_factor:.2f})")
    print(f"    Oil extraction:    {extraction*100:.1f}%")
    print(f"    Total oil:         {total_oil:.2f} mL")
    print(f"    Oil concentration: {oil_conc:.2f} mL/L")
    print(f"    Citral conc:       {citral_conc:.3f} mL/L")
    print(f"    Terpene freshness: {terpene_freshness:.2f} (oxidation after 14d)")
    print(f"    Total volume:      {total_vol:.0f} mL")
    print(f"    Final ABV:         {final_abv*100:.1f}% (evap loss: {(1 - abv_after_infusion/abv)*100:.1f}%)")
    print(f"    Brix:              {brix:.1f}")
    print(f"    Bitterness:        {bitterness:.3f} (perceived: {perceived_bitterness:.3f}, pith {pith_exposure:.3f})")
    print(f"    Burn at {serv_temp:.0f}C:      {burn:.2f}/10 (cold factor {cold_factor:.2f})")
    print(f"    Perceived sweet:   {perceived_sweet:.2f}/10")
    print(f"    Ester formation:   {ester*100:.0f}%")
    print(f"    Emulsion stability:{emulsion:.2f}")
    print(f"    Louche quality:    {louche:.2f}")
    print(f"    Congener taint:    {congener_taint:.3f}")
    print(f"    Mellow factor:     {mellow_factor:.3f}")
    print(f"    Headspace aroma:   {henry_temp:.2f}")
    print(f"    Extraction rate:   {rate:.2f}x (ABV={abv_rate:.2f} temp={temp_factor:.2f} zest={zest_area:.2f})")
    dilution = (water + sugar * 0.63) / max(vol, 1)
    print(f"    Dilution ratio:    {dilution:.2f}:1 (diluent:spirit)")
    freshness_quality = 0.3 + 0.7 * freshness_factor
    print(f"    Freshness quality: {freshness_quality:.2f}")


# ---------------------------------------------------------------------------
# 5. Recipe writer
# ---------------------------------------------------------------------------
def write_recipe(r: dict, scores: dict, filepath: str = "recipe.txt"):
    """Write a human-readable limoncello recipe from optimizer results."""
    v = r
    variety_id = get_variety(v["lemon_variety"])
    lp = VARIETY_PROFILES[variety_id]
    variety_name = lp["name"]

    lemons = int(v["lemons"])
    abv = v["spirit_abv"]
    vol = v["spirit_vol"]
    days = int(round(v["days"]))
    sugar = v["sugar_g"]
    water = v["water_ml"]
    zest = int(round(v["zest_fineness"]))
    inf_temp = int(round(v["infusion_temp_c"]))
    vod_qual = int(round(v["vodka_quality"]))
    rest = int(round(v["rest_days"]))
    serv_temp = int(round(v["serving_temp_c"]))
    fresh = int(round(v["freshness_days"]))

    # Derived values (match scoring model's ethanol evaporation)
    evap_rate = 0.0007 * 2 ** ((inf_temp - 20) / 15)
    abv_eff = abv * (1 - evap_rate * max(days, 0))
    abv_eff = max(abv_eff, abv * 0.85)  # cap at 15% loss
    total_vol_ml = vol + water + sugar * 0.63
    final_abv = (abv_eff * vol) / total_vol_ml
    brix = (sugar / total_vol_ml) * 100
    spirit_proof = abv * 200
    yield_ml = total_vol_ml
    yield_bottles = yield_ml / 750

    # Zest tool description
    zest_tools = {
        1: "vegetable peeler (wide strips)",
        2: "vegetable peeler (thin strips)",
        3: "Y-peeler or sharp paring knife (thin strips, no pith)",
        4: "fine zester",
        5: "fine zester (thorough)",
        6: "zester/grater",
        7: "fine Microplane (light passes)",
        8: "Microplane (moderate)",
        9: "Microplane (thorough)",
        10: "Microplane (very fine, maximum surface area)",
    }
    zest_tool = zest_tools.get(zest, f"fineness level {zest}")

    # Spirit description
    if abv >= 0.90:
        spirit_type = f"grain alcohol / Everclear ({spirit_proof:.0f} proof)"
    elif abv >= 0.75:
        spirit_type = f"high-proof vodka or grain spirit ({spirit_proof:.0f} proof)"
    elif abv >= 0.50:
        spirit_type = f"overproof vodka ({spirit_proof:.0f} proof)"
    else:
        spirit_type = f"standard vodka ({spirit_proof:.0f} proof)"

    # Quality description
    qual_desc = {
        1: "bottom-shelf", 2: "budget", 3: "budget",
        4: "decent", 5: "mid-range", 6: "mid-range",
        7: "good quality", 8: "premium", 9: "premium",
        10: "top-shelf",
    }
    quality = qual_desc.get(min(10, vod_qual), "good quality")

    # Simple syrup ratio
    syrup_water = water
    syrup_sugar = sugar

    lines = []
    w = lines.append

    w("=" * 65)
    w("  OPTIMIZED LIMONCELLO RECIPE")
    w("=" * 65)
    w("")
    w(f"  Yield:  ~{yield_ml/1000:.1f}L ({yield_bottles:.1f} bottles)")
    w(f"  ABV:    ~{final_abv*100:.0f}%")
    w(f"  Score:  {scores.get('composite_score', 0):.2f}/10")
    w("")
    w("-" * 65)
    w("  INGREDIENTS")
    w("-" * 65)
    w("")
    w(f"  {lemons} {variety_name} lemons, unwaxed")
    w(f"      - Buy the freshest you can find (ideally {fresh} day(s) from harvest)")
    w(f"      - Look for firm, heavy fruit with bright color and fragrant skin")
    w(f"      - Organic strongly preferred (no pesticide wax on peel)")
    w("")
    w(f"  {vol:.0f} mL {spirit_type}")
    w(f"      - {quality} quality -- clean, neutral flavor")
    if abv >= 0.90:
        w(f"      - High proof extracts oils faster and more completely")
        w(f"      - The dilution step will bring it to drinking strength")
    w("")
    w(f"  {syrup_sugar:.0f} g granulated white sugar")
    w(f"  {syrup_water:.0f} mL water (for simple syrup)")
    w("")
    w("-" * 65)
    w("  EQUIPMENT")
    w("-" * 65)
    w("")
    w(f"  - {zest_tool.split('(')[0].strip()} for zesting")
    w(f"  - Large glass jar with tight-fitting lid (at least {vol/1000 + 0.5:.1f}L)")
    w(f"  - Second jar or bottles for finished limoncello")
    w(f"  - Fine mesh strainer + cheesecloth or coffee filter")
    w(f"  - Saucepan for simple syrup")
    w("")
    w("=" * 65)
    w("  INSTRUCTIONS")
    w("=" * 65)
    w("")
    w("  STEP 1: ZEST THE LEMONS")
    w("  " + "-" * 40)
    w("")
    w(f"  Using a {zest_tool}, remove only the bright yellow")
    w(f"  zest from all {lemons} lemons.")
    w("")
    w("  *** THIS IS THE MOST IMPORTANT STEP ***")
    w("")
    w("  - Zest ONLY the yellow part (flavedo). Zero white pith.")
    w("  - The pith is bitter and will ruin your limoncello.")
    if lp["pith_thickness"] > 1.1:
        w(f"  - {variety_name} lemons have thick pith -- be extra careful.")
        w(f"    Use light pressure and rotate the lemon frequently.")
    w("  - Work under good light so you can see the color change.")
    w("  - If you see any white, you've gone too deep. Trim it off.")
    w("  - A sharp tool and light hand are better than pressing hard.")
    w("")
    w("  STEP 2: INFUSE")
    w("  " + "-" * 40)
    w("")
    w(f"  Place all the zest in your glass jar and pour in the")
    w(f"  {vol:.0f} mL of spirit. Seal tightly.")
    w("")
    if inf_temp <= 10:
        w(f"  Store in the refrigerator (~{inf_temp}degC) for {days} days.")
        w(f"  Cold infusion = slower but cleaner, less bitterness risk.")
    elif inf_temp <= 22:
        w(f"  Store in a cool, dark place (~{inf_temp}degC) for {days} days.")
    else:
        w(f"  Store at warm room temperature (~{inf_temp}degC) for {days} days.")
        w(f"  Warmth accelerates extraction -- keep out of direct sunlight.")
    w("")
    w(f"  - Shake gently once a day.")
    w(f"  - The liquid will turn deep golden-yellow as oils extract.")
    w(f"  - Keep the jar sealed to prevent alcohol evaporation.")
    w("")
    w("  STEP 3: STRAIN")
    w("  " + "-" * 40)
    w("")
    w(f"  After {days} days, strain through fine mesh lined with")
    w(f"  cheesecloth. Squeeze gently -- don't force the solids.")
    w(f"  For crystal clarity, strain a second time through a")
    w(f"  coffee filter (this takes patience but is worth it).")
    w("")
    w("  STEP 4: MAKE SIMPLE SYRUP")
    w("  " + "-" * 40)
    w("")
    w(f"  Combine {syrup_sugar:.0f}g sugar with {syrup_water:.0f}mL water in a saucepan.")
    w(f"  Heat over medium, stirring until fully dissolved.")
    w(f"  Do NOT boil -- just dissolve. Let cool COMPLETELY to room temp.")
    w("")
    w("  STEP 5: COMBINE")
    w("  " + "-" * 40)
    w("")
    w(f"  Slowly pour the cooled syrup into the strained infusion.")
    w(f"  Stir gently. It will turn cloudy/milky -- this is normal")
    w(f"  (the lemon oils form a stable emulsion, called louching).")
    w("")
    if rest > 0:
        w(f"  STEP 6: REST ({rest} days)")
        w("  " + "-" * 40)
        w("")
        w(f"  Seal and store in a cool dark place for {rest} days.")
        w(f"  This mellowing period lets flavors marry, harsh edges")
        w(f"  round off, and esters form that add depth and smoothness.")
        w("")
        w(f"  - The longer it rests, the smoother it gets.")
        w(f"  - Shake gently every few days.")
        w(f"  - Some cloudiness may settle -- this is fine.")
        w("")
        serve_step = 7
    else:
        serve_step = 6

    w(f"  STEP {serve_step}: SERVE")
    w("  " + "-" * 40)
    w("")
    if serv_temp <= -10:
        w(f"  Store in the freezer. Serve at {serv_temp}degC in chilled")
        w(f"  shot glasses straight from the freezer.")
    elif serv_temp <= 0:
        w(f"  Chill in the freezer for at least 4 hours before serving.")
        w(f"  Serve at ~{serv_temp}degC in small chilled glasses.")
    else:
        w(f"  Chill well before serving. Best at ~{serv_temp}degC.")
    w(f"  Pour 30-60 mL per serving. Sip slowly -- don't shoot it.")
    w("")
    w("=" * 65)
    w("  TIPS & NOTES")
    w("=" * 65)
    w("")
    w("  What actually makes limoncello great:")
    w("")
    w("  1. FRESH ZEST is the #1 factor. The essential oils in lemon")
    w("     peel degrade rapidly after harvest. Fruit that's been")
    w("     sitting for weeks will make flat, lifeless limoncello.")
    w("     Buy from a farmers market the day you plan to zest.")
    w("")
    w("  2. NO PITH means no bitterness. This is the single most")
    w("     common mistake. Even a small amount of white pith will")
    w("     add a harsh, lingering bitter note that you can't fix.")
    w("     Take your time. It's better to leave some zest behind")
    w("     than to include any pith.")
    w("")
    w("  3. PROPER DILUTION is what makes it smooth and drinkable.")
    w("     Too little sugar/water = harsh alcohol burn.")
    w("     Too much = cloying and syrupy. This recipe is optimized")
    w(f"     for ~{final_abv*100:.0f}% ABV and ~{brix:.0f} Brix -- the sweet spot.")
    w("")
    w("  4. TIME makes a huge difference. Both the infusion period")
    w(f"     ({days} days) and the rest period ({rest} days) matter.")
    w("     Rushing either step produces harsher, less complex results.")
    w("     The resting period is where the magic happens -- esters")
    w("     form, harsh edges mellow, and the flavor integrates.")
    w("")
    w("  5. LEMON VARIETY matters more than most recipes acknowledge.")
    if variety_id == 2:
        w("     Femminello/Sfusato (the traditional Amalfi Coast lemon)")
        w("     has more oil, higher citral content, and richer terpene")
        w("     complexity than supermarket Eureka lemons. If you can")
        w("     find them, they're worth the premium.")
    elif variety_id == 1:
        w("     Eureka/Lisbon are the standard supermarket lemon and")
        w("     work well. Look for organic, unwaxed, heavy fruit.")
    elif variety_id == 3:
        w("     Meyer lemons are sweeter with thinner peel. They make")
        w("     a softer, more floral limoncello -- not traditional but")
        w("     delicious in its own right.")
    elif variety_id == 4:
        w("     Primofiore are seasonal Italian lemons with high acidity")
        w("     that adds brightness and complexity.")
    elif variety_id == 5:
        w("     Interdonato is a Sicilian hybrid (citron x lemon) with")
        w("     mild acidity, thin pith, and complex terpene profile.")
        w("     Easier to zest cleanly than thicker-pithed varieties.")
    elif variety_id == 6:
        w("     Verna is a widely grown Spanish cultivar with balanced")
        w("     citral and good oil yield. Common in European markets.")
    w("")
    w("  Storage: Keeps 1+ year in freezer. Indefinitely if sealed.")
    w("  Cloudiness is normal and a sign of good oil extraction.")
    w("")
    w("-" * 65)
    score_str = scores.get('composite_score', 0)
    w(f"  Recipe optimized by Limoncello Optimizer (score: {score_str})")
    w(f"  Taste: {scores.get('criterion_scores', {}).get('taste', 0):.1f}  "
      f"Drinkability: {scores.get('criterion_scores', {}).get('drinkability', 0):.1f}  "
      f"Strength: {scores.get('criterion_scores', {}).get('strength', 0):.1f}  "
      f"Authenticity: {scores.get('criterion_scores', {}).get('authenticity', 0):.1f}")
    w("-" * 65)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Recipe saved to {filepath}")


# ---------------------------------------------------------------------------
# 6. Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time as _time
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    # Use all available CPU power
    _ncpu = os.cpu_count() or 1
    print(f"\n  CPU cores available: {_ncpu}")
    try:
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00000080)
            print(f"  Process priority:   HIGH")
        else:
            os.nice(-10)
            print(f"  Process priority:   HIGH (nice -10)")
    except (PermissionError, OSError, AttributeError):
        print(f"  Process priority:   default (no permission to elevate)")
    except Exception:
        pass

    total_start = _time.time()

    sanity_check()

    opt = RecipeOptimizer("Limoncello", variables, score)

    # ===================================================================
    # PHASE 1: DE global exploration (broad search, find promising basins)
    # ===================================================================
    result = opt.run(de_restarts=250, de_popsize=750, de_maxiter=5000, n_jobs=-1)
    de_best = result["rounded"]
    de_score = result["composite"]

    print_diagnostics(de_best)

    # ===================================================================
    # PHASE 2: SA primary optimization (better at narrow optima)
    # SA consistently finds improvements over DE on this landscape.
    # Run it as the main optimizer seeded from DE's best result.
    # ===================================================================
    sa1 = opt.sa_search(
        x0=de_best,
        sa_restarts=400, sa_maxiter=100000,
        nm_after=True, n_jobs=-1,
    )
    sa1_best = sa1["rounded"]
    sa1_score = sa1["composite"]

    # Take whichever is better
    if sa1_score > de_score + 1e-6:
        current_best = sa1_best
        current_score = sa1_score
        print(f"\n  SA improved over DE: {de_score:.6f} -> {sa1_score:.6f} (+{sa1_score - de_score:.6f})")
    else:
        current_best = de_best
        current_score = de_score
        print(f"\n  DE result held: {de_score:.6f}")

    # ===================================================================
    # PHASE 3: Iterative SA refinement
    # Keep running SA from the current best until no improvement found.
    # Each round uses fewer restarts but focused around the best region.
    # ===================================================================
    for round_num in range(1, 6):  # up to 5 refinement rounds
        print(f"\n  SA refinement round {round_num} (from score={current_score:.6f})...")
        sa_ref = opt.sa_search(
            x0=current_best,
            sa_restarts=200, sa_maxiter=100000,
            nm_after=True, n_jobs=-1,
        )
        if sa_ref["composite"] > current_score + 1e-6:
            improvement = sa_ref["composite"] - current_score
            current_best = sa_ref["rounded"]
            current_score = sa_ref["composite"]
            print(f"  Round {round_num} improved: +{improvement:.6f} -> {current_score:.6f}")
        else:
            print(f"  Round {round_num}: no improvement, stopping refinement.")
            break

    print_diagnostics(current_best)

    # Sensitivity analysis
    print(f"\n  SENSITIVITY (+/-5% perturbation)")
    print(f"  {'-'*61}")
    sens = opt.sensitivity(current_best)
    sorted_sens = sorted(sens.items(), key=lambda x: abs(x[1]['sensitivity']), reverse=True)
    for name, info in sorted_sens:
        s = info['sensitivity']
        mag = '#' * min(20, int(abs(s) * 5)) + '.' * (20 - min(20, int(abs(s) * 5)))
        direction = '^' if s > 0 else 'v' if s < 0 else '*'
        print(f"    {name:20s} {direction} {mag} {s:+.4f}  "
              f"({info['score_at_low']:.3f} -- {info['score_at_high']:.3f})")

    # Save result
    print(f"\n  Total evaluations: {EVAL_COUNT:,}")
    output = {
        "recipe": current_best,
        "composite_score": round(current_score, 6),
        "criterion_scores": {k: round(v, 4) for k, v in opt.score(current_best)[1].items()},
        "stats": {
            **result["stats"],
            "total_evaluations": EVAL_COUNT,
            "de_score": round(de_score, 6),
            "sa_score": round(sa1_score, 6),
            "final_score": round(current_score, 6),
        },
    }
    with open("result.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Result saved to result.json")

    # Write human-readable recipe
    write_recipe(current_best, output)

    # ===================================================================
    # PHASE 4: Verification
    # ===================================================================
    verification = opt.verify(current_best, n_random=1000000, n_restarts=200, n_jobs=-1)

    # Final cross-check with basin-hopping (different algorithm family)
    cross = opt.cross_check(current_best,
                            sa_restarts=100, sa_maxiter=50000,
                            bh_restarts=100, bh_niter=2000, n_jobs=-1)

    # If cross-check still finds improvement, one more SA pass
    if not cross["confirmed"]:
        print(f"\n  Cross-check found improvement (+{cross['improvement']:.6f}), final SA pass...")
        final_sa = opt.sa_search(
            x0=cross["overall_best_values"],
            sa_restarts=200, sa_maxiter=100000,
            nm_after=True, n_jobs=-1,
        )
        if final_sa["composite"] > current_score:
            current_best = final_sa["rounded"]
            current_score = final_sa["composite"]
            output["recipe"] = current_best
            output["composite_score"] = round(current_score, 6)
            output["criterion_scores"] = {k: round(v, 4) for k, v in final_sa["scores"].items()}
            output["stats"]["final_score"] = round(current_score, 6)
            with open("result.json", "w") as f:
                json.dump(output, f, indent=2)
            write_recipe(current_best, output)
            print_diagnostics(current_best)

    total_elapsed = _time.time() - total_start
    m, s = divmod(int(total_elapsed), 60)
    h, m = divmod(m, 60)
    print(f"\n{'='*65}")
    print(f"  DONE  Total time: {h}h {m}m {s}s")
    print(f"  Total evaluations: {EVAL_COUNT:,}")
    print(f"  Final score: {current_score:.6f}")
    print(f"{'='*65}")
