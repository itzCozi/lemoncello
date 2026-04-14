# limoncello

A general-purpose framework for finding the globally optimal recipe for any food or drink. Define your variables, write a scoring function that encodes your sensory science, and let the optimizer find the best recipe.

The included example optimizes a limoncello recipe using psychophysical sensory models, real lemon variety compositional data, chemical extraction kinetics, and emulsion physics.

## How It Works

The framework uses a 3-phase optimization pipeline:

1. **Global search** — Differential evolution with multiple restarts (Sobol initialization) to explore the full parameter space and avoid local optima.
2. **Local refinement** — Adaptive Nelder-Mead simplex to polish the best candidate to high precision.
3. **Integer sweep** — If any variables are integer-valued (e.g. "number of lemons"), an exhaustive sweep over every integer value ensures the discrete optimum is found exactly.

After optimization:
- **Sensitivity analysis** identifies which variables matter most to the final score.
- **Verification** confirms the result is the global optimum via random sampling, multi-start re-optimization, and boundary detection.
- **Cross-check** validates using two independent algorithms (simulated annealing + basin-hopping).

### Console Output

The optimizer provides detailed progress logging:
- Progress bars with ETA for each phase
- Per-restart timing and score tracking
- `+` new best found, `*` same score, `-` worse score
- Visual score bars (`████████░░ 8.1/10`) in the final result
- Sensitivity bars sorted by impact with directional arrows

## Requirements

- Python 3.9+
- NumPy
- SciPy

```
pip install numpy scipy
```

## Quick Start

```python
from framework import RecipeOptimizer, Variable

# 1. Define your variables with bounds
variables = [
    Variable("flour_g", 200, 500, unit="g", rounding=5),
    Variable("eggs", 1, 6, integer=True, unit="count"),
    Variable("sugar_g", 50, 200, unit="g", rounding=5),
    Variable("butter_g", 50, 200, unit="g", rounding=5),
    Variable("bake_temp_c", 150, 220, unit="C", rounding=5),
    Variable("bake_min", 20, 60, unit="min", rounding=1),
]

# 2. Write a scoring function
def score(v):
    # v is a dict: {"flour_g": 350.0, "eggs": 3.0, ...}
    # Encode your domain knowledge here: sensory models,
    # chemistry, physics, consumer preference curves, etc.
    #
    # Return (composite_score, {"criterion": score, ...})
    taste = ...
    texture = ...
    ease = ...
    composite = 0.50 * taste + 0.30 * texture + 0.20 * ease
    return composite, {"taste": taste, "texture": texture, "ease": ease}

# 3. Optimize (use all CPU cores)
opt = RecipeOptimizer("Cake", variables, score)
result = opt.run(de_restarts=10, de_popsize=100, de_maxiter=5000, n_jobs=-1)

# 4. Post-optimization
sens = opt.sensitivity(result["rounded"])
verification = opt.verify(result["rounded"], n_jobs=-1)
cross = opt.cross_check(result["rounded"], n_jobs=-1)
```

## API Reference

### `Variable`

Defines a recipe parameter with bounds.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Variable name (becomes dict key in scoring function) |
| `low` | `float` | Lower bound |
| `high` | `float` | Upper bound |
| `integer` | `bool` | If `True`, triggers Phase 3 integer sweep. Default `False` |
| `unit` | `str` | Display unit (e.g. "g", "mL", "C"). Cosmetic only |
| `rounding` | `float` | Round final value to nearest N (e.g. `5` for nearest 5g). Optional |

### `RecipeOptimizer`

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Recipe name (used in output headers) |
| `variables` | `list[Variable]` | List of recipe variables |
| `score_fn` | `callable` | Scoring function: `dict -> (float, dict)` |

#### `run(**kwargs) -> dict`

Runs the full optimization pipeline.

| Kwarg | Default | Description |
|-------|---------|-------------|
| `de_restarts` | 15 | Number of DE restarts (more = more confident in global optimum) |
| `de_popsize` | 50 | DE population size per restart |
| `de_maxiter` | 3000 | Max generations per DE restart |
| `nm_maxiter` | 100,000 | Nelder-Mead max iterations |
| `nm_maxfev` | 500,000 | Nelder-Mead max function evaluations |
| `verbose` | `True` | Print progress with bars, timing, and markers (+/*/-) |
| `n_jobs` | 1 | Parallel workers. `1` = sequential, `-1` = all CPU cores |

Returns a dict:
```python
{
    "values": {...},       # Raw optimized values
    "rounded": {...},      # Values after rounding rules applied
    "composite": float,    # Final composite score
    "scores": {...},       # Individual criterion scores
    "leaderboard": [...],  # Top integer-sweep results (if applicable)
    "stats": {...},        # Runtime stats
}
```

#### `sensitivity(values, perturbation=0.05) -> dict`

Perturbs each variable by ±5% and reports score change. Identifies which variables matter most. Results are best displayed sorted by absolute sensitivity.

#### `verify(values, n_random=10000, n_restarts=20, n_jobs=1) -> dict`

Runs three independent checks to confirm the result is the global optimum:

1. **Random sampling** — scores 10,000 random recipes; confirms none beat the candidate.
2. **Perturb and re-optimize** — starts Nelder-Mead from 20 random points; checks they all converge to the same score.
3. **Boundary detection** — flags variables whose optimal value sits at a bound, since the true optimum may lie outside the search space.

Returns a dict with results from all three checks and an overall `confident` bool.

#### `cross_check(values, ...) -> dict`

Independent cross-check using two fundamentally different global optimization algorithms:

1. **Simulated Annealing** (`scipy.optimize.dual_annealing`) — explores the space via probabilistic acceptance of worse solutions, escaping local optima through a cooling schedule.
2. **Basin-Hopping** (`scipy.optimize.basinhopping`) — combines random perturbation with local Nelder-Mead minimization.

If DE found the true global optimum, both algorithms converge to the same score. Any improvement proves the DE result was a local optimum. Output uses `+`/`*`/`-` markers like the main optimizer.

| Kwarg | Default | Description |
|-------|---------|-------------|
| `sa_restarts` | 10 | Number of dual annealing runs |
| `sa_maxiter` | 5000 | Max iterations per SA run |
| `bh_restarts` | 10 | Number of basin-hopping runs |
| `bh_niter` | 200 | Hop iterations per BH run |
| `n_jobs` | 1 | Parallel workers. `1` = sequential, `-1` = all CPU cores |

Returns a dict with `confirmed` bool and best scores from each method.

#### `weight_robustness(values, weight_perturb_fn, n_samples=100) -> dict`

Tests score stability under randomized scoring weights. Useful for checking whether the optimal recipe is robust to subjective weight choices.

## Writing a Good Scoring Function

The scoring function is where all the domain knowledge lives. The optimizer is general; the scoring function is specific.

- **Use real science.** Sensory perception follows known psychophysical laws (Stevens' power law for burn, Weber-Fechner for sweetness, Michaelis-Menten for receptor saturation). Using these produces better results than arbitrary curves.
- **Use real data.** Ground your models in published compositional data. The limoncello example uses lemon variety profiles from J. Food Sci. and Italian agricultural research for oil yield, terpene fractions, and acidity.
- **Validate against known recipes.** Before optimizing, test your scoring function on real published recipes. If a famous recipe scores poorly, your model has a bug.
- **Return criterion subscores.** This makes debugging much easier. If the optimizer makes a surprising choice, check which criterion drove it.
- **Clamp outputs to [0, 10].** The framework doesn't enforce a score range, but keeping everything on the same scale prevents one criterion from dominating.
- **Model what matters, skip what doesn't.** If careful zesting eliminates pith bitterness entirely, don't model bitterness as time-dependent — model it as pith-exposure-dependent.

## Parallel Execution

Pass `n_jobs=-1` to use all CPU cores, or `n_jobs=N` for a specific number of workers:

```python
result = opt.run(de_restarts=75, de_popsize=300, de_maxiter=10000, n_jobs=-1)
```

What gets parallelized:
- **Phase 1**: All DE restarts run simultaneously across workers
- **Phase 3**: All integer variable combinations run simultaneously
- **Cross-check**: All SA and BH restarts run simultaneously
- **Verify**: All Nelder-Mead re-optimization restarts run simultaneously

Phase 2 (single Nelder-Mead polish) always runs sequentially.

With `n_jobs=1` (default), the optimizer runs sequentially with per-generation progress bars and live ETA updates inside each restart. With `n_jobs > 1`, progress is reported per-restart as workers complete.

The scoring function must be defined at module level (not a lambda or nested function) for multiprocessing to work.

Speedup scales linearly with core count. On an 8-core machine, a 48-hour sequential run completes in ~6 hours.

## Limoncello Example

See [example.py](example.py) for a complete limoncello optimizer. It demonstrates:

- **13 variables**: lemons, variety, freshness, spirit ABV/volume, infusion days, sugar, water, zest fineness, infusion temp, spirit quality, rest days, serving temp
- **4 lemon variety profiles**: Eureka, Femminello, Meyer, Primofiore with published compositional data
- **3-channel sensory model**: orthonasal aroma (Henry's law), retronasal flavor (emulsion delivery), gustatory (Weber-Fechner)
- **Physical models**: two-compartment extraction kinetics, Arrhenius temperature dependence, emulsion stability (Stokes' law), pith-exposure bitterness, ester formation
- **Sanity checks against real recipes**: Giada De Laurentiis, Serious Eats, Sal De Riso, Bon Appetit, Italian Nonna classic

```
python example.py
```

## License

MIT
