# Lemoncello

A general-purpose framework for finding the globally optimal recipe for any food or drink. Define your variables, write a scoring function that encodes your sensory science, and let the optimizer find the best recipe.

The included example optimizes a limoncello recipe using real psychophysical models (Stevens' power law, Weber-Fechner, Michaelis-Menten, TRPV1 cold-gating) and chemical extraction kinetics.

## How It Works

The framework uses a 3-phase optimization pipeline:

1. **Global search** Differential evolution with multiple restarts (Sobol initialization) to explore the full parameter space and avoid local optima.
2. **Local refinement** Adaptive Nelder-Mead simplex to polish the best candidate to high precision.
3. **Integer sweep** If any variables are integer-valued (e.g. "number of eggs"), an exhaustive sweep over every integer value ensures the discrete optimum is found exactly.

After optimization, a sensitivity analysis identifies which variables matter most to the final score.

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

# 3. Optimize
opt = RecipeOptimizer("Cake", variables, score)
result = opt.run()

# 4. Sensitivity analysis
sens = opt.sensitivity(result["rounded"])
for name, info in sens.items():
    print(f"  {name}: sensitivity = {info['sensitivity']:.4f}")
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
| `verbose` | `True` | Print progress to stdout |

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

Perturbs each variable by +/- 5% and reports score change. Identifies which variables matter most.

#### `weight_robustness(values, weight_perturb_fn, n_samples=100) -> dict`

Tests score stability under randomized scoring weights. Useful for checking whether the optimal recipe is robust to subjective weight choices.

## Writing a Good Scoring Function

The scoring function is where all the domain knowledge lives. The optimizer is general; the scoring function is specific. Some tips:

- **Use real science.** Sensory perception follows known psychophysical laws (Stevens' power law for burn, Weber-Fechner for sweetness, Michaelis-Menten for saturation). Using these produces better results than arbitrary curves.
- **Return criterion subscores.** This makes debugging and interpretation much easier. If the optimizer makes a surprising choice, check which criterion drove it.
- **Add sanity checks.** Before optimizing, test your scoring function on a few known recipes to verify it produces reasonable scores.
- **Clamp outputs to [0, 10].** The framework does not enforce a score range, but keeping everything on the same scale prevents one criterion from dominating.

## Example

See [example.py](example.py) for a complete, runnable limoncello optimizer with real sensory models. It demonstrates two-compartment oil extraction kinetics, Stevens' power law burn perception, Weber-Fechner sweetness, and pith bitterness modeling.

```
python example.py
```

## License

MIT
