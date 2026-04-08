"""
Recipe Optimization Framework
==============================
A general-purpose system for finding the globally optimal recipe for any
food or drink. Works by combining:

  1. User-defined input variables with bounds (ingredients, process params)
  2. A user-supplied scoring function (mapping variables to a scalar quality)
  3. Global optimization via scipy differential_evolution (multiple restarts)
  4. Local refinement via Nelder-Mead
  5. Exhaustive sweep over integer variables

Usage
-----
    from framework import RecipeOptimizer, Variable

    variables = [
        Variable("flour_g", 200, 500, unit="g", rounding=5),
        Variable("eggs", 1, 6, integer=True, unit="count"),
        Variable("sugar_g", 50, 200, unit="g", rounding=5),
    ]

    def score(v):
        # v is a dict: {"flour_g": 350.0, "eggs": 3.0, "sugar_g": 120.0, ...}
        # Return (composite_score, {"taste": 9.1, "ease": 8.5, ...})
        ...

    opt = RecipeOptimizer("Cake", variables, score)
    result = opt.run()

The framework handles the optimization mechanics. All recipe-specific science
(extraction models, sensory perception, scoring criteria) lives in your
scoring function.
"""

import time
import json
import numpy as np
from scipy.optimize import differential_evolution, minimize
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional, Any


@dataclass
class Variable:
    """A recipe variable with bounds and metadata."""
    name: str
    low: float
    high: float
    integer: bool = False
    unit: str = ""
    rounding: Optional[float] = None  # round to nearest N (e.g. 5 for "nearest 5g")

    @property
    def bounds(self):
        return (self.low, self.high)


# Type alias for the scoring function
# Takes a dict of variable values, returns (composite_score, {criterion_name: score})
ScoreFn = Callable[[Dict[str, float]], Tuple[float, Dict[str, float]]]


class RecipeOptimizer:
    """
    Finds the globally optimal recipe given variables and a scoring function.

    The scoring function should accept a dict of variable values and return
    a tuple of (composite_score, criterion_scores_dict).
    """

    def __init__(self, name: str, variables: List[Variable], score_fn: ScoreFn):
        self.name = name
        self.variables = variables
        self.score_fn = score_fn
        self._eval_count = 0

    def _values_from_array(self, x: np.ndarray) -> Dict[str, float]:
        return {v.name: float(x[i]) for i, v in enumerate(self.variables)}

    def _objective(self, x: np.ndarray) -> float:
        self._eval_count += 1
        values = self._values_from_array(x)
        composite, _ = self.score_fn(values)
        return -composite

    def score(self, values: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Score a recipe. Returns (composite, {criterion: score})."""
        return self.score_fn(values)

    def round_values(self, values: Dict[str, float]) -> Dict[str, float]:
        """Round values according to variable rounding rules."""
        out = dict(values)
        for v in self.variables:
            if v.integer:
                out[v.name] = round(out[v.name])
            elif v.rounding is not None:
                out[v.name] = round(out[v.name] / v.rounding) * v.rounding
        return out

    def run(
        self,
        de_restarts: int = 15,
        de_popsize: int = 50,
        de_maxiter: int = 3000,
        nm_maxiter: int = 100000,
        nm_maxfev: int = 500000,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full 3-phase optimization pipeline.

        Phase 1: Multiple DE restarts for global search.
        Phase 2: Nelder-Mead from the best DE result.
        Phase 3: Exhaustive sweep over integer variables.

        Returns a dict with 'values', 'rounded', 'scores', 'composite',
        'leaderboard', and 'stats'.
        """
        start = time.time()
        bounds = [v.bounds for v in self.variables]
        integer_vars = [(i, v) for i, v in enumerate(self.variables) if v.integer]

        if verbose:
            print(f"\n{'='*65}")
            print(f"  RECIPE OPTIMIZER: {self.name}")
            print(f"  Variables: {len(self.variables)}")
            criteria_preview = "  (run score_fn to see criteria)"
            print(f"  DE restarts: {de_restarts}, popsize: {de_popsize}")
            print(f"{'='*65}")

        # -- Phase 1: DE global search --
        if verbose:
            print(f"\n--- Phase 1: Differential Evolution ({de_restarts} restarts) ---")

        best_score = float('inf')
        best_x = None

        for i in range(de_restarts):
            seed = 7 + i * 101
            self._eval_count = 0
            result = differential_evolution(
                self._objective,
                bounds=bounds,
                seed=seed,
                maxiter=de_maxiter,
                popsize=de_popsize,
                tol=1e-13,
                atol=1e-13,
                mutation=(0.5, 1.5),
                recombination=0.9,
                polish=False,
                init='sobol',
            )
            sc = -result.fun
            if verbose:
                vals = self._values_from_array(result.x)
                brief = "  ".join(f"{k}={v:.1f}" for k, v in list(vals.items())[:4])
                print(f"  Restart {i+1:2d}/{de_restarts}: score={sc:.6f}  {brief}")
            if result.fun < best_score:
                best_score = result.fun
                best_x = result.x.copy()

        p1_time = time.time() - start
        if verbose:
            print(f"  Phase 1 best: {-best_score:.6f} ({p1_time:.1f}s)")

        # -- Phase 2: Nelder-Mead refinement --
        if verbose:
            print(f"\n--- Phase 2: Nelder-Mead Local Refinement ---")

        self._eval_count = 0
        local = minimize(
            self._objective, best_x,
            method='Nelder-Mead',
            options={
                'maxiter': nm_maxiter,
                'maxfev': nm_maxfev,
                'xatol': 1e-12,
                'fatol': 1e-14,
                'adaptive': True,
            }
        )
        if verbose:
            print(f"  Nelder-Mead: {self._eval_count:,} evals, score={-local.fun:.6f}")
        if local.fun < best_score:
            best_score = local.fun
            best_x = local.x.copy()

        # -- Phase 3: Integer variable sweep --
        leaderboard = []
        if integer_vars:
            if verbose:
                print(f"\n--- Phase 3: Integer Variable Sweep ---")

            # Handle each integer variable
            for idx, var in integer_vars:
                cont_indices = [i for i in range(len(self.variables))
                                if not self.variables[i].integer]
                cont_bounds = [bounds[i] for i in cont_indices]

                for int_val in range(int(var.low), int(var.high) + 1):
                    def obj_int(x_cont, iv=int_val, ci=cont_indices, ii=idx):
                        x_full = best_x.copy()
                        x_full[ii] = iv
                        for j, ci_idx in enumerate(ci):
                            x_full[ci_idx] = x_cont[j]
                        return self._objective(x_full)

                    r = differential_evolution(
                        obj_int, bounds=cont_bounds,
                        seed=42, maxiter=2000, popsize=40,
                        tol=1e-13, atol=1e-13,
                        mutation=(0.5, 1.5), recombination=0.9,
                        polish=False, init='sobol',
                    )
                    r2 = minimize(
                        obj_int, r.x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50000, 'maxfev': 200000,
                            'xatol': 1e-12, 'fatol': 1e-14,
                            'adaptive': True,
                        }
                    )

                    x_full = best_x.copy()
                    x_full[idx] = int_val
                    for j, ci_idx in enumerate(cont_indices):
                        x_full[ci_idx] = r2.x[j]

                    sc = -self._objective(x_full)
                    leaderboard.append((sc, int_val, x_full.copy()))
                    if verbose:
                        print(f"  {var.name}={int_val:2d}: score={sc:.6f}")

                leaderboard.sort(reverse=True, key=lambda c: c[0])
                if leaderboard:
                    best_score = -leaderboard[0][0] if leaderboard[0][0] > -best_score else best_score
                    best_x = leaderboard[0][2].copy()

        total_time = time.time() - start

        # Build final result
        raw_values = self._values_from_array(best_x)
        rounded_values = self.round_values(raw_values)
        composite, criterion_scores = self.score_fn(rounded_values)

        result = {
            'values': raw_values,
            'rounded': rounded_values,
            'composite': composite,
            'scores': criterion_scores,
            'leaderboard': [(sc, iv) for sc, iv, _ in leaderboard[:10]],
            'stats': {
                'total_time_s': round(total_time, 1),
                'de_restarts': de_restarts,
            },
        }

        if verbose:
            print(f"\n{'='*65}")
            print(f"  RESULT: score={composite:.6f}")
            for k, v in rounded_values.items():
                unit = next((va.unit for va in self.variables if va.name == k), "")
                print(f"    {k}: {v} {unit}")
            print(f"  Criteria:")
            for k, v in criterion_scores.items():
                print(f"    {k}: {v:.3f}")
            print(f"  Time: {total_time:.1f}s")
            print(f"{'='*65}")

        return result

    def sensitivity(
        self,
        values: Dict[str, float],
        perturbation: float = 0.05,
    ) -> Dict[str, Dict[str, float]]:
        """
        Sensitivity analysis: perturb each variable by +/- perturbation fraction
        and report the score change. Helps identify which variables matter most.
        """
        base_score, _ = self.score_fn(values)
        results = {}

        for v in self.variables:
            val = values[v.name]
            delta = max(abs(val * perturbation), 0.001)

            low_vals = dict(values)
            low_vals[v.name] = max(v.low, val - delta)
            low_score, _ = self.score_fn(low_vals)

            high_vals = dict(values)
            high_vals[v.name] = min(v.high, val + delta)
            high_score, _ = self.score_fn(high_vals)

            results[v.name] = {
                'base': val,
                'low': low_vals[v.name],
                'high': high_vals[v.name],
                'score_at_low': low_score,
                'score_at_high': high_score,
                'score_base': base_score,
                'sensitivity': (high_score - low_score) / (2 * delta) if delta > 0 else 0,
            }

        return results

    def weight_robustness(
        self,
        values: Dict[str, float],
        weight_perturb_fn: Callable,
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Test how robust the recipe is to changes in scoring weights.
        weight_perturb_fn takes no args and returns a new score_fn.
        Returns stats on score distribution.
        """
        scores = []
        for _ in range(n_samples):
            perturbed_fn = weight_perturb_fn()
            sc, _ = perturbed_fn(values)
            scores.append(sc)

        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'range': float(np.max(scores) - np.min(scores)),
        }
