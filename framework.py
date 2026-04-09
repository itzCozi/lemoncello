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
from scipy.optimize import differential_evolution, minimize, dual_annealing, basinhopping
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

    def verify(
        self,
        values: Dict[str, float],
        n_random: int = 10000,
        n_restarts: int = 20,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Verify that the given recipe is the global optimum by running
        three independent checks:

        1. Random sampling: score n_random random recipes and check
           whether any beat the candidate.
        2. Perturb and re-optimize: start Nelder-Mead from n_restarts
           random points across the full search space. If they all
           converge back to the same score, there are no competing
           local optima.
        3. Boundary detection: flag any variable whose optimal value
           sits at or near its bound, since the true optimum may lie
           outside the search space.

        Returns a dict with results from all three checks and an
        overall 'confident' bool.
        """
        base_score, _ = self.score_fn(values)
        bounds = [v.bounds for v in self.variables]
        n_vars = len(self.variables)
        issues = []

        if verbose:
            print(f"\n{'='*65}")
            print(f"  VERIFICATION (base score: {base_score:.6f})")
            print(f"{'='*65}")

        # -- Check 1: Random sampling --
        if verbose:
            print(f"\n  1. Random sampling ({n_random:,} recipes)...")

        rng = np.random.default_rng(seed=42)
        random_scores = np.empty(n_random)
        best_random_score = -np.inf
        best_random_values = None

        for i in range(n_random):
            x = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
            vals = self._values_from_array(x)
            sc, _ = self.score_fn(vals)
            random_scores[i] = sc
            if sc > best_random_score:
                best_random_score = sc
                best_random_values = vals

        random_beat = bool(best_random_score > base_score + 1e-6)
        if random_beat:
            issues.append("random_sample_beat_candidate")

        if verbose:
            print(f"     Best random:  {best_random_score:.6f}")
            print(f"     Mean random:  {np.mean(random_scores):.4f} "
                  f"(std {np.std(random_scores):.4f})")
            print(f"     Candidate is better than {np.sum(random_scores < base_score)/n_random*100:.1f}% "
                  f"of random samples")
            if random_beat:
                print(f"     WARNING: random sample beat candidate by "
                      f"{best_random_score - base_score:.6f}")

        # -- Check 2: Perturb and re-optimize --
        if verbose:
            print(f"\n  2. Perturb and re-optimize ({n_restarts} random starts)...")

        converged_scores = []
        for i in range(n_restarts):
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
            result = minimize(
                self._objective, x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 50000,
                    'maxfev': 200000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
            sc = -result.fun
            converged_scores.append(sc)

        converged_scores = np.array(converged_scores)
        n_matching = int(np.sum(np.abs(converged_scores - base_score) < 0.01))
        n_better = int(np.sum(converged_scores > base_score + 1e-6))
        best_converged = float(np.max(converged_scores))

        if n_better > 0:
            issues.append("reoptimization_found_better")
        if n_matching < n_restarts * 0.5:
            issues.append("low_convergence_rate")

        if verbose:
            print(f"     Converged to candidate: {n_matching}/{n_restarts}")
            print(f"     Found better: {n_better}/{n_restarts}")
            print(f"     Best converged: {best_converged:.6f}")
            print(f"     Score range: {float(np.min(converged_scores)):.6f} "
                  f"to {best_converged:.6f}")

        # -- Check 3: Boundary detection --
        if verbose:
            print(f"\n  3. Boundary detection...")

        boundary_vars = []
        for v in self.variables:
            val = values[v.name]
            span = v.high - v.low
            margin = span * 0.02  # within 2% of bound
            at_low = val <= v.low + margin
            at_high = val >= v.high - margin
            if at_low or at_high:
                bound_str = f"lower ({v.low})" if at_low else f"upper ({v.high})"
                boundary_vars.append({
                    'name': v.name,
                    'value': val,
                    'bound': 'low' if at_low else 'high',
                    'bound_value': v.low if at_low else v.high,
                })
                if verbose:
                    print(f"     {v.name} = {val} is at {bound_str} "
                          f"-- true optimum may be outside bounds")

        if boundary_vars:
            issues.append("variables_at_bounds")

        if not boundary_vars and verbose:
            print(f"     No variables at bounds")

        # -- Summary --
        confident = len(issues) == 0
        if verbose:
            print(f"\n  {'='*63}")
            if confident:
                print(f"  VERIFIED: high confidence this is the global optimum")
                print(f"    - {n_random:,} random samples, none better")
                print(f"    - {n_matching}/{n_restarts} re-optimizations converged to same score")
                print(f"    - No variables stuck at bounds")
            else:
                print(f"  ISSUES FOUND ({len(issues)}):")
                for issue in issues:
                    print(f"    - {issue}")
            print(f"  {'='*63}")

        return {
            'base_score': base_score,
            'confident': confident,
            'issues': issues,
            'random_sampling': {
                'n_samples': n_random,
                'best_random': float(best_random_score),
                'mean': float(np.mean(random_scores)),
                'std': float(np.std(random_scores)),
                'percentile': float(np.sum(random_scores < base_score) / n_random * 100),
                'beat_candidate': random_beat,
            },
            'reoptimization': {
                'n_restarts': n_restarts,
                'n_matching': n_matching,
                'n_better': n_better,
                'best_converged': best_converged,
                'convergence_rate': n_matching / n_restarts,
            },
            'boundary': {
                'variables_at_bounds': boundary_vars,
            },
        }

    def cross_check(
        self,
        values: Dict[str, float],
        sa_restarts: int = 10,
        sa_maxiter: int = 5000,
        bh_restarts: int = 10,
        bh_niter: int = 200,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Independent cross-check using two fundamentally different global
        optimization algorithms: simulated annealing and basin-hopping.

        If DE found the true global optimum, both algorithms should converge
        to the same score. Any improvement proves the DE result was a local
        optimum.

        Parameters
        ----------
        values : dict
            The candidate recipe (usually the rounded DE result).
        sa_restarts : int
            Number of independent dual annealing runs.
        sa_maxiter : int
            Max iterations per dual annealing run.
        bh_restarts : int
            Number of independent basin-hopping runs.
        bh_niter : int
            Number of basin-hopping iterations per run.
        verbose : bool
            Print progress.

        Returns
        -------
        dict with keys:
            'base_score': the candidate's score
            'sa_best_score': best score found by simulated annealing
            'sa_best_values': corresponding recipe
            'bh_best_score': best score found by basin-hopping
            'bh_best_values': corresponding recipe
            'overall_best_score': best across both methods
            'overall_best_values': corresponding recipe
            'confirmed': True if neither method beat the candidate
            'improvement': score improvement (0 if confirmed)
        """
        start = time.time()
        base_score, _ = self.score_fn(values)
        bounds = [v.bounds for v in self.variables]
        x0 = np.array([values[v.name] for v in self.variables])

        if verbose:
            print(f"\n{'='*65}")
            print(f"  CROSS-CHECK (base score: {base_score:.6f})")
            print(f"{'='*65}")

        # -- Simulated Annealing (dual annealing) --
        if verbose:
            print(f"\n  Simulated Annealing ({sa_restarts} restarts, {sa_maxiter} maxiter)...")

        sa_best_score = float('inf')
        sa_best_x = None

        for i in range(sa_restarts):
            seed = 31 + i * 73
            result = dual_annealing(
                self._objective,
                bounds=bounds,
                seed=seed,
                maxiter=sa_maxiter,
                x0=x0 if i == 0 else None,
            )
            sc = -result.fun
            if verbose:
                print(f"    SA restart {i+1:2d}/{sa_restarts}: score={sc:.6f}")
            if result.fun < sa_best_score:
                sa_best_score = result.fun
                sa_best_x = result.x.copy()

        sa_score = -sa_best_score
        sa_values = self._values_from_array(sa_best_x)
        sa_rounded = self.round_values(sa_values)

        if verbose:
            print(f"    SA best: {sa_score:.6f}")

        # -- Basin-Hopping --
        if verbose:
            print(f"\n  Basin-Hopping ({bh_restarts} restarts, {bh_niter} iterations)...")

        bh_best_score = float('inf')
        bh_best_x = None
        rng = np.random.default_rng(seed=99)

        class BoundsEnforcer:
            def __init__(self, lb, ub):
                self.lb = np.array(lb)
                self.ub = np.array(ub)
            def __call__(self, **kwargs):
                x = kwargs["x_new"]
                return bool(np.all(x >= self.lb) and np.all(x <= self.ub))

        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]
        enforcer = BoundsEnforcer(lb, ub)

        for i in range(bh_restarts):
            if i == 0:
                x_start = x0.copy()
            else:
                x_start = np.array([rng.uniform(lo, hi) for lo, hi in bounds])

            # Step size proportional to variable range
            step_sizes = np.array([(hi - lo) * 0.15 for lo, hi in bounds])

            result = basinhopping(
                self._objective,
                x_start,
                niter=bh_niter,
                seed=int(rng.integers(0, 2**31)),
                minimizer_kwargs={
                    'method': 'Nelder-Mead',
                    'options': {
                        'maxiter': 50000,
                        'maxfev': 100000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    },
                },
                stepsize=float(np.mean(step_sizes)),
                accept_test=enforcer,
            )
            sc = -result.fun
            if verbose:
                print(f"    BH restart {i+1:2d}/{bh_restarts}: score={sc:.6f}")
            if result.fun < bh_best_score:
                bh_best_score = result.fun
                bh_best_x = result.x.copy()

        bh_score = -bh_best_score
        bh_values = self._values_from_array(bh_best_x)
        bh_rounded = self.round_values(bh_values)

        if verbose:
            print(f"    BH best: {bh_score:.6f}")

        # -- Overall --
        overall_best_score = max(sa_score, bh_score)
        if sa_score >= bh_score:
            overall_best_values = sa_rounded
        else:
            overall_best_values = bh_rounded

        confirmed = overall_best_score <= base_score + 1e-6
        improvement = max(0, overall_best_score - base_score)
        elapsed = time.time() - start

        if verbose:
            print(f"\n  {'='*63}")
            if confirmed:
                print(f"  CONFIRMED: neither method beat the DE result")
                print(f"    DE:  {base_score:.6f}")
                print(f"    SA:  {sa_score:.6f}")
                print(f"    BH:  {bh_score:.6f}")
            else:
                print(f"  IMPROVEMENT FOUND (+{improvement:.6f})")
                print(f"    DE:  {base_score:.6f}")
                print(f"    SA:  {sa_score:.6f}")
                print(f"    BH:  {bh_score:.6f}")
                print(f"    Best recipe: {overall_best_values}")
            print(f"  Time: {elapsed:.1f}s")
            print(f"  {'='*63}")

        return {
            'base_score': base_score,
            'sa_best_score': sa_score,
            'sa_best_values': sa_rounded,
            'bh_best_score': bh_score,
            'bh_best_values': bh_rounded,
            'overall_best_score': overall_best_score,
            'overall_best_values': overall_best_values,
            'confirmed': confirmed,
            'improvement': improvement,
            'time_s': round(elapsed, 1),
        }

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
