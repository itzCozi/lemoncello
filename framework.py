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
import sys
import os
import json
import itertools

# Prevent numpy/scipy internal threading from competing with our
# explicit process-level parallelism (must be set before numpy import)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
from scipy.optimize import differential_evolution, minimize, dual_annealing, basinhopping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional, Any


def _fmt_time(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def _bar(fraction: float, width: int = 30) -> str:
    """ASCII progress bar."""
    filled = round(fraction * width)
    return f"[{'#' * filled}{'.' * (width - filled)}] {fraction*100:5.1f}%"


# ---------------------------------------------------------------------------
# Worker functions for multiprocessing (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _de_worker(score_fn, var_names, bounds, seed, maxiter, popsize):
    """Run a single DE restart in a worker process."""
    t0 = time.time()
    def objective(x):
        vals = {name: float(x[i]) for i, name in enumerate(var_names)}
        return -score_fn(vals)[0]
    result = differential_evolution(
        objective, bounds=bounds, seed=seed,
        maxiter=maxiter, popsize=popsize,
        tol=1e-13, atol=1e-13,
        mutation=(0.5, 1.5), recombination=0.9,
        polish=False, init='sobol',
    )
    return {'score': -result.fun, 'x': result.x.tolist(), 'time': time.time() - t0}


def _sweep_combo_worker(score_fn, var_names, best_x_list, all_bounds,
                        int_indices, int_values, cont_indices, cont_bounds):
    """Run optimization for one integer combo in a worker process."""
    t0 = time.time()
    best_x = np.array(best_x_list)
    def objective(x_cont):
        x_full = best_x.copy()
        for k, idx in enumerate(int_indices):
            x_full[idx] = int_values[k]
        for k, ci_idx in enumerate(cont_indices):
            x_full[ci_idx] = x_cont[k]
        vals = {name: float(x_full[i]) for i, name in enumerate(var_names)}
        return -score_fn(vals)[0]
    r = differential_evolution(
        objective, bounds=cont_bounds,
        seed=42, maxiter=300, popsize=10,
        tol=1e-8, atol=1e-8,
        mutation=(0.5, 1.5), recombination=0.9,
        polish=False, init='sobol',
    )
    r2 = minimize(
        objective, r.x, method='Nelder-Mead',
        options={'maxiter': 10000, 'maxfev': 40000,
                 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
    )
    x_full = best_x.copy()
    for k, idx in enumerate(int_indices):
        x_full[idx] = int_values[k]
    for k, ci_idx in enumerate(cont_indices):
        x_full[ci_idx] = r2.x[k]
    vals = {name: float(x_full[i]) for i, name in enumerate(var_names)}
    sc = score_fn(vals)[0]
    return {'score': sc, 'int_values': list(int_values), 'x': x_full.tolist(),
            'time': time.time() - t0}


def _sa_worker(score_fn, var_names, bounds, seed, maxiter, x0_list):
    """Run a single SA restart in a worker process."""
    t0 = time.time()
    x0 = np.array(x0_list) if x0_list is not None else None
    def objective(x):
        vals = {name: float(x[i]) for i, name in enumerate(var_names)}
        return -score_fn(vals)[0]
    result = dual_annealing(
        objective, bounds=bounds, seed=seed, maxiter=maxiter, x0=x0,
    )
    return {'score': -result.fun, 'x': result.x.tolist(), 'time': time.time() - t0}


def _bh_worker(score_fn, var_names, bounds, x0_list, niter, seed, step_size):
    """Run a single BH restart in a worker process."""
    t0 = time.time()
    x0 = np.array(x0_list)
    lb_arr = np.array([b[0] for b in bounds])
    ub_arr = np.array([b[1] for b in bounds])
    def objective(x):
        vals = {name: float(x[i]) for i, name in enumerate(var_names)}
        return -score_fn(vals)[0]
    def accept_test(**kwargs):
        x = kwargs["x_new"]
        return bool(np.all(x >= lb_arr) and np.all(x <= ub_arr))
    result = basinhopping(
        objective, x0, niter=niter, seed=seed,
        minimizer_kwargs={
            'method': 'Nelder-Mead',
            'options': {'maxiter': 50000, 'maxfev': 100000,
                       'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
        },
        stepsize=step_size, accept_test=accept_test,
    )
    return {'score': -result.fun, 'x': result.x.tolist(), 'time': time.time() - t0}


def _nm_worker(score_fn, var_names, x0_list, maxiter=50000, maxfev=200000):
    """Run a single NM restart in a worker process."""
    def objective(x):
        vals = {name: float(x[i]) for i, name in enumerate(var_names)}
        return -score_fn(vals)[0]
    result = minimize(
        objective, np.array(x0_list), method='Nelder-Mead',
        options={'maxiter': maxiter, 'maxfev': maxfev,
                 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
    )
    return {'score': -result.fun, 'x': result.x.tolist()}


def _batch_random_worker(score_fn, var_names, bounds, n_samples, seed, base_score):
    """Evaluate a batch of random recipes in a worker process."""
    rng = np.random.default_rng(seed=seed)
    best_score = -np.inf
    best_x = None
    total = 0.0
    total_sq = 0.0
    n_below_base = 0
    for _ in range(n_samples):
        x = [rng.uniform(lo, hi) for lo, hi in bounds]
        vals = {name: float(x[i]) for i, name in enumerate(var_names)}
        sc, _ = score_fn(vals)
        total += sc
        total_sq += sc * sc
        if sc < base_score:
            n_below_base += 1
        if sc > best_score:
            best_score = sc
            best_x = list(x)
    return {
        'best_score': best_score, 'best_x': best_x,
        'sum': total, 'sum_sq': total_sq,
        'n': n_samples, 'n_below_base': n_below_base,
    }


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
        n_jobs: int = 1,
    ) -> Dict[str, Any]:
        """
        Run the full 3-phase optimization pipeline.

        Phase 1: Multiple DE restarts for global search.
        Phase 2: Nelder-Mead from the best DE result.
        Phase 3: Exhaustive sweep over integer variables.

        Returns a dict with 'values', 'rounded', 'scores', 'composite',
        'leaderboard', and 'stats'.

        Parameters
        ----------
        n_jobs : int
            Number of parallel worker processes. 1 = sequential (default),
            -1 = use all CPU cores.
        """
        start = time.time()
        bounds = [v.bounds for v in self.variables]
        integer_vars = [(i, v) for i, v in enumerate(self.variables) if v.integer]

        # Convert de_popsize from total population to scipy's multiplier
        # (scipy uses popsize * n_variables as the actual population size)
        n_vars = len(self.variables)
        scipy_popsize = max(5, de_popsize // n_vars)
        actual_pop = scipy_popsize * n_vars

        _n_jobs = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        if _n_jobs > 1:
            return self._run_parallel(
                bounds, integer_vars, de_restarts, scipy_popsize, de_maxiter,
                nm_maxiter, nm_maxfev, verbose, _n_jobs, start,
                actual_pop,
            )

        n_int = len(integer_vars)
        n_cont = len(self.variables) - n_int
        total_int_combos = 1
        for _, v in integer_vars:
            total_int_combos *= int(v.high - v.low + 1)

        if verbose:
            print(f"\n{'='*65}")
            print(f"   RECIPE OPTIMIZER: {self.name}")
            print(f"{'='*65}")
            print(f"  Variables:    {len(self.variables)} ({n_cont} continuous, {n_int} integer)")
            print(f"  DE config:    {de_restarts} restarts x pop {actual_pop} x {de_maxiter} maxiter")
            if integer_vars:
                int_names = ', '.join(v.name for _, v in integer_vars)
                print(f"  Int sweep:    {total_int_combos} combos ({int_names})")
            est_evals = de_restarts * actual_pop * de_maxiter
            print(f"  Est. evals:   ~{est_evals:,.0f} (Phase 1 only)")
            print(f"{'='*65}")

        # -- Phase 1: DE global search --
        if verbose:
            print(f"\n  Phase 1: Differential Evolution")
            print(f"  {'-'*61}")

        best_score = float('inf')
        best_x = None
        p1_start = time.time()
        improvements = 0
        restart_times = []

        for i in range(de_restarts):
            restart_start = time.time()
            seed = 7 + i * 101
            self._eval_count = 0

            # Callback to show live progress within each DE restart
            _de_gen = [0]
            def _de_callback(xk, convergence=0):
                _de_gen[0] += 1
                g = _de_gen[0]
                if verbose and (g % 50 == 0 or g == 1):
                    elapsed_r = time.time() - restart_start
                    if g > 0:
                        eta_r = elapsed_r / g * (de_maxiter - g)
                    else:
                        eta_r = 0
                    sc_now = -self._objective(xk)
                    bar = _bar(g / de_maxiter, 20)
                    print(f"\r  restart {i+1}/{de_restarts}  gen {g:4d}/{de_maxiter}  "
                          f"{bar}  score={sc_now:.4f}  ({_fmt_time(elapsed_r)}, "
                          f"ETA {_fmt_time(eta_r)})   ", end="", flush=True)

            result = differential_evolution(
                self._objective,
                bounds=bounds,
                seed=seed,
                maxiter=de_maxiter,
                popsize=scipy_popsize,
                tol=1e-13,
                atol=1e-13,
                mutation=(0.5, 1.5),
                recombination=0.9,
                polish=False,
                init='sobol',
                callback=_de_callback,
            )
            restart_elapsed = time.time() - restart_start
            restart_times.append(restart_elapsed)
            sc = -result.fun
            improved = result.fun < best_score
            if improved:
                improvements += 1
                best_score = result.fun
                best_x = result.x.copy()

            if verbose:
                if sc > -best_score + 1e-8:
                    marker = ' +'
                elif abs(sc - (-best_score)) < 1e-8:
                    marker = ' *'
                else:
                    marker = ' -'
                elapsed = time.time() - start
                avg_per = elapsed / (i + 1)
                eta = avg_per * (de_restarts - i - 1)
                progress = _bar((i + 1) / de_restarts, 30)
                line = (f"\r  {i+1:3d}/{de_restarts}  {progress}  "
                        f"score={sc:.6f}  best={-best_score:.6f}  "
                        f"({_fmt_time(restart_elapsed)}, ETA {_fmt_time(eta)}){marker}")
                print(f"{line:<100}")
                sys.stdout.flush()

        p1_time = time.time() - p1_start
        if verbose:
            avg_t = sum(restart_times) / len(restart_times)
            print(f"  {'-'*61}")
            print(f"  Phase 1 done: best={-best_score:.6f}  "
                  f"({improvements} improvements in {de_restarts} restarts, "
                  f"{_fmt_time(p1_time)}, avg {_fmt_time(avg_t)}/restart)")

        # -- Phase 2: Nelder-Mead refinement --
        if verbose:
            print(f"\n  Phase 2: Nelder-Mead Local Refinement")
            print(f"  {'-'*61}")

        p2_start = time.time()
        self._eval_count = 0
        pre_nm = -best_score
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
        p2_time = time.time() - p2_start
        nm_improved = local.fun < best_score
        if nm_improved:
            best_score = local.fun
            best_x = local.x.copy()
        if verbose:
            delta = -local.fun - pre_nm
            if nm_improved:
                status = f"+ improved by {delta:.6f}"
            else:
                status = "* no improvement"
            print(f"  {self._eval_count:,} evals, {_fmt_time(p2_time)}  "
                  f"score={-local.fun:.6f} ({status})")

        # -- Phase 3: Integer variable sweep --
        leaderboard = []
        if integer_vars:
            if verbose:
                print(f"\n  Phase 3: Integer Variable Sweep")
                print(f"  {'-'*61}")

            p3_start = time.time()
            # Handle each integer variable
            for idx, var in integer_vars:
                cont_indices = [i for i in range(len(self.variables))
                                if not self.variables[i].integer]
                cont_bounds = [bounds[i] for i in cont_indices]
                n_vals = int(var.high - var.low + 1)
                sweep_best = -float('inf')
                sweep_best_val = None

                if verbose:
                    print(f"  Sweeping {var.name} ({int(var.low)}-{int(var.high)}, {n_vals} values):")

                for j, int_val in enumerate(range(int(var.low), int(var.high) + 1)):
                    val_start = time.time()
                    def obj_int(x_cont, iv=int_val, ci=cont_indices, ii=idx):
                        x_full = best_x.copy()
                        x_full[ii] = iv
                        for k, ci_idx in enumerate(ci):
                            x_full[ci_idx] = x_cont[k]
                        return self._objective(x_full)

                    _p3_gen = [0]
                    def _p3_callback(xk, convergence=0, _j=j, _iv=int_val, _nv=n_vals, _vn=var.name):
                        _p3_gen[0] += 1
                        g = _p3_gen[0]
                        if verbose and (g % 50 == 0 or g == 1):
                            el = time.time() - val_start
                            bar = _bar(g / 300, 12)
                            print(f"\r    {_vn}={_iv:2d} ({_j+1}/{_nv})  gen {g:4d}/300  "
                                  f"{bar}  ({_fmt_time(el)})   ", end="", flush=True)

                    r = differential_evolution(
                        obj_int, bounds=cont_bounds,
                        seed=42, maxiter=300, popsize=10,
                        tol=1e-8, atol=1e-8,
                        mutation=(0.5, 1.5), recombination=0.9,
                        polish=False, init='sobol',
                        callback=_p3_callback,
                    )
                    r2 = minimize(
                        obj_int, r.x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 10000, 'maxfev': 40000,
                            'xatol': 1e-10, 'fatol': 1e-12,
                            'adaptive': True,
                        }
                    )

                    x_full = best_x.copy()
                    x_full[idx] = int_val
                    for k, ci_idx in enumerate(cont_indices):
                        x_full[ci_idx] = r2.x[k]

                    sc = -self._objective(x_full)
                    leaderboard.append((sc, int_val, x_full.copy()))
                    val_time = time.time() - val_start
                    is_best = sc > sweep_best
                    if is_best:
                        sweep_best = sc
                        sweep_best_val = int_val

                    if verbose:
                        if is_best:
                            marker = ' +'
                        elif abs(sc - sweep_best) < 1e-8:
                            marker = ' *'
                        else:
                            marker = ' -'
                        progress = _bar((j + 1) / n_vals, 15)
                        line = (f"\r    {var.name}={int_val:2d}  {progress}  "
                                f"score={sc:.6f}  ({_fmt_time(val_time)}){marker}")
                        print(f"{line:<80}")
                        sys.stdout.flush()

                leaderboard.sort(reverse=True, key=lambda c: c[0])
                if leaderboard:
                    best_score = -leaderboard[0][0] if leaderboard[0][0] > -best_score else best_score
                    best_x = leaderboard[0][2].copy()

                if verbose:
                    print(f"  Best {var.name}={sweep_best_val} (score={sweep_best:.6f})")

            p3_time = time.time() - p3_start
            if verbose:
                print(f"  {'-'*61}")
                print(f"  Phase 3 done: {_fmt_time(p3_time)}")

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
            print(f"   OPTIMAL RECIPE (score={composite:.6f})")
            print(f"{'='*65}")
            print(f"  Recipe:")
            for k, v in rounded_values.items():
                unit = next((va.unit for va in self.variables if va.name == k), "")
                print(f"    {k:20s} = {v:<10} {unit}")
            print(f"\n  Scores:")
            max_k = max(len(k) for k in criterion_scores)
            for k, v in criterion_scores.items():
                bar = '#' * int(v) + '.' * (10 - int(v))
                print(f"    {k:{max_k}s}  {bar}  {v:.3f}/10")
            print(f"\n  Total time: {_fmt_time(total_time)}")
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

    def sa_search(
        self,
        x0: Optional[Dict[str, float]] = None,
        sa_restarts: int = 100,
        sa_maxiter: int = 50000,
        nm_after: bool = True,
        verbose: bool = True,
        n_jobs: int = 1,
    ) -> Dict[str, Any]:
        """
        Run simulated annealing (dual_annealing) as a primary global optimizer.

        SA explores the landscape via probabilistic trajectory moves and is
        often better than DE at finding narrow optima in mixed-integer spaces.

        Parameters
        ----------
        x0 : dict or None
            Starting recipe (seed the first restart). If None, all restarts
            start from random points.
        sa_restarts : int
            Number of independent SA runs.
        sa_maxiter : int
            Max iterations per SA run.
        nm_after : bool
            Run Nelder-Mead polish on the best SA result.
        verbose : bool
            Print progress.
        n_jobs : int
            Parallel workers. -1 = all cores.

        Returns
        -------
        dict with 'values', 'rounded', 'composite', 'scores', 'stats'.
        """
        start = time.time()
        bounds = [v.bounds for v in self.variables]
        var_names = [v.name for v in self.variables]
        _n_jobs = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)

        if x0 is not None:
            x0_arr = np.array([x0[v.name] for v in self.variables])
        else:
            x0_arr = None

        if verbose:
            print(f"\n{'='*65}")
            print(f"   SA SEARCH: {self.name} ({sa_restarts} restarts, {_n_jobs} workers)")
            print(f"{'='*65}")

        best_score = -float('inf')
        best_x = None
        improvements = 0

        with ProcessPoolExecutor(max_workers=_n_jobs) as pool:
            futures = {}
            seed_rng = np.random.default_rng(seed=77)
            ranges = np.array([b[1] - b[0] for b in bounds])
            lb = np.array([b[0] for b in bounds])
            ub = np.array([b[1] for b in bounds])
            for i in range(sa_restarts):
                seed = 31 + i * 73
                # Seed 50% of workers from x0 with small perturbation
                if x0_arr is not None and i < sa_restarts // 2:
                    if i == 0:
                        x0_list = x0_arr.tolist()
                    else:
                        pert = seed_rng.normal(0, 0.03) * ranges
                        x0_list = np.clip(x0_arr + pert, lb, ub).tolist()
                else:
                    x0_list = None
                fut = pool.submit(
                    _sa_worker, self.score_fn, var_names, bounds,
                    seed, sa_maxiter, x0_list,
                )
                futures[fut] = i

            completed = 0
            pending = set(futures.keys())
            while pending:
                import concurrent.futures as _cf
                done, pending = _cf.wait(pending, timeout=2.0,
                                         return_when=_cf.FIRST_COMPLETED)
                if not done and verbose:
                    elapsed = time.time() - start
                    progress = _bar(completed / sa_restarts, 30)
                    active = min(_n_jobs, sa_restarts - completed)
                    eta_str = ""
                    if completed > 0:
                        per = elapsed / completed
                        remaining = (sa_restarts - completed) / _n_jobs * per
                        eta_str = f"  ETA {_fmt_time(remaining)}"
                    print(f"\r  {completed:3d}/{sa_restarts}  {progress}  "
                          f"running {active} workers...  "
                          f"({_fmt_time(elapsed)}){eta_str}          ",
                          end="", flush=True)
                    continue
                for fut in done:
                    completed += 1
                    res = fut.result()
                    sc = res['score']
                    x = np.array(res['x'])

                    if sc > best_score + 1e-8:
                        marker = ' +'
                        improvements += 1
                        best_score = sc
                        best_x = x.copy()
                    elif abs(sc - best_score) < 1e-8:
                        marker = ' *'
                    else:
                        marker = ' -'

                    if verbose:
                        elapsed = time.time() - start
                        eta = elapsed / completed * (sa_restarts - completed) / _n_jobs if completed > 0 else 0
                        progress = _bar(completed / sa_restarts, 30)
                        line = (f"\r  {completed:3d}/{sa_restarts}  {progress}  "
                                f"score={sc:.6f}  best={best_score:.6f}  "
                                f"({_fmt_time(res['time'])}, ETA {_fmt_time(eta)}){marker}")
                        print(f"{line:<100}")
                        sys.stdout.flush()

        sa_time = time.time() - start
        if verbose:
            print(f"  {'-'*61}")
            print(f"  SA done: best={best_score:.6f}  "
                  f"({improvements} improvements in {sa_restarts} restarts, "
                  f"{_fmt_time(sa_time)})")

        # Optional NM polish (parallel multi-start from perturbed points)
        if nm_after and best_x is not None:
            if _n_jobs > 1:
                if verbose:
                    print(f"\n  Nelder-Mead polish ({_n_jobs} parallel starts)...")
                pre_nm = best_score
                nm_rng = np.random.default_rng(seed=1234)
                ranges = np.array([b[1] - b[0] for b in bounds])
                lb = np.array([b[0] for b in bounds])
                ub = np.array([b[1] for b in bounds])
                nm_x0s = [best_x.tolist()]
                for _ in range(_n_jobs - 1):
                    pert = nm_rng.normal(0, 0.005) * ranges
                    nm_x0s.append(np.clip(best_x + pert, lb, ub).tolist())
                with ProcessPoolExecutor(max_workers=_n_jobs) as pool:
                    nm_futs = [
                        pool.submit(_nm_worker, self.score_fn, var_names, x0,
                                    100000, 500000)
                        for x0 in nm_x0s
                    ]
                    nm_results = [f.result() for f in nm_futs]
                best_nm = max(nm_results, key=lambda r: r['score'])
                nm_score = best_nm['score']
                if nm_score > best_score + 1e-10:
                    best_score = nm_score
                    best_x = np.array(best_nm['x'])
                if verbose:
                    delta = nm_score - pre_nm
                    status = f"+ improved by {delta:.6f}" if delta > 1e-10 else "* no improvement"
                    print(f"  {_n_jobs} starts  score={nm_score:.6f} ({status})")
            else:
                if verbose:
                    print(f"\n  Nelder-Mead polish from SA best...")
                self._eval_count = 0
                pre_nm = best_score
                local = minimize(
                    self._objective, best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100000,
                        'maxfev': 500000,
                        'xatol': 1e-12,
                        'fatol': 1e-14,
                        'adaptive': True,
                    }
                )
                nm_score = -local.fun
                if nm_score > best_score + 1e-10:
                    best_score = nm_score
                    best_x = local.x.copy()
                if verbose:
                    delta = nm_score - pre_nm
                    status = f"+ improved by {delta:.6f}" if delta > 1e-10 else "* no improvement"
                    print(f"  {self._eval_count:,} evals  score={nm_score:.6f} ({status})")

        total_time = time.time() - start
        raw_values = self._values_from_array(best_x)
        rounded_values = self.round_values(raw_values)
        composite, criterion_scores = self.score_fn(rounded_values)

        if verbose:
            print(f"\n{'='*65}")
            print(f"   SA RESULT (score={composite:.6f})")
            print(f"{'='*65}")
            print(f"  Recipe:")
            for k, v in rounded_values.items():
                unit = next((va.unit for va in self.variables if va.name == k), "")
                print(f"    {k:20s} = {v:<10} {unit}")
            print(f"\n  Scores:")
            max_k = max(len(k) for k in criterion_scores)
            for k, v in criterion_scores.items():
                bar = '#' * int(v) + '.' * (10 - int(v))
                print(f"    {k:{max_k}s}  {bar}  {v:.3f}/10")
            print(f"\n  Total time: {_fmt_time(total_time)}")
            print(f"{'='*65}")

        return {
            'values': raw_values,
            'rounded': rounded_values,
            'composite': composite,
            'scores': criterion_scores,
            'stats': {
                'total_time_s': round(total_time, 1),
                'sa_restarts': sa_restarts,
                'improvements': improvements,
                'n_jobs': _n_jobs,
            },
        }

    def verify(
        self,
        values: Dict[str, float],
        n_random: int = 10000,
        n_restarts: int = 20,
        verbose: bool = True,
        n_jobs: int = 1,
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
        var_names = [v.name for v in self.variables]
        _n_jobs = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        issues = []

        if verbose:
            print(f"\n{'='*65}")
            print(f"  VERIFICATION (base score: {base_score:.6f}, {_n_jobs} workers)")
            print(f"{'='*65}")

        # -- Check 1: Random sampling (parallel) --
        if verbose:
            print(f"\n  1. Random sampling ({n_random:,} recipes, {_n_jobs} workers)...", flush=True)

        if _n_jobs > 1:
            chunk_size = (n_random + _n_jobs - 1) // _n_jobs
            with ProcessPoolExecutor(max_workers=_n_jobs) as pool:
                futures = []
                for i in range(_n_jobs):
                    n_chunk = min(chunk_size, n_random - i * chunk_size)
                    if n_chunk <= 0:
                        break
                    fut = pool.submit(
                        _batch_random_worker, self.score_fn, var_names,
                        bounds, n_chunk, 42 + i * 997, base_score,
                    )
                    futures.append(fut)
                results = [f.result() for f in futures]

            best_random_score = max(r['best_score'] for r in results)
            best_result = max(results, key=lambda r: r['best_score'])
            best_random_values = self._values_from_array(np.array(best_result['best_x']))
            total_sum = sum(r['sum'] for r in results)
            total_sum_sq = sum(r['sum_sq'] for r in results)
            total_n = sum(r['n'] for r in results)
            total_below = sum(r['n_below_base'] for r in results)
            mean_random = total_sum / total_n
            std_random = max(0.0, total_sum_sq / total_n - mean_random ** 2) ** 0.5
            percentile = total_below / total_n * 100
        else:
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
            mean_random = float(np.mean(random_scores))
            std_random = float(np.std(random_scores))
            percentile = float(np.sum(random_scores < base_score) / n_random * 100)

        random_beat = bool(best_random_score > base_score + 1e-6)
        if random_beat:
            issues.append("random_sample_beat_candidate")

        if verbose:
            print(f"     Best random:  {best_random_score:.6f}")
            print(f"     Mean random:  {mean_random:.4f} "
                  f"(std {std_random:.4f})")
            print(f"     Candidate is better than {percentile:.1f}% "
                  f"of random samples")
            if random_beat:
                print(f"     WARNING: random sample beat candidate by "
                      f"{best_random_score - base_score:.6f}")

        # -- Check 2: Perturb and re-optimize --
        if verbose:
            print(f"\n  2. Perturb and re-optimize ({n_restarts} random starts, {_n_jobs} workers)...", flush=True)

        rng = np.random.default_rng(seed=42)
        if _n_jobs > 1:
            x0_lists = [
                np.array([rng.uniform(lo, hi) for lo, hi in bounds]).tolist()
                for _ in range(n_restarts)
            ]
            with ProcessPoolExecutor(max_workers=_n_jobs) as pool:
                futures = [
                    pool.submit(_nm_worker, self.score_fn, var_names, x0)
                    for x0 in x0_lists
                ]
                converged_scores = [fut.result()['score'] for fut in futures]
        else:
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
                'mean': mean_random,
                'std': std_random,
                'percentile': percentile,
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
        n_jobs: int = 1,
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

        _n_jobs = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        if _n_jobs > 1:
            return self._cross_check_parallel(
                values, bounds, x0, base_score,
                sa_restarts, sa_maxiter, bh_restarts, bh_niter,
                verbose, _n_jobs,
            )

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
            _sa_iter = [0]
            _sa_start = time.time()
            def _sa_callback(x, f, context):
                _sa_iter[0] += 1
                n = _sa_iter[0]
                if verbose and (n % 100 == 0 or n == 1):
                    el = time.time() - _sa_start
                    sc_now = -f
                    bar = _bar(n / sa_maxiter, 15)
                    print(f"\r    SA {i+1}/{sa_restarts}  iter {n:4d}/{sa_maxiter}  "
                          f"{bar}  score={sc_now:.4f}  ({_fmt_time(el)})   ", end="", flush=True)
            result = dual_annealing(
                self._objective,
                bounds=bounds,
                seed=seed,
                maxiter=sa_maxiter,
                x0=x0 if i == 0 else None,
                callback=_sa_callback,
            )
            sc = -result.fun
            improved = result.fun < sa_best_score
            if improved:
                sa_best_score = result.fun
                sa_best_x = result.x.copy()
            if verbose:
                if sc > -sa_best_score + 1e-8:
                    marker = ' +'
                elif abs(sc - (-sa_best_score)) < 1e-8:
                    marker = ' *'
                else:
                    marker = ' -'
                line = f"\r    SA {i+1:2d}/{sa_restarts}: score={sc:.6f}  best={-sa_best_score:.6f}{marker}"
                print(f"{line:<80}")
                sys.stdout.flush()

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

            _bh_iter = [0]
            _bh_start = time.time()
            def _bh_callback(x, f, accepted):
                _bh_iter[0] += 1
                n = _bh_iter[0]
                if verbose and (n % 10 == 0 or n == 1):
                    el = time.time() - _bh_start
                    sc_now = -f
                    bar = _bar(n / bh_niter, 15)
                    print(f"\r    BH {i+1}/{bh_restarts}  hop {n:3d}/{bh_niter}  "
                          f"{bar}  score={sc_now:.4f}  ({_fmt_time(el)})   ", end="", flush=True)
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
                callback=_bh_callback,
            )
            sc = -result.fun
            improved = result.fun < bh_best_score
            if improved:
                bh_best_score = result.fun
                bh_best_x = result.x.copy()
            if verbose:
                if sc > -bh_best_score + 1e-8:
                    marker = ' +'
                elif abs(sc - (-bh_best_score)) < 1e-8:
                    marker = ' *'
                else:
                    marker = ' -'
                line = f"\r    BH {i+1:2d}/{bh_restarts}: score={sc:.6f}  best={-bh_best_score:.6f}{marker}"
                print(f"{line:<80}")
                sys.stdout.flush()

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

    # ------------------------------------------------------------------
    # Parallel implementations
    # ------------------------------------------------------------------

    def _run_parallel(self, bounds, integer_vars, de_restarts, de_popsize,
                      de_maxiter, nm_maxiter, nm_maxfev, verbose, n_jobs, start,
                      actual_pop=None):
        """Parallel implementation of run()."""
        n_int = len(integer_vars)
        n_cont = len(self.variables) - n_int
        total_int_combos = 1
        for _, v in integer_vars:
            total_int_combos *= int(v.high - v.low + 1)
        if actual_pop is None:
            actual_pop = de_popsize * len(self.variables)

        if verbose:
            print(f"\n{'='*65}")
            print(f"   RECIPE OPTIMIZER: {self.name}")
            print(f"{'='*65}")
            print(f"  Variables:    {len(self.variables)} ({n_cont} continuous, {n_int} integer)")
            print(f"  DE config:    {de_restarts} restarts x pop {actual_pop} x {de_maxiter} maxiter")
            print(f"  Parallel:     {n_jobs} workers")
            if integer_vars:
                int_names = ', '.join(v.name for _, v in integer_vars)
                print(f"  Int sweep:    {total_int_combos} combos ({int_names})")
            est_evals = de_restarts * actual_pop * de_maxiter
            print(f"  Est. evals:   ~{est_evals:,.0f} (Phase 1 only)")
            print(f"{'='*65}")

        var_names = [v.name for v in self.variables]

        # -- Calibration: run a mini-DE to measure actual restart speed --
        est_restart_s = None
        if verbose:
            try:
                _cal_count = [0]
                _cal_times = []
                _cal_t0 = [time.time()]
                def _cal_callback(xk, convergence=0):
                    _cal_count[0] += 1
                    now = time.time()
                    _cal_times.append(now - _cal_t0[0])
                    _cal_t0[0] = now
                def _cal_obj(x):
                    vals = {name: float(x[i]) for i, name in enumerate(var_names)}
                    return -self.score_fn(vals)[0]
                differential_evolution(
                    _cal_obj, bounds=bounds, seed=999,
                    maxiter=20, popsize=de_popsize,
                    tol=0, atol=0,
                    mutation=(0.5, 1.5), recombination=0.9,
                    polish=False, init='sobol',
                    callback=_cal_callback,
                )
                # Use last 15 gens (skip first 5 for init overhead)
                steady = _cal_times[5:] if len(_cal_times) > 10 else _cal_times
                avg_gen_s = sum(steady) / len(steady) if steady else 0.1
                # Calibration ran single-threaded; parallel workers compete for
                # CPU cache/memory bandwidth, so scale up by ~1.5x
                parallel_overhead = 1.5 if n_jobs > 1 else 1.0
                est_restart_s = avg_gen_s * de_maxiter * parallel_overhead
                total_batches = (de_restarts + n_jobs - 1) // n_jobs
                est_phase1 = est_restart_s * total_batches
                print(f"  Calibration:  {avg_gen_s*1000:.0f}ms/gen  "
                      f"(~{_fmt_time(est_restart_s)} per restart)")
                print(f"  Est. Phase 1: ~{_fmt_time(est_phase1)}  "
                      f"(first results in ~{_fmt_time(est_restart_s)})")
                print(f"{'='*65}")
            except Exception:
                print(f"{'='*65}")

        # -- Phase 1: Parallel DE --
        if verbose:
            print(f"\n  Phase 1: Differential Evolution ({n_jobs} workers)")
            print(f"  {'-'*61}")

        best_score = -float('inf')
        best_x = None
        p1_start = time.time()
        improvements = 0

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {}
            for i in range(de_restarts):
                seed = 7 + i * 101
                fut = pool.submit(
                    _de_worker, self.score_fn, var_names, bounds,
                    seed, de_maxiter, de_popsize,
                )
                futures[fut] = i

            completed = 0
            pending = set(futures.keys())
            while pending:
                import concurrent.futures as _cf
                done, pending = _cf.wait(pending, timeout=2.0,
                                         return_when=_cf.FIRST_COMPLETED)
                if not done and verbose:
                    elapsed = time.time() - p1_start
                    progress = _bar(completed / de_restarts, 30)
                    active = min(n_jobs, de_restarts - completed)
                    eta_str = ""
                    if est_restart_s and completed == 0:
                        eta_str = f"  ~{_fmt_time(est_restart_s)} per restart"
                    elif completed > 0:
                        per_restart = elapsed / completed
                        remaining = (de_restarts - completed) / n_jobs * per_restart
                        eta_str = f"  ETA {_fmt_time(remaining)}"
                    print(f"\r  {completed:3d}/{de_restarts}  {progress}  "
                          f"running {active} workers...  "
                          f"({_fmt_time(elapsed)}){eta_str}          ",
                          end="", flush=True)
                    continue
                for fut in done:
                    completed += 1
                    res = fut.result()
                    sc = res['score']
                    x = np.array(res['x'])

                    if sc > best_score + 1e-8:
                        marker = ' +'
                        improvements += 1
                        best_score = sc
                        best_x = x.copy()
                    elif abs(sc - best_score) < 1e-8:
                        marker = ' *'
                    else:
                        marker = ' -'

                    if verbose:
                        elapsed = time.time() - start
                        eta = elapsed / completed * (de_restarts - completed)
                        progress = _bar(completed / de_restarts, 30)
                        line = (f"\r  {completed:3d}/{de_restarts}  {progress}  "
                                f"score={sc:.6f}  best={best_score:.6f}  "
                                f"({_fmt_time(res['time'])}, ETA {_fmt_time(eta)}){marker}")
                        print(f"{line:<100}")
                        sys.stdout.flush()

        p1_time = time.time() - p1_start
        if verbose:
            print(f"  {'-'*61}")
            print(f"  Phase 1 done: best={best_score:.6f}  "
                  f"({improvements} improvements in {de_restarts} restarts, "
                  f"{_fmt_time(p1_time)})")

        # -- Phase 2: Nelder-Mead (parallel multi-start) --
        if verbose:
            print(f"\n  Phase 2: Nelder-Mead Local Refinement ({n_jobs} parallel starts)")
            print(f"  {'-'*61}")

        p2_start = time.time()
        pre_nm = best_score
        nm_rng = np.random.default_rng(seed=9999)
        ranges = np.array([b[1] - b[0] for b in bounds])
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        nm_x0s = [best_x.tolist()]
        for _ in range(n_jobs - 1):
            pert = nm_rng.normal(0, 0.005) * ranges
            nm_x0s.append(np.clip(best_x + pert, lb, ub).tolist())
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            nm_futs = [
                pool.submit(_nm_worker, self.score_fn, var_names, x0,
                            nm_maxiter, nm_maxfev)
                for x0 in nm_x0s
            ]
            nm_results = [f.result() for f in nm_futs]
        best_nm = max(nm_results, key=lambda r: r['score'])
        p2_time = time.time() - p2_start
        nm_score = best_nm['score']
        nm_improved = nm_score > best_score + 1e-10
        if nm_improved:
            best_score = nm_score
            best_x = np.array(best_nm['x'])
        if verbose:
            delta = nm_score - pre_nm
            status = f"+ improved by {delta:.6f}" if nm_improved else "* no improvement"
            print(f"  {n_jobs} starts, {_fmt_time(p2_time)}  "
                  f"score={nm_score:.6f} ({status})")

        # -- Phase 3: Parallel Integer Sweep --
        leaderboard = []
        if integer_vars:
            if verbose:
                print(f"\n  Phase 3: Integer Sweep ({total_int_combos} combos, {n_jobs} workers)")
                print(f"  {'-'*61}")

            p3_start = time.time()
            int_indices = [idx for idx, _ in integer_vars]
            cont_indices = [i for i in range(len(self.variables))
                            if not self.variables[i].integer]
            cont_bounds = [bounds[i] for i in cont_indices]

            int_ranges = [range(int(v.low), int(v.high) + 1) for _, v in integer_vars]
            combos = list(itertools.product(*int_ranges))

            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                futures = {}
                for combo in combos:
                    fut = pool.submit(
                        _sweep_combo_worker, self.score_fn, var_names,
                        best_x.tolist(), bounds, int_indices, list(combo),
                        cont_indices, cont_bounds,
                    )
                    futures[fut] = combo

                sweep_best = -float('inf')
                completed = 0
                pending = set(futures.keys())
                while pending:
                    import concurrent.futures as _cf
                    done, pending = _cf.wait(pending, timeout=2.0,
                                             return_when=_cf.FIRST_COMPLETED)
                    if not done and verbose:
                        elapsed = time.time() - p3_start
                        progress = _bar(completed / len(combos), 15)
                        active = min(n_jobs, len(combos) - completed)
                        print(f"\r    {completed:3d}/{len(combos)} {progress} "
                              f"{active} workers ({_fmt_time(elapsed)})   ",
                              end="", flush=True)
                        continue
                    for fut in done:
                        completed += 1
                        res = fut.result()
                        sc = res['score']
                        combo = futures[fut]
                        x_full = np.array(res['x'])
                        leaderboard.append((sc, combo, x_full.copy()))

                        if sc > sweep_best + 1e-8:
                            marker = ' +'
                            sweep_best = sc
                        elif abs(sc - sweep_best) < 1e-8:
                            marker = ' *'
                        else:
                            marker = ' -'

                        if verbose:
                            combo_str = ', '.join(
                                f"{integer_vars[k][1].name}={combo[k]}"
                                for k in range(len(integer_vars))
                            )
                            progress = _bar(completed / len(combos), 15)
                            line = (f"\r    {completed:3d}/{len(combos)}  {progress}  "
                                    f"{combo_str}  score={sc:.6f}  "
                                    f"({_fmt_time(res['time'])}){marker}")
                            print(f"{line:<100}")
                            sys.stdout.flush()

            leaderboard.sort(reverse=True, key=lambda x: x[0])
            if leaderboard and leaderboard[0][0] > best_score:
                best_score = leaderboard[0][0]
                best_x = leaderboard[0][2].copy()

            p3_time = time.time() - p3_start
            if verbose:
                best_combo = leaderboard[0][1]
                combo_str = ', '.join(
                    f"{integer_vars[k][1].name}={best_combo[k]}"
                    for k in range(len(integer_vars))
                )
                print(f"  {'-'*61}")
                print(f"  Phase 3 done: best={leaderboard[0][0]:.6f} "
                      f"({combo_str}), {_fmt_time(p3_time)}")

        total_time = time.time() - start

        # Build result
        raw_values = self._values_from_array(best_x)
        rounded_values = self.round_values(raw_values)
        composite, criterion_scores = self.score_fn(rounded_values)

        result = {
            'values': raw_values,
            'rounded': rounded_values,
            'composite': composite,
            'scores': criterion_scores,
            'leaderboard': [
                (sc, {integer_vars[k][1].name: combo[k]
                      for k in range(len(integer_vars))})
                for sc, combo, _ in leaderboard[:10]
            ],
            'stats': {
                'total_time_s': round(total_time, 1),
                'de_restarts': de_restarts,
                'n_jobs': n_jobs,
            },
        }

        if verbose:
            print(f"\n{'='*65}")
            print(f"   OPTIMAL RECIPE (score={composite:.6f})")
            print(f"{'='*65}")
            print(f"  Recipe:")
            for k, v in rounded_values.items():
                unit = next((va.unit for va in self.variables if va.name == k), "")
                print(f"    {k:20s} = {v:<10} {unit}")
            print(f"\n  Scores:")
            max_k = max(len(k) for k in criterion_scores)
            for k, v in criterion_scores.items():
                bar = '#' * int(v) + '.' * (10 - int(v))
                print(f"    {k:{max_k}s}  {bar}  {v:.3f}/10")
            print(f"\n  Total time: {_fmt_time(total_time)}")
            print(f"{'='*65}")

        return result

    def _cross_check_parallel(self, values, bounds, x0, base_score,
                               sa_restarts, sa_maxiter, bh_restarts, bh_niter,
                               verbose, n_jobs):
        """Parallel implementation of cross_check()."""
        start = time.time()
        var_names = [v.name for v in self.variables]

        if verbose:
            print(f"\n{'='*65}")
            print(f"  CROSS-CHECK (base score: {base_score:.6f}, {n_jobs} workers)")
            print(f"{'='*65}")

        # -- Parallel SA --
        if verbose:
            print(f"\n  Simulated Annealing ({sa_restarts} restarts, {n_jobs} workers)...")

        sa_best_score = -float('inf')
        sa_best_x = None

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {}
            for i in range(sa_restarts):
                seed = 31 + i * 73
                x0_list = x0.tolist() if i == 0 else None
                fut = pool.submit(
                    _sa_worker, self.score_fn, var_names, bounds,
                    seed, sa_maxiter, x0_list,
                )
                futures[fut] = i

            completed = 0
            pending = set(futures.keys())
            while pending:
                import concurrent.futures as _cf
                done, pending = _cf.wait(pending, timeout=2.0,
                                         return_when=_cf.FIRST_COMPLETED)
                if not done and verbose:
                    elapsed = time.time() - start
                    progress = _bar(completed / sa_restarts, 15)
                    active = min(n_jobs, sa_restarts - completed)
                    print(f"\r    {completed:2d}/{sa_restarts}  {progress}  "
                          f"running {active} workers...  "
                          f"({_fmt_time(elapsed)})          ",
                          end="", flush=True)
                    continue
                for fut in done:
                    completed += 1
                    res = fut.result()
                    sc = res['score']
                    x = np.array(res['x'])

                    if sc > sa_best_score + 1e-8:
                        marker = ' +'
                        sa_best_score = sc
                        sa_best_x = x.copy()
                    elif abs(sc - sa_best_score) < 1e-8:
                        marker = ' *'
                    else:
                        marker = ' -'

                    if verbose:
                        progress = _bar(completed / sa_restarts, 15)
                        line = (f"\r    {completed:2d}/{sa_restarts}  {progress}  "
                                f"score={sc:.6f}  best={sa_best_score:.6f}  "
                                f"({_fmt_time(res['time'])}){marker}")
                        print(f"{line:<80}")
                        sys.stdout.flush()

        sa_score = sa_best_score
        sa_values = self._values_from_array(sa_best_x)
        sa_rounded = self.round_values(sa_values)
        if verbose:
            print(f"    SA best: {sa_score:.6f}")

        # -- Parallel BH --
        if verbose:
            print(f"\n  Basin-Hopping ({bh_restarts} restarts, {n_jobs} workers)...")

        bh_best_score = -float('inf')
        bh_best_x = None
        rng = np.random.default_rng(seed=99)
        step_sizes = np.array([(hi - lo) * 0.15 for lo, hi in bounds])
        step_size = float(np.mean(step_sizes))

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {}
            for i in range(bh_restarts):
                if i == 0:
                    x_start = x0.tolist()
                else:
                    x_start = [float(rng.uniform(lo, hi)) for lo, hi in bounds]
                seed = int(rng.integers(0, 2**31))
                fut = pool.submit(
                    _bh_worker, self.score_fn, var_names, bounds,
                    x_start, bh_niter, seed, step_size,
                )
                futures[fut] = i

            completed = 0
            bh_start = time.time()
            pending = set(futures.keys())
            while pending:
                import concurrent.futures as _cf
                done, pending = _cf.wait(pending, timeout=2.0,
                                         return_when=_cf.FIRST_COMPLETED)
                if not done and verbose:
                    elapsed = time.time() - bh_start
                    progress = _bar(completed / bh_restarts, 15)
                    active = min(n_jobs, bh_restarts - completed)
                    print(f"\r    {completed:2d}/{bh_restarts}  {progress}  "
                          f"running {active} workers...  "
                          f"({_fmt_time(elapsed)})          ",
                          end="", flush=True)
                    continue
                for fut in done:
                    completed += 1
                    res = fut.result()
                    sc = res['score']
                    x = np.array(res['x'])

                    if sc > bh_best_score + 1e-8:
                        marker = ' +'
                        bh_best_score = sc
                        bh_best_x = x.copy()
                    elif abs(sc - bh_best_score) < 1e-8:
                        marker = ' *'
                    else:
                        marker = ' -'

                    if verbose:
                        progress = _bar(completed / bh_restarts, 15)
                        line = (f"\r    {completed:2d}/{bh_restarts}  {progress}  "
                                f"score={sc:.6f}  best={bh_best_score:.6f}  "
                                f"({_fmt_time(res['time'])}){marker}")
                        print(f"{line:<80}")
                        sys.stdout.flush()
                    sys.stdout.flush()

        bh_score = bh_best_score
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
            print(f"  Time: {_fmt_time(elapsed)}")
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
