"""Hyperparameter sweep over train.main, scored against Gerver's sofa.

Every setting is a dict of overrides (dotted paths into TrainConfig) applied to
BASE, and runs once per seed. Four designs:

  grid      full factorial over GRID_AXES: the knobs expected to interact
            (exploration noise, step size, horizon, schedule)
  ablation  one-at-a-time over ABLATION_AXES around BASE: the remaining PPO,
            optimiser, network and env knobs, each measured on its own
  random    N_RANDOM settings sampled uniformly from GRID_AXES and ABLATION_AXES
            together, to catch interactions the other two designs miss
  all       the union of the three (duplicates removed)

Runs go to runs/<sweep>/<setting>_seed<s>/ (each with its own stdout.log); a run
whose metrics.jsonl is already complete is skipped, so a sweep can be relaunched
with the same command. Locally:

    python sweep.py --name big --mode all --dry-run
    python sweep.py --name big --mode grid --workers 8
    python sweep.py --name big --summarize-only

modal_app.py runs the same settings remotely (build_runs, run_one, is_complete,
summarize are its interface to this file).

The summary scores each setting against the known sofas at the run's own N
(eval area is measured at the training resolution, which overstates the true
area by an amount that depends on N): the gap to Gerver's area at that N, the
share of the Hammersley -> Gerver gap closed, and the mean distance between the
best checkpoint's rotation path and Gerver's.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from sofa_rl.config import NetworkConfig, TrainConfig

BASE = replace(TrainConfig(), network=NetworkConfig(log_std_init=-0.5))

# training budget as a multiple of BASE.total_steps (so modal_app.py --smoke, which
# shortens BASE, also shortens every budget)
BUDGET = "budget"

# full factorial: 4 * 4 * 4 * 2 * 2 = 256 settings
GRID_AXES: Dict[str, List[Any]] = {
    # exploration: -0.5 was best of {-3, -2, -1, -0.5}; 0 and 0.5 test whether more helps
    "network.log_std_init": [-1.0, -0.5, 0.0, 0.5],
    # 1e-3 collapsed at log_std -3; untested at high noise, as is 3e-3
    "optim.actor_lr": [1e-4, 3e-4, 1e-3, 3e-3],
    # horizon: path-level noise scales like sigma / sqrt(N), so N and log_std interact;
    # larger N also measures the area more accurately
    "env.n_steps": [100, 200, 400, 800],
    # constant keeps the step size high throughout; linear decays it to 0 by the end of the budget
    "optim.lr_schedule": ["linear", "constant"],
    # the area was still rising after 2M steps when continued from a 1M run
    BUDGET: [1, 3],
}

# one at a time around BASE: 10 axes, 28 non-base values
ABLATION_AXES: Dict[str, List[Any]] = {
    "ppo.lam": [0.8, 0.9, 0.95, 0.99, 1.0],
    "ppo.clip_eps": [0.05, 0.1, 0.2, 0.3],
    "ppo.epochs": [3, 5, 10, 20],
    # 4096 is one minibatch per epoch (full batch)
    "ppo.minibatch_size": [64, 256, 1024, 4096],
    # counteracts the shrinking std that limited the log_std -3 runs
    "ppo.entropy_coef": [0.0, 1e-3, 3e-3, 1e-2],
    # more episodes per batch: lower-variance advantages, fewer updates
    "ppo.steps_per_batch": [2048, 4096, 8192, 16384],
    "optim.critic_lr": [3e-4, 1e-3, 3e-3, 1e-2],
    "network.hidden_dim": [32, 64, 128, 256],
    # Gerver's corner moves at up to 1.42 per radian, so 1.5 is the tightest bound that contains it
    "env.max_corner_speed": [1.5, 2.0, 3.0],
    BUDGET: [1, 3],
}

# every axis with its values; modal_app.py records this in sweep.json
AXES: Dict[str, List[Any]] = {**GRID_AXES, **ABLATION_AXES}

N_RANDOM = 100
RANDOM_SEED = 0
SEEDS = [0, 1, 2]
TAIL = 10  # updates averaged for the end-of-run eval area
MAX_NAME_LENGTH = 120


def override(cfg: TrainConfig, key: str, value: Any) -> TrainConfig:
    """replace() one field given a dotted path, e.g. 'network.log_std_init'."""
    head, _, rest = key.partition(".")
    if not rest:
        return replace(cfg, **{head: value})
    return replace(cfg, **{head: override(getattr(cfg, head), rest, value)})


def lookup(cfg: Any, key: str) -> Any:
    for part in key.split("."):
        cfg = getattr(cfg, part)
    return cfg


def _base_value(key: str) -> Any:
    return 1 if key == BUDGET else lookup(BASE, key)


def apply_overrides(overrides: Dict[str, Any]) -> TrainConfig:
    cfg = BASE
    for key, value in overrides.items():
        if key == BUDGET:
            cfg = replace(cfg, total_steps=BASE.total_steps * value)
        else:
            cfg = override(cfg, key, value)
    return cfg


def _has_field(cfg: Any, key: str) -> bool:
    for part in key.split("."):
        if not hasattr(cfg, part):
            return False
        cfg = getattr(cfg, part)
    return True


def setting_name(overrides: Dict[str, Any]) -> str:
    """Readable `key=value,...` name, or a hash when that would be too long for a directory."""
    if not overrides:
        return "base"
    name = ",".join(f"{key.split('.')[-1]}={value}" for key, value in sorted(overrides.items()))
    if len(name) <= MAX_NAME_LENGTH:
        return name
    return "cfg-" + hashlib.sha1(name.encode()).hexdigest()[:10]


def _non_base(overrides: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in overrides.items() if v != _base_value(k)}


def grid_settings() -> List[Dict[str, Any]]:
    keys = list(GRID_AXES)
    return [_non_base(dict(zip(keys, combo))) for combo in itertools.product(*GRID_AXES.values())]


def ablation_settings() -> List[Dict[str, Any]]:
    out = [{}]
    for key, values in ABLATION_AXES.items():
        out += [{key: v} for v in values if v != _base_value(key)]
    return out


def random_settings() -> List[Dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    return [_non_base({k: rng.choice(v) for k, v in AXES.items()}) for _ in range(N_RANDOM)]


def settings(mode: str) -> List[Dict[str, Any]]:
    """Each setting is a dict of overrides applied to BASE; duplicates are removed."""
    designs = {"grid": grid_settings, "ablation": ablation_settings, "random": random_settings}
    if mode == "all":
        chosen = [grid_settings(), ablation_settings(), random_settings()]
    elif mode in designs:
        chosen = [designs[mode]()]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    out, seen = [], set()
    for overrides in itertools.chain(*chosen):
        name = setting_name(overrides)
        if name not in seen:
            seen.add(name)
            out.append(overrides)
    return out


def build_runs(sweep: str, mode: str, seeds: List[int]) -> List[Tuple[str, TrainConfig]]:
    runs = []
    for overrides in settings(mode):
        cfg = apply_overrides(overrides)
        name = setting_name(overrides)
        for seed in seeds:
            runs.append((name, replace(cfg, seed=seed, run_name=f"{sweep}/{name}_seed{seed}")))
    return runs


def n_updates(cfg: TrainConfig) -> int:
    return cfg.total_steps // cfg.ppo.steps_per_batch


def is_complete(cfg: TrainConfig, runs_dir: Path) -> bool:
    metrics = runs_dir / cfg.run_name / "metrics.jsonl"
    if not metrics.exists():
        return False
    with open(metrics) as f:
        return sum(1 for _ in f) >= n_updates(cfg)


def run_one(cfg: TrainConfig) -> str:
    import torch
    import train

    torch.set_num_threads(1)
    run_dir = train.RUNS_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "stdout.log", "w") as log, contextlib.redirect_stdout(log):
        train.main(cfg)
    return cfg.run_name


@lru_cache(maxsize=None)
def _reference(n_steps: int) -> Tuple[float, float, np.ndarray]:
    """(Hammersley area, Gerver area, Gerver path) at this N."""
    from sofa_env import gerver_path, hammersley_path, replay

    gerver = gerver_path(n_steps)
    return replay(hammersley_path(n_steps)).area, replay(gerver).area, gerver


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _run_result(run_dir: Path) -> Dict[str, Any]:
    with open(run_dir / "metrics.jsonl") as f:
        evals = np.array([json.loads(line)["eval_area"] for line in f])
    cfg = json.loads((run_dir / "config.json").read_text())
    n_steps = cfg["env"]["n_steps"]
    ham, ger, ger_path = _reference(n_steps)
    tail = float(evals[-TAIL:].mean())

    path_dist = np.nan
    best = run_dir / "best.pt"
    if best.exists():
        import torch

        update = torch.load(best, weights_only=False)["update"]
        corners = np.load(run_dir / "paths" / f"update_{update:04d}.npz")["corners"]
        if corners.shape == ger_path.shape:  # an episode whose sofa reached area 0 ends early
            path_dist = float(np.linalg.norm(corners - ger_path, axis=1).mean())

    overrides = {
        k: v for k, v in _flatten(cfg).items()
        if k not in ("seed", "run_name") and _has_field(BASE, k) and v != lookup(BASE, k)
    }
    return {
        "overrides": overrides,
        "tail": tail,
        "best": float(evals.max()),
        "gap_to_gerver": ger - tail,
        "frac_closed": (tail - ham) / (ger - ham),
        "path_dist": path_dist,
        "updates": len(evals),
    }


def summarize(sweep_dir: Path) -> List[Dict[str, Any]]:
    """One row per setting, aggregated over seeds; written to summary.csv, best first."""
    by_setting: Dict[str, List[Dict[str, Any]]] = {}
    for run_dir in sorted(p for p in sweep_dir.iterdir() if (p / "metrics.jsonl").exists()):
        if (run_dir / "metrics.jsonl").stat().st_size == 0:
            continue
        setting, _, _ = run_dir.name.rpartition("_seed")
        by_setting.setdefault(setting, []).append(_run_result(run_dir))
    if not by_setting:
        print(f"no finished runs in {sweep_dir}")
        return []

    override_keys = sorted({k for runs in by_setting.values() for r in runs for k in r["overrides"]})
    rows = []
    for setting, runs in by_setting.items():
        tail = np.array([r["tail"] for r in runs])
        row = {
            "setting": setting,
            "seeds": len(runs),
            "tail_mean": tail.mean(),
            "tail_std": tail.std(ddof=1) if len(runs) > 1 else 0.0,
            "tail_min": tail.min(),
            "best_mean": np.mean([r["best"] for r in runs]),
            "gap_to_gerver": np.mean([r["gap_to_gerver"] for r in runs]),
            "frac_closed": np.mean([r["frac_closed"] for r in runs]),
            "frac_closed_min": np.min([r["frac_closed"] for r in runs]),
            "path_dist_to_gerver": np.nanmean([r["path_dist"] for r in runs]),
            "updates": min(r["updates"] for r in runs),
        }
        row.update({k: runs[0]["overrides"].get(k, lookup(BASE, k)) for k in override_keys})
        rows.append(row)
    rows.sort(key=lambda r: r["gap_to_gerver"])

    with open(sweep_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'setting':60s} seeds  eval(last {TAIL}) mean ± std  gap to Gerver  closed (worst seed)  path dist")
    for r in rows:
        print(f"{r['setting'][:60]:60s} {r['seeds']:5d}  {r['tail_mean']:.4f} ± {r['tail_std']:.4f}"
              f"     {r['gap_to_gerver']:+.4f}   {100 * r['frac_closed']:5.0f}% ({100 * r['frac_closed_min']:4.0f}%)"
              f"     {r['path_dist_to_gerver']:.3f}")
    return rows


def main():
    import train

    global N_RANDOM
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="sweep directory under runs/")
    parser.add_argument("--mode", choices=["grid", "ablation", "random", "all"], default="grid")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--n-random", type=int, default=N_RANDOM)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    N_RANDOM = args.n_random

    sweep_dir = train.RUNS_DIR / args.name
    if args.summarize_only:
        summarize(sweep_dir)
        return

    runs = build_runs(args.name, args.mode, args.seeds)
    todo = [cfg for _, cfg in runs if not is_complete(cfg, train.RUNS_DIR)]
    n_settings = len({name for name, _ in runs})
    print(f"{n_settings} settings x {len(args.seeds)} seeds = {len(runs)} runs "
          f"({len(runs) - len(todo)} already complete), {args.workers} workers")
    if args.dry_run:
        for name, cfg in runs:
            print(f"  {cfg.run_name}{'' if cfg in todo else '  (done)'}")
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "sweep.json").write_text(json.dumps({
        "mode": args.mode, "seeds": args.seeds, "base": asdict(BASE),
        "grid_axes": GRID_AXES, "ablation_axes": ABLATION_AXES, "axes": AXES,
        "n_random": N_RANDOM, "random_seed": RANDOM_SEED,
    }, indent=2))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, cfg): cfg.run_name for cfg in todo}
        for i, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                future.result()
                print(f"[{i}/{len(todo)}] done   {name}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(todo)}] FAILED {name}: {e!r}", flush=True)

    summarize(sweep_dir)


if __name__ == "__main__":
    main()
