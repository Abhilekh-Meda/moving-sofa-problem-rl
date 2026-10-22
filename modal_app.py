"""Run sweep.py's runs on Modal, one CPU container per run, all in parallel.

Runs are written to the Modal volume `sofa-runs` with the same layout as local
runs/ (<sweep>/<setting>_seed<s>/). Runs already complete in the volume are
skipped, so a sweep can be relaunched with the same command.

    modal run modal_app.py --name logstd_lr --mode ablation --seeds 0,1,2
    modal run modal_app.py --name smoke --smoke              # 2 updates per run
    modal volume get sofa-runs logstd_lr runs/               # pull results locally
    python sweep.py --name logstd_lr --summarize-only
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List

import modal

RUNS = "/runs"

app = modal.App("sofa-rl")
volume = modal.Volume.from_name("sofa-runs", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy==2.4.6", "gymnasium==1.3.0", "shapely==2.1.2")
    .add_local_python_source("sofa_env", "sofa_rl", "train", "sweep")
)


@app.function(image=image, volumes={RUNS: volume}, cpu=1.0, memory=1024, timeout=3 * 60 * 60)
def train_remote(cfg_dict: Dict[str, Any]) -> str:
    import sweep
    import train
    from sofa_rl.config import TrainConfig

    train.RUNS_DIR = Path(RUNS)
    name = sweep.run_one(TrainConfig.from_dict(cfg_dict))
    volume.commit()
    return name


@app.function(image=image, volumes={RUNS: volume})
def incomplete(cfg_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    import sweep
    from sofa_rl.config import TrainConfig

    return [d for d in cfg_dicts if not sweep.is_complete(TrainConfig.from_dict(d), Path(RUNS))]


@app.function(image=image, volumes={RUNS: volume})
def finish(name: str, sweep_meta: Dict[str, Any]) -> None:
    import sweep

    volume.reload()
    sweep_dir = Path(RUNS) / name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "sweep.json").write_text(json.dumps(sweep_meta, indent=2))
    sweep.summarize(sweep_dir)
    volume.commit()


@app.local_entrypoint()
def main(name: str, mode: str = "ablation", seeds: str = "0,1,2", smoke: bool = False):
    import sweep

    if smoke:
        sweep.BASE = replace(sweep.BASE, total_steps=2 * sweep.BASE.ppo.steps_per_batch)
    seed_list = [int(s) for s in seeds.split(",")]
    runs = sweep.build_runs(name, mode, seed_list)
    todo = incomplete.remote([asdict(cfg) for _, cfg in runs])
    print(f"{len(runs)} runs ({len(runs) - len(todo)} already complete in volume sofa-runs)", flush=True)

    failed = 0
    for result in train_remote.map(todo, return_exceptions=True, order_outputs=False) if todo else []:
        if isinstance(result, Exception):
            failed += 1
            print(f"FAILED: {result!r}", flush=True)
        else:
            print(f"done   {result}", flush=True)
    print(f"{len(todo) - failed}/{len(todo)} runs finished", flush=True)

    finish.remote(name, {"mode": mode, "seeds": seed_list, "base": asdict(sweep.BASE), "axes": sweep.AXES})
