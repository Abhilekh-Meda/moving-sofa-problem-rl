# Moving sofa with reinforcement learning

The [moving sofa problem](https://en.wikipedia.org/wiki/Moving_sofa_problem) asks for the
largest-area shape that can be carried around a right-angled corner in a hallway of width 1.
Hammersley's sofa (1968) has area π/2 + 2/π ≈ 2.2074; Gerver's sofa (1992) has area
≈ 2.21953 and is believed to be optimal.

This project trains a PPO agent to find the shape and the motion together. Starting from
nothing but a block of material, it learns a Gerver-like sofa with a certified area of
**at least 2.218873**, 0.00066 below Gerver's.

## Formulation

Instead of choosing a shape and then steering it, we fix the sofa and move the hallway
around it (the *rotation path* view of Romik, 2016):

- The hallway turns from θ = 0 to π/2 in N equal steps. At step k it is
  `L_k = c_k + R(θ_k) L`: the L-shaped corridor rotated by θ_k with its inner corner moved
  to c_k.
- The agent's action moves the corner: c_k = c_{k-1} + a_k · (max speed · Δθ), with
  a_k ∈ [-1, 1]². Its observation is `[k/N, c_x, c_y]`.
- The sofa is what survives: the starting block (the strip 0 ≤ y ≤ 1 that fits through the
  horizontal arm) intersected with every hallway position, and finally with the vertical
  arm's strip. Only the largest connected piece is kept.
- The reward at each step is the area that step cut away, divided by the starting area.
  With γ = 1 the return is the final sofa area (minus a constant), so maximising return is
  maximising area. Any path gives a valid sofa; there are no collisions to avoid.

## Measuring area honestly

Cutting at N sampled angles misses material that would be cut between them, so the
sampled area overstates the true area of the continuous motion, by about 1.2 / N
(0.003 at N = 200). That is larger than the differences between good policies, so the
environment offers two corrections:

- `substeps=K`: each action is applied as K straight-line sub-moves, each with its own cut,
  so the area is measured at N·K angles while the agent still makes N decisions.
- `reward_area="certified"`: keep only points provably inside the hallway for the whole
  continuous motion of each step, giving a lower bound on the true area. The bound uses the
  maximum overshoot of a smooth function between two samples (|f''| / 8); the derivation is
  in [`sofa_env/certified.py`](sofa_env/certified.py).

Training logs this certified area for the deterministic policy after every update and keeps
the checkpoint with the highest one.

## Training recipe

| stage | command | what it does | output |
|---|---|---|---|
| 1. base | `python train.py` | PPO at N = 200 on the sampled area: log_std −0.5, entropy 1e-3, actor lr 3e-4 (linear decay), 1M steps | `runs/base/` |
| 2. fine-tune | `python finetune.py` | starts from `runs/base/best.pt` and trains on the certified area with 4 substeps: actor lr 1e-4, entropy 0, 0.5M steps | `runs/finetune/` |

On one CPU core stage 1 took about 70 minutes and stage 2 about 20 (fewer steps, but each
step does 4 certified cuts). `--run-name` writes to another folder; an existing run folder
is never overwritten.
All settings live in `BASE_CONFIG` (`train.py`) and `FINETUNE_CONFIG` (`finetune.py`).

Each run folder holds `config.json`, `metrics.jsonl` (one row per update), `best.pt` /
`last.pt`, and `paths/update_XXXX.npz` with the deterministic policy's corner path after
every update. Since the environment is deterministic, a saved path reproduces its sofa
exactly:

```python
import numpy as np
from sofa_env import replay

corners = np.load("runs/finetune/paths/update_0000.npz")["corners"]
sofa = replay(corners, substeps=4, reward_area="certified").shape   # a shapely Polygon
```

## Results

Certified true areas (hallway checked at ~6400 angles along the continuous motion):

| sofa | area |
|---|---|
| Hammersley | 2.20742 |
| stage 1 (base) | ≥ 2.218846 (upper bound 2.218950) |
| **stage 2 (fine-tune)** | **≥ 2.218873** (upper bound 2.218978) |
| Gerver | 2.21953 |

The learned sofa has Gerver's features, not Hammersley's: a taller notch, smoothed notch
shoulders and rounded outer corners. Its corner path is on average 0.012 (at most 0.02)
from Gerver's.

What did and did not matter, from the sweeps that led here:

- **Exploration noise was the main lever.** Raising the initial log_std from −3 to −0.5
  took the learned area from ~1.88 to ~2.216 and made seeds agree.
- **The training resolution N does not change the true area reached** (N = 100 to 400 land
  within 0.0002 of each other once measured with certified bounds); it only changes how much
  the sampled area overstates it.
- **Fine-tuning helps only briefly.** Continuing to train a converged policy, at larger N or
  on the certified reward, drifts below its starting area after the first few updates: a
  PPO step sized to the policy's noise moves the whole path by more than the remaining gains.
  The certified-reward fine-tune drifted least and holds the best certified area.

## Visualising a run

```bash
python render.py runs/base            # needs ffmpeg
```

writes `learning.mp4` (the sofa after every update, against Gerver's), `motion.mp4` (the
final sofa moving through the corridor beside Gerver's), and the `carving` / `discovery` /
`every` animations of the block being shaved down along successive attempts. `--only`
renders one of them.

## Layout

```
sofa_env/
  env.py          SofaEnv: the rotation-path environment (substeps, certified reward)
  certified.py    certified cut: points that stay in the hallway for a whole step
  corridor.py     the L-shaped hallway polygon
  paths.py        replay(): rebuild a sofa from a saved corner path
  reference.py    Hammersley's and Gerver's corner paths (Gerver's from Romik 2016)
sofa_rl/
  networks/       Gaussian actor and value critic (MLPs)
  policies/       action sampling and log-probabilities
  agents/         PPO: rollout collection, GAE, clipped-surrogate update
  config.py       TrainConfig and its parts (env, network, PPO, optimiser)
train.py          stage 1 (BASE_CONFIG) and the training loop
finetune.py       stage 2 (FINETUNE_CONFIG)
render.py         animations from a run's saved paths
```

## Setup

```bash
conda env create -f environment.yml
conda activate sofa-problem
```

## References

- J. M. Hammersley, *On the enfeeblement of mathematical skills by "Modern Mathematics"*, 1968.
- J. L. Gerver, *On moving a sofa around a corner*, Geometriae Dedicata 42, 1992.
- D. Romik, *Differential equations and exact solutions in the moving sofa problem*,
  Experimental Mathematics, 2018 ([arXiv:1606.08111](https://arxiv.org/abs/1606.08111)).
- J. Baek, *Optimality of Gerver's sofa*, 2024 ([arXiv:2411.19826](https://arxiv.org/abs/2411.19826)).
