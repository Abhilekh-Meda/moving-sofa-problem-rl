# sofa_env

Gymnasium-style environment for the moving-sofa problem: transport a fixed
polygon through an L-shaped corridor via per-step rigid-motion actions,
without ever leaving the corridor.

Registered as `Sofa-v0` (see `sofa_env/env.py`), and importable directly:

```python
from sofa_env import SofaEnv

env = SofaEnv(shape=my_polygon, reward_fn=my_reward_fn)
obs, info = env.reset()
```

## Spaces

| | shape | dtype | bounds |
|---|---|---|---|
| `action_space` | `(3,)` | `float32` | `[-max_dxy, max_dxy] x [-max_dxy, max_dxy] x [-max_dtheta, max_dtheta]` |
| `observation_space` | `(6,)` | `float32` | see below |

**Action** — `[dx, dy, dtheta]`: a per-step rigid-motion delta added to the
current pose. `max_dxy`/`max_dtheta` (constructor args, default `0.05`
each) set the action bounds; `env.step()` clips to them. No constraint on
direction or on `dtheta` accumulating monotonically — the shape can rotate
however much, however often, in whatever direction.

**Observation** — `[x_norm, y_norm, sin(theta), cos(theta), time_left_norm, geo_dist_norm]`:

| index | name | range | meaning |
|---|---|---|---|
| 0 | `x_norm` | `[-2, 2]` | `x / arm_length` |
| 1 | `y_norm` | `[-2, 2]` | `y / arm_length` |
| 2 | `sin(theta)` | `[-1, 1]` | periodic-safe heading |
| 3 | `cos(theta)` | `[-1, 1]` | periodic-safe heading |
| 4 | `time_left_norm` | `[0, 1]` | `1 - step_count / max_steps` |
| 5 | `geo_dist_norm` | `[0, inf)` | corner-aware geodesic distance to goal, normalized by the distance at episode start (see Geodesic distance below) |

`theta` itself is unbounded and not clipped internally (it's periodic) —
`sin`/`cos` are what the agent actually sees.

## Pose

Internal state is `env.pose = [x, y, theta]` (`float64`, not part of the
public observation directly). The shape is recentred on construction so
its local origin is its own centroid; `_posed_polygon(pose)` rotates about
the origin in local coordinates, then translates by `(x, y)`.

## `reset(*, seed=None, options=None)`

Returns `(obs, info)`.

`options` (dict, optional):
- `start_x` (default `-arm_length / 2`)
- `start_y` (default `corridor_width / 2`)

`info` on reset: `{"colliding": bool}` — whether the starting pose is
already outside the corridor (only possible with a non-default start).

## `step(action)`

Returns `(obs, reward, terminated, truncated, info)`.

- `action` is clipped to `action_space` bounds before being applied.
- `collision_mode="terminate"` (default): a colliding new pose ends the
  episode (`terminated=True`), and the colliding pose is kept as `env.pose`.
- `collision_mode="reject"`: a colliding step is rolled back (pose stays at
  `prev_pose`); the episode continues.
- `success = (not collided) and goal_region.contains(new_polygon)` — fully
  inside the goal region, no partial-overlap credit.
- `terminated = success or (collided and collision_mode == "terminate")`.
- `truncated = step_count >= max_steps`.
- `info`: `{"colliding": bool, "success": bool, "geodesic_dist": float}`.

## `reward_fn` (required — no default)

```python
RewardFn = Callable[..., float]  # keyword args: prev_pose, new_pose, collided, success
```

Must be supplied at construction; `SofaEnv(...)` raises `ValueError`
without one. Called every step with:

| kwarg | type | meaning |
|---|---|---|
| `prev_pose` | `np.ndarray (3,)` | pose before this step |
| `new_pose` | `np.ndarray (3,)` | pose after this step (post-clip, post-collision-handling) |
| `collided` | `bool` | whether the new pose left the corridor |
| `success` | `bool` | whether the new pose is fully inside the goal region |

See `sofa_env/rewards.py` (currently empty — no reward functions defined
yet).

## Constructor (`SofaEnv(...)`)

| arg | default | meaning |
|---|---|---|
| `shape` | required | `Polygon` or `(N, 2)` array of points (traversal order; see `shape.polygon_from_points`) |
| `corridor_width` | `1.0` | width of both arms of the L |
| `arm_length` | `2 * corridor_width` | length of each arm |
| `goal_depth` | `arm_length / 2` | goal region is the strip `arm_length - goal_depth` deep at the far end of the vertical arm; must satisfy `0 < goal_depth < arm_length` |
| `max_steps` | `1000` | episode truncation horizon |
| `max_dxy` | `0.05` | per-step translation bound (both x and y) |
| `max_dtheta` | `0.05` | per-step rotation bound (radians) |
| `collision_mode` | `"terminate"` | `"terminate"` or `"reject"` (see `step()` above) |
| `geodesic_resolution` | `40` | grid cells per unit of `corridor_width` for the geodesic distance field |
| `reward_fn` | required | see above |
| `render_mode` | `None` | `"human"` or `"rgb_array"` |

## Corridor geometry (`corridor.build_l_corridor`)

L-shaped union of two rectangles, corner at the origin:

```
horizontal arm: x in [-arm_length, width], y in [0, width]
vertical arm:   x in [0, width],            y in [-arm_length, width]
```

Episodes start deep in the horizontal arm (`x = -arm_length/2` by default)
and the goal is deep in the vertical arm (`y <= -goal_depth`) — the shape
must turn the corner.

Containment (`_is_contained`) only checks sampled poses, not the
continuous sweep between consecutive steps — a fast-rotating/translating
shape could in principle clip through a wall between two sampled poses
without being detected. Noted as a known limitation in `env.py`.

## Geodesic distance (`geodesic.py`)

Straight-line distance to the goal cuts through the corner wall for points
in different arms, so `geo_dist_norm` instead uses a grid-based multi-source
Dijkstra flood from the goal cells, restricted to in-corridor cells — this
respects the L-shape. Built once per env (`geodesic_resolution` controls
grid fineness), queried in O(1) per step via nearest-cell lookup
(`geodesic.query`).

## Rendering (`rendering.py`)

- `env.render(save_path=None)` — draws the current pose in the corridor.
  Returns an `(H, W, 3)` uint8 array if `render_mode="rgb_array"`;
  otherwise draws to a live matplotlib window (`"human"` mode) and returns
  `None`. Sofa is drawn blue normally, red if colliding.
- `record_rollout(env, policy=None, max_steps=None, seed=None, reset_options=None)`
  — runs one episode (random actions if `policy=None`), returns a list of
  `rgb_array` frames. Temporarily overrides `env.render_mode` and restores
  it afterward.
- `save_gif(frames, path, fps=20)` — writes frames to a `.gif` via
  `imageio`.

## Shape input (`shape.polygon_from_points`)

Points must already be in boundary-traversal order (no ordering is
inferred) and form a valid, non-self-intersecting polygon — raises
`ValueError` otherwise. `SofaEnv` also accepts a `shapely.Polygon` directly.
