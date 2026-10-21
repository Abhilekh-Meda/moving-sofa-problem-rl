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


|                     | shape  | dtype     | bounds                                                                  |
| ------------------- | ------ | --------- | ----------------------------------------------------------------------- |
| `action_space`      | `(3,)` | `float32` | `[-max_dxy, max_dxy] x [-max_dxy, max_dxy] x [-max_dtheta, max_dtheta]` |
| `observation_space` | `(6,)` | `float32` | see below                                                               |


**Action** — `[dx, dy, dtheta]`: a per-step rigid-motion delta added to the
current pose. `max_dxy`/`max_dtheta` (constructor args, default `0.05`
each) set the action bounds; `env.step()` clips to them. No constraint on
direction or on `dtheta` accumulating monotonically — the shape can rotate
however much, however often, in whatever direction.

**Observation** — `[x_norm, y_norm, sin(theta), cos(theta), time_left_norm, geo_dist_norm]`:


| index | name             | range      | meaning                                                                                                           |
| ----- | ---------------- | ---------- | ----------------------------------------------------------------------------------------------------------------- |
| 0     | `x_norm`         | `[-2, 2]`  | `x / arm_length`                                                                                                  |
| 1     | `y_norm`         | `[-2, 2]`  | `y / arm_length`                                                                                                  |
| 2     | `sin(theta)`     | `[-1, 1]`  | periodic-safe heading                                                                                             |
| 3     | `cos(theta)`     | `[-1, 1]`  | periodic-safe heading                                                                                             |
| 4     | `time_left_norm` | `[0, 1]`   | `1 - step_count / max_steps`                                                                                      |
| 5     | `geo_dist_norm`  | `[0, inf)` | shape's distance to goal, divided by its value at the default start (see Distance to goal below)                  |


`theta` itself is unbounded and not clipped internally (it's periodic) —
`sin`/`cos` are what the agent actually sees.

`geo_dist_norm` uses a fixed normalizer: the shape's distance to the goal
at the default start pose, computed once at construction. It does not depend on
where a given episode starts, so a pose maps to the same value in every
episode. It is 1.0 at the default start, 0 inside the goal region, and
greater than 1 for poses farther from the goal than the default start.

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
- `start_theta` (default `0.0`, radians)

The start pose is not validated: `reset()` accepts any pose and reports
whether it is outside the corridor. `info` on reset: `{"colliding": bool}`
(only possible with a non-default start). A caller choosing its own start
poses should check this flag and pick another pose if it is set.

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
`geodesic_dist` is the shape's raw (unnormalized) distance to goal at the
current pose.



## `reward_fn` (required — no default)

```python
RewardFn = Callable[..., float]  # keyword args: prev_dist, new_dist, prev_pose, new_pose, collided, success
```

Must be supplied at construction; `SofaEnv(...)` raises `ValueError`
without one. Called every step with:


| kwarg       | type              | meaning                                                                    |
| ----------- | ----------------- | -------------------------------------------------------------------------- |
| `prev_dist` | `float`           | shape's raw distance to goal at `prev_pose` (see Distance to goal below)   |
| `new_dist`  | `float`           | shape's raw distance to goal at `new_pose`                                 |
| `prev_pose` | `np.ndarray (3,)` | pose before this step                                                      |
| `new_pose`  | `np.ndarray (3,)` | pose after this step (post-clip, post-collision-handling)                  |
| `collided`  | `bool`            | whether the new pose left the corridor                                     |
| `success`   | `bool`            | whether the new pose is fully inside the goal region                       |

See `sofa_env/rewards.py` (currently empty — no reward functions defined
yet).

## Constructor (`SofaEnv(...)`)


| arg                   | default              | meaning                                                                                                                                |
| --------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `shape`               | required             | `Polygon` or `(N, 2)` array of points (traversal order; see `shape.polygon_from_points`)                                               |
| `corridor_width`      | `1.0`                | width of both arms of the L                                                                                                            |
| `arm_length`          | `2 * corridor_width` | length of each arm                                                                                                                     |
| `goal_depth`          | `arm_length / 2`     | goal region is the strip `arm_length - goal_depth` deep at the far end of the vertical arm; must satisfy `0 < goal_depth < arm_length` |
| `max_steps`           | `1000`               | episode truncation horizon                                                                                                             |
| `max_dxy`             | `0.05`               | per-step translation bound (both x and y)                                                                                              |
| `max_dtheta`          | `0.05`               | per-step rotation bound (radians)                                                                                                      |
| `collision_mode`      | `"terminate"`        | `"terminate"` or `"reject"` (see `step()` above)                                                                                       |
| `boundary_spacing`    | `0.025`              | spacing of the boundary points used for the distance to goal, as a fraction of `corridor_width`                                       |
| `reward_fn`           | required             | see above                                                                                                                              |
| `render_mode`         | `None`               | `"human"` or `"rgb_array"`                                                                                                             |




## Corridor geometry (`corridor.build_l_corridor`)

L-shaped union of two rectangles, corner at the origin:

```
horizontal arm: x in [-arm_length, width], y in [0, width]
vertical arm:   x in [0, width],            y in [-arm_length, width]
```

Episodes start deep in the horizontal arm by default (`x = -arm_length/2`,
`y = width/2`, `theta = 0`; see `reset()` options to start elsewhere)
and the goal is deep in the vertical arm (`y <= -goal_depth`) — the shape
must turn the corner.

Containment (`_is_contained`) only checks sampled poses, not the
continuous sweep between consecutive steps — a fast-rotating/translating
shape could in principle clip through a wall between two sampled poses
without being detected. Noted as a known limitation in `env.py`.

## Distance to goal (`geodesic.py`, `SofaEnv._geodesic_shape_dist`)

**Point distance.** Straight-line distance to the goal cuts through the
corner wall for points in different arms. The geodesic distance (shortest
path that stays in the corridor) bends only at reflex vertices, and the
L-corridor has one, the inner corner at (0, 0). So
`geodesic.corridor_geodesic(pts, goal_depth)` computes it exactly in closed
form for points of shape `(P, 2)`:

```
x >= 0 (vertical arm, corner square):  max(0, y + goal_depth)   # straight down
x <  0 (horizontal arm):               hypot(x, y) + goal_depth  # to the corner, then down
```

The two branches agree along the opening `x = 0, y >= 0`, so the distance
is continuous everywhere in the corridor. It is specific to this L-corridor
geometry.

**Shape distance.** The shape's distance to the goal is the mean of
`corridor_geodesic` over points spaced `boundary_spacing * corridor_width`
apart along the shape's boundary (computed once from `base_shape`, then
posed each step). Using the whole boundary rather than the centroid means:

- it reflects where the shape actually is, including non-convex shapes
whose centroid can lie outside the shape (and so inside a wall) at a
valid pose;
- it depends on `theta`, not only on `(x, y)`;
- it is 0 exactly when every boundary point is in the goal region, i.e.
the whole shape is in the goal.

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