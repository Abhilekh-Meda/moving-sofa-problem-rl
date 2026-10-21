"""Reward functions for the SofaEnv.

SofaEnv has no default reward -- `reward_fn=` is required at construction
(see env.py). Any reward_fn must accept these keyword arguments and return
a float:
    prev_dist, new_dist, prev_pose, new_pose, collided, success

prev_dist / new_dist are the geodesic distance from the shape's centre (x, y)
to the goal before and after the step (raw, not normalized). Both are always
finite.
"""
