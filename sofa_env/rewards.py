"""Reward functions for the SofaEnv.

SofaEnv has no default reward -- `reward_fn=` is required at construction
(see env.py). Any reward_fn must accept these keyword arguments and return
a float:
    prev_dist, new_dist, prev_pose, new_pose, collided, success

prev_dist / new_dist are the shape's distance to the goal before and after
the step: the mean geodesic distance over points along the shape's boundary
(raw, not normalized; 0 when the whole shape is in the goal region). Both are
always finite.
"""
