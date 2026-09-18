"""Reward functions for the SofaEnv.

SofaEnv has no default reward -- `reward_fn=` is required at construction
(see env.py). Any reward_fn must accept these keyword arguments and return
a float:
    prev_pose, new_pose, collided, success
"""
