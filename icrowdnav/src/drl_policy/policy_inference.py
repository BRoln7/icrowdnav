#!/usr/bin/env python
"""Evaluate a trained iCrowdNav PPO policy."""

import argparse
import os

import gymnasium as gym
import rospy
import socbev_gym  # noqa: F401
import torch
from stable_baselines3 import PPO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate iCrowdNav PPO")
    parser.add_argument(
        "--checkpoint",
        default=os.path.expanduser(
            "~/drl_logdir/policy_training/social-bev/rl_model_340000_steps.zip"
        ),
        help="PPO zip checkpoint",
    )
    parser.add_argument("--social-mode", action="store_true", default=True)
    parser.add_argument("--no-social-mode", dest="social_mode", action="store_false")
    parser.add_argument(
        "--quality-analysis",
        action="store_true",
        default=True,
        help="Skip random robot reset (fixed evaluation protocol).",
    )
    parser.add_argument(
        "--no-quality-analysis", dest="quality_analysis", action="store_false"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rospy.init_node("inference_node", log_level=rospy.INFO)
    checkpoint = os.path.expanduser(args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError("Checkpoint not found: %s" % checkpoint)

    env = gym.make(
        "social-bev-v0",
        train_mode=False,
        social_mode=args.social_mode,
        quality_analysis=args.quality_analysis,
    )
    model = PPO.load(
        checkpoint,
        env=env,
        tensorboard_log=os.path.dirname(checkpoint),
        verbose=2,
    )

    obs, _ = env.reset()
    while not rospy.is_shutdown():
        with torch.inference_mode():
            action, _states = model.predict(obs, deterministic=False)
            obs, _reward, done, _truncated, _info = env.step(action)
            if done:
                obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    main()
