#!/usr/bin/env python
"""Train the iCrowdNav PPO policy."""

import argparse
import os

import gymnasium as gym
import rospy
import socbev_gym  # noqa: F401  registers social-bev-v0
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from bev_perception.bev_perception import BevExtractor, SocialBevExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Train iCrowdNav PPO")
    parser.add_argument(
        "--log-dir",
        default=os.path.expanduser("~/drl_logdir/policy_training"),
        help="TensorBoard / checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional PPO zip to resume. If empty, train from scratch.",
    )
    parser.add_argument("--timesteps", type=int, default=10000000)
    parser.add_argument("--save-freq", type=int, default=10000)
    parser.add_argument("--social-mode", action="store_true", default=True)
    parser.add_argument("--no-social-mode", dest="social_mode", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    rospy.init_node("training_node", log_level=rospy.INFO)
    os.makedirs(args.log_dir, exist_ok=True)

    env = gym.make("social-bev-v0", social_mode=args.social_mode)
    env = Monitor(env, args.log_dir)
    env.reset()

    extractor = SocialBevExtractor if args.social_mode else BevExtractor
    policy_kwargs = dict(
        features_extractor_class=extractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256], vf=[128]),
    )
    ppo_kwargs = dict(
        tensorboard_log=args.log_dir,
        verbose=2,
        n_epochs=6,
        n_steps=1024,
        batch_size=128,
        learning_rate=5e-4,
    )

    checkpoint = os.path.expanduser(args.checkpoint) if args.checkpoint else ""
    if checkpoint:
        rospy.loginfo("Resuming from %s", checkpoint)
        model = PPO.load(checkpoint, env=env, **ppo_kwargs)
    else:
        rospy.loginfo("Training PPO from scratch")
        model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, **ppo_kwargs)

    callback = CheckpointCallback(save_freq=args.save_freq, save_path=args.log_dir)
    model.learn(
        total_timesteps=args.timesteps,
        log_interval=5,
        tb_log_name="drl_policy",
        callback=callback,
        reset_num_timesteps=True,
    )
    model.save(os.path.join(args.log_dir, "policy_model"))
    rospy.loginfo("Training finished.")
    env.close()


if __name__ == "__main__":
    main()
