"""
train_rl.py - Reinforcement Learning Policy Training Script
============================================================
Trains the PPO continuous vehicle controller using curriculum learning
across procedural cities, traffic conditions, signals, and pedestrians.

Usage:
  python train_rl.py --episodes 1000 --save-path rl_policy.pt --headless
  python train_rl.py --episodes 2000 --resume rl_policy.pt --save-path rl_policy_v2.pt
"""

import argparse
import os
import time
from typing import Dict, Any
import numpy as np

from rl_agent import PPOAgent
from rl_env import CityDrivingRLEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train Auto Drive RL Agent (PPO)")
    parser.add_argument("--episodes", type=int, default=500, help="Total training episodes")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--save-path", type=str, default="rl_policy.pt", help="Path to save trained policy")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing policy checkpoint to resume from")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless without display")
    parser.add_argument("--traffic", type=int, default=40, help="Traffic density")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Episodes between checkpoints")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO mini-batch size")
    parser.add_argument("--epochs", type=int, default=6, help="PPO epochs per update")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    return parser.parse_args()


def train():
    args = parse_args()

    print("=======================================================", flush=True)
    print("      CITY DRIVING SIMULATOR - RL AGENT TRAINING       ", flush=True)
    print("=======================================================", flush=True)
    print(f"  Episodes:       {args.episodes}", flush=True)
    print(f"  Save Path:      {args.save_path}", flush=True)
    print(f"  Base Seed:      {args.seed}", flush=True)
    print(f"  Traffic Count:  {args.traffic}", flush=True)
    print("=======================================================\n", flush=True)

    # Initialize environment
    env = CityDrivingRLEnv(
        seed=args.seed,
        traffic=args.traffic,
        frame_skip=2,
        max_steps=400,
        curriculum_level=1
    )

    # Initialize PPO Agent
    obs_dim = 28
    act_dim = 3
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, lr=args.lr)

    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
        print(f"[TRAIN] Resumed policy weights from {args.resume}", flush=True)

    # Metrics tracking
    episode_rewards = []
    success_count = 0
    collision_count = 0
    red_light_count = 0
    start_time = time.time()

    for episode in range(1, args.episodes + 1):
        progress_pct = episode / float(args.episodes)
        if progress_pct < 0.25:
            env.curriculum_level = 1
        elif progress_pct < 0.60:
            env.curriculum_level = 2
        else:
            env.curriculum_level = 3

        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        step = 0

        while not done:
            step += 1
            action, log_prob, val = agent.select_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, log_prob, val, done)
            ep_reward += reward
            obs = next_obs

            if info.get("collision", False):
                collision_count += 1
            if info.get("red_light", False):
                red_light_count += 1
            if info.get("destination", False):
                success_count += 1

        agent.finish_path(last_val=0.0)
        losses = agent.update(batch_size=args.batch_size, epochs=args.epochs)

        episode_rewards.append(ep_reward)
        avg_reward = np.mean(episode_rewards[-25:])
        success_rate = (success_count / episode) * 100.0

        if episode % 5 == 0 or episode <= 3 or episode == args.episodes:
            elapsed = time.time() - start_time
            print(f"Ep {episode:4d}/{args.episodes:4d} | Level {env.curriculum_level} | "
                  f"Ep Reward: {ep_reward:6.1f} | Avg(25): {avg_reward:6.1f} | "
                  f"Success: {success_rate:4.1f}% | Collisions: {collision_count} | "
                  f"RedLights: {red_light_count} | Time: {elapsed:.1f}s", flush=True)

        if episode % args.checkpoint_interval == 0:
            agent.save(args.save_path)

    agent.save(args.save_path)
    total_time = time.time() - start_time
    print("\n=======================================================", flush=True)
    print("                TRAINING COMPLETED                     ", flush=True)
    print(f"  Total Time:       {total_time:.1f}s", flush=True)
    print(f"  Final Model:      {args.save_path}", flush=True)
    print(f"  Overall Success:  {(success_count / args.episodes) * 100:.1f}%", flush=True)
    print("=======================================================\n", flush=True)


if __name__ == "__main__":
    train()
