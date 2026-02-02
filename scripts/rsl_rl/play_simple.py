# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Simplified script to play a pre-trained policy.

Handles both JIT (TorchScript) and regular checkpoint formats.
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a pre-trained RL policy.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--task", type=str, default="Gen3-Reach-v0", help="Name of the task."
)
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to the policy checkpoint."
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import gen3.tasks  # noqa: F401


def main():
    """Play with a pre-trained policy."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Load the policy checkpoint
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO]: Loading policy from: {checkpoint_path}")

    # Try loading as JIT model first, fall back to state dict
    policy = None
    try:
        # Load as TorchScript (JIT) model
        policy = torch.jit.load(checkpoint_path, map_location=env.unwrapped.device)
        print("[INFO]: Loaded as TorchScript (JIT) model.")
    except Exception as e:
        print(f"[INFO]: Not a JIT model ({e}), trying as state dict...")

        # Load as regular checkpoint
        loaded_dict = torch.load(
            checkpoint_path, map_location=env.unwrapped.device, weights_only=False
        )

        # Create a simple MLP policy matching the saved architecture
        from torch import nn

        obs_dim = env.unwrapped.observation_space.shape[0]
        act_dim = env.unwrapped.action_space.shape[0]

        print(f"[INFO]: Observation dim: {obs_dim}, Action dim: {act_dim}")

        # Build actor network (same architecture as RSL-RL PPO: [64, 64] with ELU)
        actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, act_dim),
        ).to(env.unwrapped.device)

        # Load weights
        if "model_state_dict" in loaded_dict:
            model_state = loaded_dict["model_state_dict"]
            actor_state = {
                k.replace("actor.", ""): v
                for k, v in model_state.items()
                if k.startswith("actor.")
            }
            actor.load_state_dict(actor_state)
            print("[INFO]: Loaded actor weights from checkpoint.")

        policy = actor

    dt = env.unwrapped.physics_dt

    # Reset environment
    obs_dict, _ = env.reset()

    # Extract observation tensor
    if isinstance(obs_dict, dict):
        obs = obs_dict.get("policy", obs_dict.get("obs", list(obs_dict.values())[0]))
    else:
        obs = obs_dict

    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, device=env.unwrapped.device, dtype=torch.float32)

    print(f"[INFO]: Starting simulation. Press Ctrl+C to stop.")
    print(f"[INFO]: Observation shape: {obs.shape}")

    # Simulate environment
    step_count = 0
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # Get action from policy
            actions = policy(obs)

            # Step environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)

            # Extract observation tensor
            if isinstance(obs_dict, dict):
                obs = obs_dict.get(
                    "policy", obs_dict.get("obs", list(obs_dict.values())[0])
                )
            else:
                obs = obs_dict

            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(
                    obs, device=env.unwrapped.device, dtype=torch.float32
                )

        step_count += 1
        if step_count % 100 == 0:
            print(f"[INFO]: Step {step_count}, Reward: {rewards.mean().item():.4f}")

        # Time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
