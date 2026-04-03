"""
Training script for Double Pendulum RL agent using PPO.

This script trains a Proximal Policy Optimization (PPO) agent to balance
both poles in the double inverted pendulum environment.

Usage:
    python train.py --timesteps 200000 --reward_type shaped --save_path models/ppo_shaped.zip
"""

import argparse
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from environment import DoublePendulumEnv


class MetricsCallback(BaseCallback):
    """Custom callback to log metrics to CSV."""
    
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.rewards = []
        self.timesteps_list = []
        
    def _on_step(self) -> bool:
        """Log metrics every 1000 steps."""
        if self.num_timesteps % 1000 == 0:
            if len(self.model.ep_info_buffer) > 0:
                # Get mean reward from episode buffer
                ep_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
                mean_reward = np.mean(ep_rewards)
            else:
                mean_reward = 0.0
            
            self.timesteps_list.append(self.num_timesteps)
            self.rewards.append(mean_reward)
            
            if self.num_timesteps % 10000 == 0:
                print(f"Timestep: {self.num_timesteps}, Mean Reward: {mean_reward:.3f}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Save metrics to CSV on training completion."""
        import csv
        
        csv_path = os.path.join(self.log_dir, "training_metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "mean_reward"])
            for ts, reward in zip(self.timesteps_list, self.rewards):
                writer.writerow([ts, reward])
        
        print(f"Metrics saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on Double Pendulum")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200000,
        help="Total timesteps for training"
    )
    parser.add_argument(
        "--reward_type",
        choices=["baseline", "shaped"],
        default="shaped",
        help="Type of reward function"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="models/ppo_model.zip",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate for PPO"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path("logs") / args.reward_type
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training Double Pendulum RL Agent")
    print(f"{'='*60}")
    print(f"Reward Type: {args.reward_type}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Save Path: {args.save_path}")
    print(f"Log Dir: {log_dir}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = DoublePendulumEnv(reward_type=args.reward_type)
    
    # Configure logging
    logger = configure(str(log_dir), ["stdout", "csv"])
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    model.set_logger(logger)
    
    # Create callback
    callback = MetricsCallback(log_dir=str(log_dir))
    
    # Train the agent
    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save the model
    model.save(args.save_path)
    print(f"\nModel saved to {args.save_path}")
    
    # Close environment
    env.close()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
