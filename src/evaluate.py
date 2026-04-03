"""
Evaluation script to visualize a trained Double Pendulum agent.

This script loads a pre-trained PPO model and runs it in the environment
with rendering enabled to visualize the agent's performance.

Usage:
    python evaluate.py --model_path models/ppo_shaped.zip --episodes 3
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from environment import DoublePendulumEnv


def evaluate_agent(model_path, episodes=3, save_gif=False, gif_path="agent_demo.gif"):
    """
    Evaluate a trained agent in the environment.
    
    Args:
        model_path (str): Path to the saved model
        episodes (int): Number of episodes to run
        save_gif (bool): Whether to save a GIF
        gif_path (str): Path to save the GIF
    """
    
    # Load the model
    print(f"Loading model from {model_path}...")
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Create environment
    env = DoublePendulumEnv(reward_type='shaped')
    
    # Load the model
    model = PPO.load(model_path, env=env)
    
    print(f"Model loaded successfully!")
    print(f"\nRunning {episodes} episodes with rendering...")
    
    frames = []
    total_reward = 0
    episode_rewards = []
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        print(f"\n--- Episode {episode + 1} / {episodes} ---")
        
        while not done:
            # Get action from model (deterministic)
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Render
            env.render()
            
            # Optionally save frame for GIF
            if save_gif:
                try:
                    import pygame
                    frame = pygame.surfarray.array3d(env.screen)
                    frame = np.transpose(frame, (1, 0, 2))  # pygame uses (width, height, channels)
                    frames.append(frame)
                except Exception as e:
                    print(f"Warning: Could not capture frame for GIF: {e}")
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1} - Total Reward: {episode_reward:.3f}, Steps: {env.steps}")
        total_reward += episode_reward
    
    # Summary
    avg_reward = np.mean(episode_rewards)
    print(f"\n{'='*60}")
    print(f"Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total Episodes: {episodes}")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Best Episode Reward: {max(episode_rewards):.3f}")
    print(f"Worst Episode Reward: {min(episode_rewards):.3f}")
    print(f"{'='*60}\n")
    
    # Save GIF if requested
    if save_gif and frames:
        try:
            import imageio
            print(f"Saving GIF to {gif_path}...")
            imageio.mimsave(gif_path, frames, fps=30)
            print(f"GIF saved successfully!")
        except Exception as e:
            print(f"Warning: Could not save GIF: {e}")
    
    # Close environment
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/ppo_shaped.zip",
        help="Path to the trained model"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--save_gif",
        action="store_true",
        help="Save evaluation as GIF"
    )
    parser.add_argument(
        "--gif_path",
        type=str,
        default="media/agent_demo.gif",
        help="Path to save the GIF"
    )
    
    args = parser.parse_args()
    
    # Create directories if needed
    Path(args.gif_path).parent.mkdir(parents=True, exist_ok=True)
    
    evaluate_agent(
        args.model_path,
        episodes=args.episodes,
        save_gif=args.save_gif,
        gif_path=args.gif_path
    )


if __name__ == "__main__":
    main()
