# Double Pendulum Control with Reinforcement Learning

A comprehensive project implementing a Double Inverted Pendulum environment and training a PPO (Proximal Policy Optimization) agent to balance two poles simultaneously using reinforcement learning.

## Overview

This project demonstrates the complete RL workflow: environment design, reward function engineering, agent training, and policy evaluation. The challenge is to balance two interconnected poles on a moving cart—a notoriously difficult control problem that requires sophisticated learning algorithms.

### Key Features

- **Custom Physics-Based Environment**: Built from scratch using the pymunk 2D physics engine
- **Dual Reward Functions**: Compare baseline (sparse) vs shaped (dense) reward strategies
- **PPO Agent**: State-of-the-art policy gradient algorithm from Stable Baselines3
- **Containerized Setup**: Fully reproducible environment using Docker
- **Comprehensive Logging**: Training metrics and performance visualization

### Environment Design

### Physical System

The Double Pendulum environment consists of:

1. **Cart**: A movable platform on a frictionless horizontal track
   - Mass: 1.0 kg
   - Width: 0.4 m
   - Height: 0.3 m

2. **Pole 1**: First pole attached to the cart via a pivot joint
   ```
   - Mass: 1.0 kg
   - Length: 0.5 m
   - Angle: theta₁

3. **Pole 2**: Second pole attached to pole 1 via a pivot joint
   - Mass: 1.0 kg
   - Length: 0.5 m
   - Angle: theta₂

### Physics Simulation

- **Engine**: pymunk (Chipmunk2D wrapper)
- **Timestep**: 1/60 second (60 FPS)
- **Gravity**: 9.81 m/s²
- **Damping**: 0.1 (small friction for stability)
- **Constraints**: 
  - GrooveJoint: Restricts cart to horizontal track
  - PivotJoint: Connects cart to pole1 and pole1 to pole2

### Observation Space

The agent observes 6 continuous values:

```
observation = [
    cart_position,          # Position of cart on track (-2.4 to 2.4 m)
    cart_velocity,          # Velocity of cart (-∞ to +∞ m/s)
    pole1_angle,            # Angle from vertical (-π to +π radians)
    pole1_angular_velocity, # Angular velocity (-∞ to +∞ rad/s)
    pole2_angle,            # Angle from vertical (-π to +π radians)
    pole2_angular_velocity  # Angular velocity (-∞ to +∞ rad/s)
]
```

**Space Type**: `gym.spaces.Box(shape=(6,), low=-∞, high=+∞, dtype=float32)`

### Action Space

The agent outputs a continuous force value:

```
action = [force]  # Force applied to cart in [-1.0, 1.0] (normalized)
```

**Space Type**: `gym.spaces.Box(shape=(1,), low=-1.0, high=1.0, dtype=float32)`

**Scaling**: Actions are scaled to a maximum force of 100 N during step execution.

### Episode Termination

An episode ends when:

1. Either pole falls over (|angle| > 36°)
2. Cart moves off the track (|cart_position| > 2.4 m)
3. Maximum steps reached (500 steps = ~8.3 seconds)

### Reward Function Design

Two reward function strategies are implemented to demonstrate the impact of reward shaping:

### Baseline Reward (Sparse)

The baseline reward focuses solely on the primary goal: keeping both poles upright.

```
R_baseline = cos(theta₁) + cos(theta₂)
```

**Characteristics:**
- Maximum reward when both poles are perfectly vertical (theta = 0)
- Reward = 2.0 when both poles are upright
- Reward becomes increasingly negative as poles tilt
- Provides minimal guidance for intermediate states

**Formula Explanation:**
- `cos(0) = 1.0` (maximum reward at upright position)
- `cos(π/4) ≈ 0.707` (partial credit for partial upright)
- `cos(π/2) = 0` (no reward when pole is horizontal)
- `cos(π) = -1.0` (penalty when pole is inverted)

### Shaped Reward (Dense)

The shaped reward augments the baseline with additional penalty terms to provide richer feedback and accelerate learning.

```
R_shaped = R_upright + R_center + R_velocity + R_action + R_cart_velocity
```

Where:

1. **Upright Bonus** (Primary Goal)
   ```
   R_upright = cos(theta₁) + cos(theta₂)
   ```
   - Maximum: 2.0
   - Rationale: Core objective is to keep poles balanced

2. **Center Penalty** (Stability Term)
   ```
   R_center = -|cart_x| × 0.1
   ```
   - Penalty for moving away from track center
   - Coefficient: 0.1
   - Encourages agents to keep cart centered
   - Maximum penalty: -0.7 when cart is at edge (±7m)
   - Rationale: Helps stabilize the system and prevents running off-track

3. **Angular Velocity Penalty** (Smoothness Term)
   ```
   R_velocity = -(|ω₁| + |ω₂|) × 0.01
   ```
   - Penalty for excessive angular velocities
   - Coefficient: 0.01
   - Encourages smooth, controlled motion
   - Rationale: Prevents jerky movements and promotes stability

4. **Action Penalty** (Energy Efficiency Term)
   ```
   R_action = -(action²) × 0.001
   ```
   - Penalty for large force commands
   - Coefficient: 0.001
   - Encourages efficient, minimal-force solutions
   - Rationale: Promotes energy-efficient control strategies

5. **Cart Velocity Penalty** (Damping Term)
   ```
   R_cart_velocity = -|v_cart| × 0.01
   ```
   - Penalty for fast cart movement
   - Coefficient: 0.01
   - Encourages gentle, controlled cart motions
   - Rationale: Prevents excessive cart wandering

### Why Both Functions?

**Baseline Reward:**
- Simpler to understand
- Tests the agent's ability to learn from sparse feedback
- Demonstrates the limitations of naive reward design

**Shaped Reward:**
- Provides denser feedback signals
- Accelerates convergence and improves final performance
- Demonstrates reward shaping's power in RL
- Real-world policies often require such engineering

This comparison helps understand a fundamental concept in RL: the trade-off between simplicity and learning efficiency.

### How to Run

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB of free disk space
- ~30-60 minutes for full training on CPU

### Building the Docker Image

```bash
# Navigate to project directory
cd double-pendulum-control-rl

# Build the Docker image
docker-compose build
```

### Training the Agent

#### Train with Shaped Reward (Recommended)

```bash
docker-compose run train
```

This runs the default training: 200,000 timesteps with shaped reward function.

#### Train with Baseline Reward

```bash
docker-compose run train-baseline
```

#### Custom Training Parameters

To train with custom parameters, modify the docker-compose.yml or run directly:

```bash
docker-compose run train python src/train.py \
    --timesteps 300000 \
    --reward_type shaped \
    --save_path models/ppo_custom.zip \
    --learning_rate 3e-4 \
    --batch_size 64
```

**Available Arguments:**
- `--timesteps`: Total training steps (default: 200000)
- `--reward_type`: 'baseline' or 'shaped' (default: 'shaped')
- `--save_path`: Path to save model (default: models/ppo_model.zip)
- `--learning_rate`: PPO learning rate (default: 3e-4)
- `--batch_size`: Training batch size (default: 64)

### Evaluating a Trained Agent

#### Run Evaluation with Visualization

```bash
docker-compose run evaluate
```

This loads `models/ppo_shaped.zip` and runs 3 evaluation episodes with pygame visualization.

#### Evaluate a Different Model

```bash
docker-compose run evaluate python src/evaluate.py \
    --model_path models/ppo_baseline.zip \
    --episodes 5
```

#### Save Evaluation as GIF

```bash
docker-compose run evaluate python src/evaluate.py \
    --model_path models/ppo_shaped.zip \
    --episodes 3 \
    --save_gif \
    --gif_path media/agent_demo.gif
```

**Available Arguments:**
- `--model_path`: Path to saved model (default: models/ppo_shaped.zip)
- `--episodes`: Number of episodes to run (default: 3)
- `--save_gif`: Flag to save evaluation as GIF (no argument needed)
- `--gif_path`: Path to save GIF (default: media/agent_demo.gif)

### Plotting Learning Curves

Generate a comparison plot of baseline vs shaped reward training:

```bash
docker-compose run plot
```

This creates `reward_comparison.png` comparing the learning performance of both reward functions.

### Complete Workflow Example

```bash
# Build the image
docker-compose build

# Train both variants (run in parallel)
docker-compose run train &
docker-compose run train-baseline &

# Wait for training to complete, then plot
docker-compose run plot

# Evaluate the trained agent
docker-compose run evaluate
```

### Video Demo Run Commands (8-15 min)

Use these commands in order during your demo recording.

#### Step 1: Show project files

```powershell
Get-ChildItem
Get-ChildItem src
```

#### Step 2: Build Docker image

```bash
docker-compose build
```

#### Step 3: Check the setup

```powershell
python verify_setup.py
```

#### Step 4: Confirm environment spaces

```powershell
python -c "from src.environment import DoublePendulumEnv; env=DoublePendulumEnv(reward_type='baseline'); print('obs:', env.observation_space.shape); print('act:', env.action_space.shape); env.close()"
```

#### Step 5: Quick shaped reward training

```bash
docker-compose run --rm train python src/train.py --timesteps 1000 --reward_type shaped --save_path models/demo_shaped.zip
```

#### Step 6: Quick baseline reward training

```bash
docker-compose run --rm train python src/train.py --timesteps 1000 --reward_type baseline --save_path models/demo_baseline.zip
```

#### Step 7: Evaluate shaped model and save GIF

```bash
docker-compose run --rm evaluate python src/evaluate.py --model_path models/demo_shaped.zip --episodes 1 --save_gif --gif_path media/demo_eval.gif
```

#### Step 8: Generate reward comparison plot

```bash
docker-compose run --rm plot python plot_rewards.py
```

#### Step 9: Show output artifacts

```powershell
Get-ChildItem media
Get-ChildItem models
Get-Item reward_comparison.png
```

#### Step 10: Show README top section during the explanation

```powershell
Get-Content README.md | Select-Object -First 40
```

#### Step 11: Show project status at the end

```powershell
git status --short
```

### Suggested Demo Order

1. Introduction and problem statement
2. High-level architecture
3. Code walkthrough
4. Environment and reward explanation
5. Training demo
6. Evaluation demo
7. Plot generation
8. Required output files
9. Testing and verification
10. Hardest part, limitations, and improvements
11. Closing summary

### Demo Testing Checklist

1. `docker-compose build` succeeds without errors.
2. Shaped training creates `models/demo_shaped.zip`.
3. Baseline training creates `models/demo_baseline.zip`.
4. Evaluation command creates `media/demo_eval.gif`.
5. Plot command creates or updates `reward_comparison.png`.
6. `python verify_setup.py` completes successfully.

## Project Structure

```
double-pendulum-control-rl/
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Container orchestration
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── README.md                 # This file
├── plot_rewards.py           # Script to generate learning curves
│
├── src/
│   ├── __init__.py
│   ├── environment.py        # DoublePendulumEnv class
│   ├── train.py              # Training script
│   └── evaluate.py           # Evaluation script
│
├── models/                   # Trained model storage
│   ├── ppo_baseline.zip      # Baseline reward model
│   ├── ppo_shaped.zip        # Shaped reward model
│   └── ...
│
├── logs/                     # Training logs
│   ├── baseline/
│   │   └── training_metrics.csv
│   └── shaped/
│       └── training_metrics.csv
│
└── media/                    # Visualizations and GIFs
    ├── agent_initial.gif     # Early stage performance
    ├── agent_final.gif       # Trained agent performance
    └── reward_comparison.png  # Learning curves plot
```

## Key Implementation Details

### DoublePendulumEnv Class

Located in `src/environment.py`, this class implements the Gym interface:

- `__init__()`: Initialize physics space and agent spaces
- `reset()`: Clear environment and create new episode
- `step(action)`: Apply action, update physics, return observation/reward/done
- `render()`: Visualize state using pygame
- `_get_observation()`: Extract state vector from physics simulation
- `_calculate_reward()`: Compute reward based on selected function
- `_is_done()`: Check termination conditions
- `close()`: Clean up resources

### Training Configuration

The PPO agent uses these hyperparameters:

```python
{
    'policy': 'MlpPolicy',              # 2-layer MLP
    'learning_rate': 3e-4,              # Adam optimizer
    'n_steps': 2048,                    # Rollout buffer size
    'batch_size': 64,                   # Mini-batch size
    'n_epochs': 10,                     # Update epochs per rollout
    'gamma': 0.99,                      # Discount factor
    'gae_lambda': 0.95,                 # GAE lambda parameter
    'clip_range': 0.2                   # PPO clipping parameter
}
```

These values are tuned for stability and performance on this task.

## Performance Expectations

### Typical Learning Curves

- **Baseline Reward**: Slower convergence, final mean reward ~0.5-1.0
- **Shaped Reward**: Faster convergence, final mean reward ~1.5-2.0 (maximum possible)

### Training Timeline

- **First 10k steps**: Random exploration, minimal progress
- **10k-50k steps**: Policy begins learning, occasional success
- **50k-150k steps**: Rapid improvement, consistent balancing
- **150k+ steps**: Convergence toward optimal policy, marginal improvements

### Hardware Performance

- **CPU (4-core)**: ~3-5 hours for 200k steps
- **GPU (CUDA)**: ~30-60 minutes for 200k steps
- **Memory**: ~2-3 GB RAM required

## Troubleshooting

### Agent Not Learning

1. **Check Reward Signal**: Visualize rewards to ensure they're meaningful
   ```python
   env = DoublePendulumEnv(reward_type='shaped')
   obs = env.reset()
   for _ in range(100):
       action = np.random.uniform(-1, 1, (1,))
       obs, reward, done, _ = env.step(action)
       print(f"Reward: {reward:.3f}")  # Should see variation
   ```

2. **Verify Observation Space**: Ensure agent sees all required information
   - Check that both angular velocities are included
   - Verify observation values change with action

3. **Tune Hyperparameters**: Try different learning rates or batch sizes
   - Increase learning rate if convergence is too slow
   - Reduce learning rate if training is unstable

### Physics Simulation Issues

1. **Unstable Behavior**: Poles spinning wildly or exploding
   - Reduce time step (dt) or increase constraints' iterations
   - Check joint anchor points are correct

2. **Unrealistic Motion**: Cart or poles moving unnaturally
   - Verify mass values are reasonable
   - Check gravity and damping parameters

### Docker Issues

1. **Display/Rendering Errors**: Can't visualize pygame window
   - On Linux: Ensure X11 forwarding is enabled
   - May need: `xhost +local:docker`

2. **Out of Memory**: Docker container runs out of RAM
   - Increase Docker memory allocation
   - Reduce batch size or n_steps

## References

### Papers

- Schulman et al., "Proximal Policy Optimization Algorithms" (2017) - PPO algorithm
- Ng et al., "Policy Invariance Under Reward Transformations" - Reward shaping theory
- Brockman et al., "OpenAI Gym" - RL environment standard

### Documentation

- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Pymunk Documentation](https://pymunk.readthedocs.io/)
- [Pygame Documentation](https://www.pygame.org/docs/)
- [OpenAI Gym](https://gym.openai.com/)

## License

This project is provided as educational material.

## Author

Created for the Partnr RL Challenge - Double Pendulum Control Task

---

**Last Updated**: March 2026
