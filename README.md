# Double Pendulum Control with Reinforcement Learning

This project builds a custom 2D double inverted pendulum environment with pymunk + pygame and trains a PPO agent (Stable-Baselines3) to balance both poles.

## Project Overview

- Environment class: `DoublePendulumEnv` in `src/environment.py`
- Training entrypoint: `src/train.py`
- Evaluation entrypoint: `src/evaluate.py`
- Plotting utility: `plot_rewards.py`

### Environment Design

The environment models a cart moving horizontally with two linked poles.

Core physics components:

- Cart constrained by a `GrooveJoint`
- Pole 1 attached to cart via `PivotJoint`
- Pole 2 attached to pole 1 via `PivotJoint`
- Fixed-step simulation using `pymunk.Space.step(dt)` at 60 Hz (`dt = 1/60`)

State and action spaces:

- Observation space: `Box(shape=(6,))`
- Observation vector: `[cart_x, cart_vx, theta1, omega1, theta2, omega2]`
- Action space: `Box(shape=(1,), low=-1.0, high=1.0)`
- Action is scaled to force in the environment step

Episode termination:

- Cart exceeds track bounds
- Either pole exceeds angle threshold
- Maximum episode length reached

### Reward Function Design

Two reward modes are implemented via `reward_type`: `baseline` and `shaped`.

Baseline reward:

- `R_baseline = cos(theta1) + cos(theta2)`
- Purpose: reward upright poles directly

Shaped reward:

- `R_shaped = cos(theta1) + cos(theta2)`
- `- 0.1 * |cart_x|`
- `- 0.01 * (|omega1| + |omega2|)`
- `- 0.001 * action^2`
- `- 0.01 * |cart_vx|`

Behavior encouraged/discouraged by each term:

- Upright term: encourages balancing both poles vertically
- Center penalty: discourages drifting toward track edges
- Angular velocity penalty: discourages violent swinging
- Action penalty: discourages excessive force usage
- Cart velocity penalty: discourages unnecessary rushing/oscillation

### PPO Hyperparameter Tuning

Method used:

- Manual tuning with short runs first, then longer runs for confirmation

Most impactful hyperparameters:

- `learning_rate`: strongest impact on training stability
- `n_steps`: impacted rollout quality and update consistency
- `batch_size`: improved smoothness of gradient updates

### Physics Design Challenges

Main issues encountered:

- Simulation stability
- Joint anchor correctness
- Sensitivity to mass/inertia and timestep choices

How they were handled:

- Fixed timestep at `1/60`
- Reasonable mass/inertia values
- Corrected joint anchor placement
- Added damping to reduce jitter/instability

### How to Run

1. Build containers

```bash
docker-compose build
```

1. Train with shaped reward

```bash
docker-compose run --rm train
```

1. Train with baseline reward

```bash
docker-compose run --rm train-baseline
```

1. Custom training run

```bash
docker-compose run --rm train python src/train.py --timesteps 1000 --reward_type shaped --save_path models/test.zip
```

1. Evaluate a trained model

```bash
docker-compose run --rm evaluate python src/evaluate.py --model_path models/test.zip --episodes 1
```

1. Generate reward comparison plot

```bash
docker-compose run --rm plot
```

## Robustness Evaluation Plan

Proposed disturbance test:

- Apply random force disturbances during evaluation
- Run multiple disturbance levels and random seeds

Metrics:

- Success rate
- Average episode length
- Average reward
- Recovery time after disturbance

## Repository Outputs

- `reward_comparison.png`
- `media/agent_initial.gif`
- `media/agent_final.gif`

Generated artifacts (not committed):

- `logs/`
- `models/`

## Notes

- Training logs include `timesteps` and `mean_reward`
- Evaluation supports rendering and optional GIF generation
- `.env.example` documents environment variables
