# Double Pendulum Control with Reinforcement Learning

This project implements a custom 2D double inverted pendulum control task using pymunk and pygame, and trains a PPO agent with Stable Baselines3.

### Environment Design

The environment class is implemented in src/environment.py as DoublePendulumEnv.

System components:
- Cart body constrained to horizontal movement with a GrooveJoint.
- Pole 1 attached to the cart with a PivotJoint.
- Pole 2 attached to pole 1 with a PivotJoint.
- Physics integrated via pymunk.Space.step(dt) at 60 Hz.

State and action spaces:
- Observation space: Box(shape=(6,))
- Observation vector: [cart_x, cart_vx, theta1, omega1, theta2, omega2]
- Action space: Box(shape=(1,), low=-1.0, high=1.0)
- Action is scaled to cart force in step().

Episode termination:
- Cart leaves track bounds.
- Either pole angle exceeds limit.
- Maximum episode step count is reached.

### Reward Function Design

Two reward modes are supported through reward_type.

Baseline reward:
- Formula: R_baseline = cos(theta1) + cos(theta2)
- Rationale: Directly rewards upright pole angles.

Shaped reward:
- Formula:
  R_shaped = cos(theta1) + cos(theta2)
             - 0.1 * |cart_x|
             - 0.01 * (|omega1| + |omega2|)
             - 0.001 * action^2
             - 0.01 * |cart_vx|
- Rationale:
  - Keeps poles upright (core objective).
  - Penalizes leaving center to avoid running off track.
  - Penalizes high angular velocity for smoother stabilization.
  - Penalizes large forces for control efficiency.
  - Penalizes high cart velocity for reduced oscillations.

### How to Run

1. Build containers

```bash
docker-compose build
```

2. Train with shaped reward

```bash
docker-compose run --rm train
```

3. Train with baseline reward

```bash
docker-compose run --rm train-baseline
```

4. Custom training run

```bash
docker-compose run --rm train python src/train.py --timesteps 1000 --reward_type shaped --save_path models/test.zip
```

5. Evaluate a trained model

```bash
docker-compose run --rm evaluate python src/evaluate.py --model_path models/test.zip --episodes 1
```

6. Generate reward curve comparison plot

```bash
docker-compose run --rm plot
```

## Outputs

- Trained models: models/
- Training logs (CSV): logs/
- Reward curve figure: reward_comparison.png
- GIFs: media/agent_initial.gif and media/agent_final.gif

## Notes

- The train script logs mean reward over timesteps to CSV (training_metrics.csv).
- The evaluation script can render with pygame and optionally save GIFs.
- .env.example documents configurable environment variables.
