
# Smart Rocket Landing GNC System

## Overview
This project involves the simulation and reinforcement learning training of a rocket landing task. The rocket, represented as a rigid body with a thin rod, is modeled in a simulated environment where it must land on a pre-defined point on the ground. The simulation is built using the Gymnasium framework.

## Environment
- The environment simulates a rocket with physical properties such as acceleration, angular acceleration, and air resistance.
- The simulation area has defined world boundaries within which the rocket operates.
- The rocket's task is to land at a specific target point with constraints on its velocity and orientation at the moment of landing.

## Features
- **Control Mechanism:** The rocket is controlled through actions that modify its thrust and nozzle angular acceleration from predetermined options defined in an Action Table.
- **State Representation:** The state of the rocket includes its position, velocity, angle, and more.
- **Reward Function:** A reward function is implemented based on the distance from the target point, the angle of the rocket, and the speed and angle at the moment of contact with the ground.

## Reinforcement Learning
- The model is trained using the Proximal Policy Optimization (PPO) algorithm from `stable-baselines3`.
- The learning environment is vectorized to allow multiple instances for faster training.
- The model learns to optimize actions to successfully land the rocket within the defined constraints.

## Requirements
- Gymnasium
- NumPy
- OpenCV
- PyTorch
- Stable Baselines3

## Usage
To train the model, run the `train.py` script which initializes the environment, loads or creates a new model, and starts the training process. After training, the model is saved for future use or further training.

```python
# Training Example
python train_rocket.py
```

To test the model, one is able to perform the inference by running the `inference.py` script, when a model has already been trained.

```python
# Inference Example
python train_rocket.py
```

## Author
Leonardo Russo.

