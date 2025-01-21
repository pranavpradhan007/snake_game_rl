# Snake Game AI with Deep Q-Learning

A Python implementation of the classic Snake game where an AI agent learns to play using Deep Q-Learning (DQL).

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Training Process](#training-process)
- [Usage](#usage)
- [Customization](#customization)

## Overview

This project implements a Snake game where an AI agent learns to play through reinforcement learning. The agent uses Deep Q-Learning to make decisions based on the game state, receiving rewards for eating food and penalties for collisions.

## Requirements

```
pygame
torch
numpy
matplotlib
```

Install requirements using:
```bash
pip install pygame torch numpy matplotlib
```

## Project Structure

```
snake_game_ai/
│
├── snake.py           # Main game implementation
│   ├── SnakeGame     # Game environment
│   ├── Linear_QNet   # Neural network model
│   ├── QTrainer      # Training mechanism
│   └── Agent         # AI agent implementation
│
└── model.pth         # Saved model weights (generated during training)
```

## Implementation Details

### Game Environment (SnakeGame Class)
The game environment provides the interface for the agent to interact with the game.

**State Space (11 values):**
- Danger detection (3 values)
  - Straight ahead
  - Right side
  - Left side
- Current direction (4 values)
  - Left, Right, Up, Down
- Food location (4 values)
  - Left, Right, Up, Down

**Action Space (3 possible actions):**
- [1,0,0] → Straight
- [0,1,0] → Right turn
- [0,0,1] → Left turn

**Reward Structure:**
```python
Rewards = {
    'eat_food': 10,
    'game_over': -10,
    'survive': 0
}
```

### Neural Network Architecture
```python
Input Layer (11 neurons)
    ↓
Hidden Layer (256 neurons + ReLU)
    ↓
Output Layer (3 neurons)
```

### Agent Implementation
The agent uses several key reinforcement learning concepts:

1. **Epsilon Greedy Strategy:**
```python
epsilon = 80 - n_games  # Decreases as more games are played
```

2. **Experience Replay:**
```python
memory = deque(maxlen=100000)  # Stores game experiences
```

3. **Q-Learning Update:**
```python
Q_new = reward + gamma * max(next_predicted Q values)
```

## Training Process

### State Representation
The game state is converted into 11 binary values representing:
- Danger detection (3 values)
- Current direction (4 values)
- Food location relative to snake (4 values)

### Learning Algorithm
1. **Get Current State**
   - Collect 11 binary values representing game state

2. **Predict Action**
   - Use epsilon-greedy strategy
   - Either random action or neural network prediction

3. **Perform Action**
   - Execute move in game environment
   - Collect reward and new state

4. **Train Short Memory**
   - Update Q-values based on immediate experience

5. **Store Experience**
   - Add to replay memory for later training

6. **Train Long Memory**
   - Batch training on random samples from replay memory
   - Occurs after each game

## Usage

Run the training:
```python
python snake.py
```

The program will:
1. Open a game window showing the snake's gameplay
2. Display a plot showing score progression
3. Print game statistics in the console
4. Save the model when new high scores are achieved

Stop training:
- Close the game window
- Press Ctrl+C in the terminal

## Customization

### Adjustable Parameters:
```python
# Game Settings
BLOCK_SIZE = 20
SPEED = 40

# Training Parameters
LEARNING_RATE = 0.001
HIDDEN_LAYER_SIZE = 256
MEMORY_SIZE = 100000
BATCH_SIZE = 1000
GAMMA = 0.9  # Discount rate
```

### Model Saving/Loading:
```python
# Save model
model.save('model.pth')

# Load model
model.load('model.pth')
```

## Performance Metrics

The training progress is visualized through:
- Real-time score display
- Plot showing:
  - Individual game scores
  - Moving average score
  - High score tracking

The model saves automatically when achieving new high scores, allowing for continuous improvement tracking.

---

This implementation demonstrates the application of deep reinforcement learning to a classic game environment, showing how an AI agent can learn complex behaviors through trial and error.
