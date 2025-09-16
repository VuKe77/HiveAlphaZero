# HiveAlphaZero

An implementation of the **AlphaZero algorithm** adapted for the board game **Hive** and extended to chess using a Gym environment.  
This project contains a Hive environment built with `pygame`, an AlphaZero training pipeline, and supporting tools for training, evaluation, and visualization.

---

## 📌 Features

- **Hive Environment**  
  - Custom implementation of the Hive board game with `pygame` visualization.  
  - Rule enforcement, move generation, and game state tracking.  

- **AlphaZero Training Pipeline**  
  - Self-play using Monte Carlo Tree Search (MCTS).  
  - Neural network evaluation of states and moves.  
  - Replay buffer and iterative training.  

- **Chess Environment (Gym)**  
  - A variant for training AlphaZero on chess via OpenAI Gym.  

- **Docker Support**  
  - Provided `Dockerfile` for containerized execution.  

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/VuKe77/HiveAlphaZero.git
cd HiveAlphaZero
