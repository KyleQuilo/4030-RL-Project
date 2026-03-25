# 4030 RL Project

## Safe RL Navigation for TurtleBot3 — PPO vs SAC
**AISE 4030 Phase 3 | RL Group 10**

## Project Summary

This project trains a TurtleBot3 Burger robot to autonomously navigate to goal positions while avoiding obstacles in Gazebo, using ROS 2 as the robot middleware. Two reinforcement learning algorithms are implemented and compared:

- **PPO** (Proximal Policy Optimization) — on-policy baseline
- **SAC** (Soft Actor-Critic) — off-policy advanced algorithm

The environment is wrapped in a Gymnasium-compatible API. Both agents are trained with identical observation spaces, reward functions, and evaluation protocols so performance differences reflect algorithm choice rather than implementation differences.

## System Requirements

- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Sim
- Python 3.10 or later
- NVIDIA GPU recommended for full training runs (CPU sufficient for short test runs)

## Python Dependencies

Install all required packages:
```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
gymnasium
stable-baselines3
numpy
pyyaml
torch
matplotlib
pandas
```

## ROS Packages

TurtleBot3 packages for ROS 2 Jazzy must be installed via `apt`:
```bash
sudo apt install ros-jazzy-turtlebot3* ros-jazzy-turtlebot3-gazebo
```

## Setup

### 1. Source ROS 2
```bash
source /opt/ros/jazzy/setup.bash
```

### 2. Set the TurtleBot3 model
```bash
export TURTLEBOT3_MODEL=burger
```

### 3. Launch the Gazebo simulation
```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```
Leave this terminal running for all subsequent commands.

### 4. Verify ROS topics
```bash
ros2 topic list
ros2 topic info /scan
ros2 topic info /odom
ros2 topic info /cmd_vel
```

## Compute Options

### Option B: Local Machine with NVIDIA GPU (recommended)

If your machine has an NVIDIA GPU, train locally for full control and no session timeouts.

**Requirements:**
- Correct NVIDIA drivers installed
- CUDA toolkit installed (matching your PyTorch build)
- cuDNN installed
- PyTorch built with CUDA support (`pip install torch --index-url https://download.pytorch.org/whl/cu121`)

**Verify CUDA is available:**
```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**Enable GPU training** by setting `device: cuda` in `config.yaml` under the `training` section (this is already the default). Change to `device: cpu` if no GPU is available.

### Option A: CPU-Only

Set `device: cpu` in `config.yaml`. Suitable for short test runs but will be slow for 500k timesteps.

---

## Running Training

With Gazebo running, open a second terminal:

### Train both PPO and SAC (recommended)
```bash
python3 training_script.py --algo both
```

### Train PPO only
```bash
python3 training_script.py --algo ppo
```

### Train SAC only
```bash
python3 training_script.py --algo sac
```

Training outputs are saved to `ppo_results/` and `sac_results/` respectively, including:
- Model checkpoints every 50,000 steps
- Best model checkpoint (`best_model.zip`)
- Per-episode metrics CSV (`metrics.csv`)

## Evaluating Trained Agents

```bash
python3 training_script.py --algo both --eval-only --n-eval-episodes 50
```

This loads saved models from `ppo_results/ppo_model.zip` and `sac_results/sac_model.zip` and runs them in deterministic (exploitation-only) mode.

## Generating Comparison Plots

After training completes:
```bash
python3 plot_results.py
```

Plots are saved to `plots/`:
- `reward_curves.png` — learning speed comparison
- `loss_curves.png` — loss convergence comparison
- `stability.png` — rolling reward standard deviation
- `exploration.png` — entropy parameter schedule
- `final_performance.png` — mean ± std bar chart

Custom paths:
```bash
python3 plot_results.py --ppo-csv ppo_results/metrics.csv \
                         --sac-csv sac_results/metrics.csv \
                         --out-dir plots/
```

## Reproducing Results

All hyperparameters and training settings are stored in `config.yaml`. To reproduce a training run exactly:

1. Ensure `training.seed: 42` (or set to your desired seed) in `config.yaml`
2. Use the same ROS 2 and Gazebo world (`turtlebot3_world`)
3. Run `python3 training_script.py --algo both`

Key reproducibility settings in `config.yaml`:
- `training.seed: 42`
- `training.ppo_timesteps: 500000`
- `training.sac_timesteps: 500000`

## Configuration

All parameters are in `config.yaml`. Key sections:

| Section | Description |
|---|---|
| `env` | LiDAR beams, control rate, episode length, velocity limits |
| `goal` | Goal distance range, success tolerances |
| `reward` | All reward coefficients from Phase 1 MDP design |
| `ppo` | PPO hyperparameters (gamma, n_steps, lr, clip_range, etc.) |
| `sac` | SAC hyperparameters (buffer_size, tau, ent_coef, etc.) |
| `training` | Timesteps, eval frequency, checkpoint frequency, seed |
| `paths` | Output paths for models, logs, and metrics CSVs |

## Reward Function

The reward function implements the Phase 1 MDP design:

| Component | Value | Condition |
|---|---|---|
| Progress shaping | `+10 × (d_prev − d_curr)` | Every step |
| Time penalty | `−0.5` | Every step |
| Safety margin | `−2.0` | When min LiDAR range < 0.30 m |
| Success | `+200` | Distance < 0.25 m and heading < 15° |
| Collision | `−200` | Min LiDAR range < 0.20 m |
| Timeout | `−50` | Episode reaches max_steps (600) |

## Observation Space (64-dim)

| Index | Feature |
|---|---|
| 0–59 | Normalized LiDAR beams (60 beams, range [0, 1]) |
| 60 | Normalized distance to goal (dist / 5.0) |
| 61 | Normalized heading error to goal (radians / π) |
| 62 | Current linear velocity (m/s) |
| 63 | Current angular velocity (rad/s) |

## Action Space (2-dim)

| Index | Action | Range |
|---|---|---|
| 0 | Linear velocity v | [0.0, 0.22] m/s |
| 1 | Angular velocity w | [−2.0, 2.0] rad/s |

## Code Structure

| File | Description |
|---|---|
| `environment.py` | Gymnasium environment wrapper — ROS 2 topics, full reward, goal logic |
| `sensor_processing.py` | LiDAR downsampling and normalization |
| `ppo_agent.py` | PPO baseline agent (SB3 wrapper) |
| `sac_agent.py` | SAC advanced agent (SB3 wrapper) |
| `policy_network.py` | Policy network interface documentation (SB3 MlpPolicy used) |
| `rollout_buffer.py` | Buffer interface documentation (SB3 buffers used) |
| `training_script.py` | Training loop, callbacks, CSV logging, evaluation |
| `plot_results.py` | Comparison plots for Phase 3 report |
| `utils.py` | YAML config loader |
| `config.yaml` | All hyperparameters and paths |
| `requirements.txt` | Python package dependencies |
| `ppo_results/` | PPO training outputs (models, metrics CSV, checkpoints) |
| `sac_results/` | SAC training outputs (models, metrics CSV, checkpoints) |
| `plots/` | Generated comparison figures |
| `launch/` | ROS 2 launch file references |

## Google Colab Training

To train on Colab (free T4 GPU):

1. Upload the project folder to Google Drive
2. Open a new Colab notebook, select Runtime → T4 GPU
3. In the first cell:
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/4030-RL-Project/
!pip install -r requirements.txt
```
4. Note: ROS 2 and Gazebo cannot run in Colab. For Colab training, the environment must be adapted to a standard Gymnasium environment (e.g., by mocking the ROS layer or using a local simulation with a bridge).
