# Quadruped Climbing with Isaac Lab

## Overview

This project implements a reinforcement learning environment for training quadruped robots to climb steep terrain and pyramid-shaped hills using Isaac Lab. The environment features curriculum learning, where robots progressively learn to navigate increasingly difficult slopes.

**Key Features:**

- **Curriculum Learning**: Progressive difficulty scaling from flat terrain to steep slopes
- **Pyramid Terrain**: Custom pyramid-shaped hills with configurable slopes and platform sizes
- **Goal-based Rewards**: Robots are rewarded for reaching terrain centers (hilltops)
- **Multiple Terrains**: Support for various terrain types including slopes and pyramids
- **Robust Training**: Optimized PPO configuration for stable quadruped locomotion learning

**Keywords:** quadruped, climbing, reinforcement learning, curriculum learning, isaac lab, locomotion

## Installation

### Prerequisites

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  I recommend using the conda installation as it simplifies calling Python scripts from the terminal.
- NVIDIA GPU with CUDA support (tested with RTX 3000 series)
- Python 3.10+

### Setup

1. **Clone the repository** outside of your Isaac Lab installation:

   ```bash
   git clone <repository-url>
   cd Quadrupeds_Climbing
   ```

2. **Install the project in editable mode**:

   ```bash
   # Activate your Isaac Lab environment first
   conda activate isaaclab
   
   # Install the project
   python -m pip install -e source/quad_climbing
   ```

3. **Verify installation** by listing available tasks:

   ```bash
   python scripts/list_envs.py
   ```

   You should see tasks like:
   - `Isaac-Slope-Unitree-Go2-v0`
   - `Isaac-Slope-Play-Unitree-Go2-v0`

## Development

### Project Structure

```
Quadrupeds_Climbing/
├── source/quad_climbing/
│   └── quad_climbing/
│       ├── tasks/
│       │   └── manager_based/
│       │       └── quad_climbing/
│       │           ├── slope_env_cfg.py      # Environment configuration
│       │           ├── mdp/
│       │           │   ├── rewards.py        # Reward functions
│       │           │   ├── events.py         # Reset events
│       │           │   └── curriculums.py    # Curriculum functions
│       │           └── go2/
│       │               └── agents/
│       │                   └── rsl_rl_ppo_cfg.py  # PPO configuration
│       └── terrains/
│           └── config/
│               └── slope.py                  # Terrain definitions
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py                         # Training script
│   │   └── play.py                          # Evaluation script
│   ├── zero_agent.py                        # Zero-action testing
│   └── random_agent.py                      # Random-action testing
└── logs/                                    # Training logs and models
```

### Key Components

1. **Environment Configuration** (`slope_env_cfg.py`):
   - Defines robot, terrain, and simulation parameters
   - Configures reward terms and curriculum

2. **Reward Functions** (`rewards.py`):
   - `distance_improvement`: Rewards approaching hill centers
   - Various stability and efficiency rewards

3. **Curriculum System** (`curriculums.py`):
   - `pyramid_max_height`: Dynamically adjusts terrain difficulty
   - Progressive learning from easy to hard terrains

4. **Reset Events** (`events.py`):
   - `reset_state_curriculum`: Spawns robots at appropriate heights
   - Handles robot placement based on difficulty level
   - **Note**: Contains hardcoded height values that must be recalculated if terrain parameters change

### Important: Hardcoded Values

⚠️ **Critical Configuration Note**: The current implementation contains hardcoded height values that are specific to the default terrain configuration. If you modify any of the following parameters, you **must** recalculate and update the hardcoded values:

**Parameters that affect height calculations:**
- Slope range (currently 0° to 45°)
- Number of curriculum levels (currently 10)
- Terrain size (currently 8m × 8m)
- Platform width (currently 2.0m)

**Files containing hardcoded values:**
1. **`events.py`** - `reset_state_curriculum` function

2. **`curriculums.py`** - 'terrain_levels_vel' aka curriculum progression function

**To modify terrain parameters:**
1. Calculate new maximum heights using the `pyramid_max_height` function in curriculums.py to print out the new heights based on your terrain configuration
2. Generate new height arrays for each curriculum level
3. Update the hardcoded values in both `events.py` and `curriculums.py`
4. Ensure spawn heights in `events.py` is the negative of the hill heights because the top of the hill is the origin


### Adding New Features

To add custom reward functions:

```python
# In rewards.py
def my_custom_reward(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Custom reward function."""
    # Your reward logic here
    return reward_tensor

# In slope_env_cfg.py
@configclass
class RewardsCfg:
    my_reward = RewTerm(func=mdp.my_custom_reward, weight=1.0)
```

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**:
   ```bash
   # Reduce number of environments
   python scripts/rsl_rl/train.py --task Isaac-Slope-Unitree-Go2-v0 --num_envs 512
   ```

2. **CUDA Errors**:
   ```bash
   # Check GPU status
   nvidia-smi
   
   # Restart if needed
   sudo nvidia-smi --gpu-reset
   ```

3. **Import Errors**:
   ```bash
   # Reinstall package
   pip install -e source/quad_climbing
   ```

4. **Training Instability**:
   - Reduce learning rate in `rsl_rl_ppo_cfg.py`
   - Increase number of environments for better sample diversity
   - Check reward scaling and clipping

5. **Terrain Configuration Issues**:
   - If you modify terrain parameters (slope, levels, size), you must recalculate hardcoded height values
   - Check that spawn heights in `events.py` match terrain heights from curriculum
   - Verify that robots don't spawn underground or too high above terrain

### Performance Tips

- Use 2048+ environments for stable training
- Monitor GPU memory usage with `nvidia-smi`
- Save checkpoints frequently (`save_interval=100`)
- Use tensorboard for training visualization:
  ```bash
  tensorboard --logdir logs/rsl_rl/
  ```

## Usage

### Training

Train a quadruped robot to climb hills using PPO:

```bash
# Basic training with 2048 environments
python scripts/rsl_rl/train.py --task Isaac-Slope-Unitree-Go2-v0 --num_envs 2048

# Training with fewer environments (for limited GPU memory)
python scripts/rsl_rl/train.py --task Isaac-Slope-Unitree-Go2-v0 --num_envs 512

# Training with custom parameters
python scripts/rsl_rl/train.py --task Isaac-Slope-Unitree-Go2-v0 --num_envs 1024 --max_iterations 5000
```

### Evaluation

Evaluate a trained model:

```bash
# Play with a trained model
python scripts/rsl_rl/play.py --task Isaac-Slope-Play-Unitree-Go2-v0 --num_envs 16 --checkpoint /path/to/model.pt

# Example with specific checkpoint
python scripts/rsl_rl/play.py --task Isaac-Slope-Play-Unitree-Go2-v0 --num_envs 5 --checkpoint logs/rsl_rl/unitree_go2_slope/2025-07-10_14-24-44/model_1000.pt
```

### Testing Environment

Test the environment setup with dummy agents:

```bash
# Zero-action agent (robot stays still)
python scripts/zero_agent.py --task Isaac-Slope-Unitree-Go2-v0

# Random-action agent (robot moves randomly)
python scripts/random_agent.py --task Isaac-Slope-Unitree-Go2-v0
```

## Environment Details

### Terrain Configuration

The environment features pyramid-shaped hills with configurable parameters:

- **Slope Range**: 0° to 45° (configurable)
- **Platform Width**: 2.0m at hilltop
- **Terrain Size**: 8m × 8m per environment
- **Curriculum Levels**: 10 difficulty levels

### Reward Structure

The robot is rewarded for:
- **Distance Improvement**: Moving closer to the hill center (goal)
- **Stability**: Maintaining upright orientation
- **Energy Efficiency**: Minimizing joint torques and accelerations
- **Foot Contact**: Proper ground contact patterns

### Curriculum Learning

The training uses curriculum learning with:
- **Progressive Difficulty**: Starts with flat terrain, increases slope gradually
- **Adaptive Spawning**: Robots spawn at appropriate heights for their skill level
- **Dynamic Goals**: Hill centers serve as navigation targets

### Observations

The robot receives observations including:
- Joint positions and velocities
- Base orientation and angular velocity
- Contact forces at feet
- Terrain height information
- Goal direction and distance

## Configuration

### Training Parameters

Key training parameters can be adjusted in `go2/agents/rsl_rl_ppo_cfg.py`:

```python
# Episode length (50 seconds at 0.05s timestep)
num_steps_per_env = 1000

# Total training iterations
max_iterations = 5000

# Network architecture
policy = RslRlPpoActorCriticCfg(
    actor_hidden_dims=[1024, 512, 256],
    critic_hidden_dims=[1024, 512, 256],
)
```

### Environment Parameters

Modify environment settings in `slope_env_cfg.py`:

```python
# Terrain configuration
terrain_cfg = SLOPE_TERRAIN_CFG

# Reward weights
rewards = RewardsCfg(
    distance_to_goal=RewTerm(func=mdp.distance_improvement, weight=1.0),
    lin_vel_z_l2=RewTerm(func=mdp.lin_vel_z_l2, weight=-3.0),
    # ... other rewards
)
```

## Results

Training typically shows:
- **Convergence**: 2000-3000 iterations for basic climbing
- **Success Rate**: 80%+ on moderate slopes (20-30°)
- **Robustness**: Handles terrain variations and disturbances

Trained models are saved in `logs/rsl_rl/unitree_go2_slope/` with timestamps.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) for the robotics simulation framework
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) for the physics simulation
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) for the reinforcement learning algorithms
- Unitree Robotics for the Go2 quadruped robot model

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{quadruped_climbing_2025,
  title={Quadruped Climbing with Curriculum Learning in Isaac Lab},
  author={Leo Wang},
  year={2025},
  link={\url{https://github.com/leo01110111/Quadrupeds_Climbing}}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

