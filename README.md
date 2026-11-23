# Motor-Realistic RL

**Training legged robots for real motors, not ideal ones.**

Most simulated robot policies assume perfect motors: instant torque, no current limits, no delay. This project adds an electrical model of the motors to a simulated Unitree Go1 quadruped and asks two questions:

1. Which electrical effects break a policy trained with ideal motors, and by how much?
2. Does training with a realistic motor model bring the robustness back?

> **Status:** in progress (started November 2026). Baseline training is being set up; no results yet.

## How it works

```
Policy  ->  PD controller  ->  Motor model  ->  MuJoCo physics
  ^                                                   |
  +--------------  Sensor model  <--------------------+
```

The simulator stays standard. The work is in the two models between the policy and the physics.

**Motor model** (each effect has an on/off switch, so they can be tested one at a time):

- Control delay
- Voltage and current limits
- Back-EMF, winding resistance and inductance
- Battery sag under load
- Motor heating and torque derating

**Sensor model:** encoder quantization and sensor delay.

## Planned experiments

| # | Experiment | Result it produces |
| --- | --- | --- |
| 1 | Train with ideal motors, then test with each effect switched on | Effects ranked by how much they hurt performance |
| 2 | Retrain with the motor model on and its parameters randomized | How much robustness comes back |
| 3 | Sweep control delay, battery voltage and current limit | The point where the policy breaks |

**Metrics:** survival time, velocity-tracking error, peak motor current, and electrical energy per meter walked.

## Repo layout

| Path | What it is |
| --- | --- |
| `learning/notebooks/locomotion.ipynb` | Baseline Go1 training notebook (runs on Colab) |

The motor model, environment wrapper, experiment scripts and tests will be added as the project progresses.

## Running it

**Training (Google Colab, GPU):** [open the notebook in Colab](https://colab.research.google.com/github/nayans3/motor-realistic-rl/blob/main/learning/notebooks/locomotion.ipynb), set the runtime to GPU, and run the cells in order.

**Local development (motor model and tests):** Python 3.12 with

```
pip install mujoco numpy scipy matplotlib pytest
```

## Credits

- Simulation, robot models and the baseline notebook come from [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) by Google DeepMind, under the Apache License 2.0. `learning/notebooks/locomotion.ipynb` is adapted from their notebook and keeps its original copyright notice.
- Built with help from an AI coding assistant (Claude). The design decisions, experiments and conclusions are my own.

## License

MIT (see `LICENSE`). Third-party code keeps its own license.

---

Nayan Saini · Electrical Engineering, University of Illinois Urbana-Champaign
