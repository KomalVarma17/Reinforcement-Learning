# Reinforcement-Learning

# Traffic Light Control with Reinforcement Learning

This project applies **reinforcement learning (RL) agents** to optimize traffic signal control at a two-road intersection. The intersection is modeled as a dynamic environment where an agent must perceive queue states, make decisions, and act in real time. The framework reflects how RL pipelines can be extended to autonomous robotics tasks, where perception and decision-making are tightly coupled.

---

## ✨ Highlights

* **Custom Gymnasium environment** simulating a stochastic traffic intersection (arrivals, departures, red→halt lag).
* **Three TD agents**: SARSA, Expected SARSA, Value-Function SARSA.
* **Exploration**: ε-greedy with normalized-Q softmax weighting (all actions have non-zero probability).
* **State-space truncation** for tractability (queue length capped at 20).
* **Episode truncation**: 1800 steps (30 minutes @ 1s per step).
* **Reproducible training/testing** with visualizations of queue dynamics.

---

## 1) Problem Formulation

* **Environment**: Two orthogonal roads (east–west and north–south) controlled by a single light (no yellow phase).
* **Objective**: Minimize total queue length (equivalent to minimizing average waiting time).
* **Time**: Discrete, 1 second per step.

**Traffic Model**

* **Arrivals**:

  * Road-1: Bernoulli(p = 0.28)
  * Road-2: Bernoulli(p = 0.40)
* **Departures**:

  * If green: probability = 0.9
  * If red: departure probability decays linearly to 0 over 10 seconds (mimicking realistic clearing).
* **Safety Constraint**: When a road turns red at time *t*, the other road must remain red until *t+9*. It may turn green at *t+10* unless the first road reactivates.

**MDP Design**

* **State**: `(q1, q2, phase, cooldown)`

  * `q1, q2`: queue lengths (truncated at 20 for learning)
  * `phase`: {R1\_green, R2\_green, both\_red}
  * `cooldown`: \[0–10] counter for red→halt decay
* **Actions**: {switch\_to\_R1, switch\_to\_R2, hold}
* **Reward**: `-(q1 + q2)` per step
* **Discount Factor**: γ = 0.997

---

## 2) Algorithms

* **SARSA (on-policy TD control)**

  * Online update using `(s, a, r, s’, a’)`.
  * Behavior policy: ε-greedy with softmax weighting over normalized Q-values.
* **Expected SARSA**

  * Replaces sampled `Q(s’,a’)` with expectation under ε-greedy policy.
  * Provides smoother updates, reducing variance.
* **Value-Function SARSA (variant)**

  * Maintains only `V(s)`; computes `Q(s,a)` from value estimates on the fly.
  * Requires fewer parameters than full Q-tables; suitable for memory-limited systems.

---

## 3) Training

* Trained for up to **2000 episodes** with learning rate α = 0.1 and discount γ = 0.997.
* Used **ε-greedy exploration** with probabilities scaled by normalized Q-values.
* State truncation ensured feasible Q-table sizes.
* Policies saved as `.npy` files (e.g., `policy1.npy`, `policy2.npy`, `policy3.npy`).

**Outputs during training**:

* Episode returns (average congestion).
* Optional moving averages for stability tracking.

---

## 4) Testing & Visualization

* Each policy tested on a **single 30-minute episode (1800 steps)**.
* Outputs include:

  * Line plots: queue lengths of both roads vs. time.
  * Printed metrics: average total queue length + action sequences.
* Plots (`SA.png`, `ES.png`, `VFS.png`) compare the three learned policies.

---

## 5) Key Implementation Details

* **Queue truncation**: queue length >20 treated as 20 for learning (full queues still simulated).
* **Safety/cooldown**: enforced 10-second red overlap after switching.
* **Reset**: queues initialized randomly between 0–10.
* **Reward**: negative total queue length, encouraging congestion minimization.

---

## 6) Results

* All three agents successfully reduced congestion versus naive switching.
* **Expected SARSA** and **Value-Function SARSA** produced smoother control, with lower variance in queue lengths.
* Learned policies adapt effectively to stochastic arrivals, simulating real-time adaptive decision-making.

---

## 7) Broader Applications

* Framework illustrates how **agent-based decision-making** in dynamic, uncertain systems translates to domains like **autonomous navigation, robotic scheduling, and perception-to-action pipelines**.
* Queue lengths here represent environment perception; in robotics, similar RL methods can map **sensor data (e.g., object detections, depth, LiDAR counts)** into real-time control decisions.
* The pipeline (state design → policy learning → deployment) parallels **embedded robotic control loops** and could be extended via **ROS2 integration** for hardware testing.

---

## 8) Reproducing Results

1. Train policies (SARSA, Expected SARSA, Value-Function SARSA).
2. Test each policy on the intersection environment.
3. Compare visualizations (`SA.png`, `ES.png`, `VFS.png`) and average queue length statistics.

All agents demonstrated improved performance over baseline, with Expected SARSA and Value-Function SARSA showing the most stable learning outcomes.

