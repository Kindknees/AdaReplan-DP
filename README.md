# AdaReplan-DP: Closed-Loop Adaptive Replanning for Diffusion Policies

## Authors & Contributors
黃冠瑋、鄭芯薇、黃浚瑀

This repository implements an adaptive replanner for closed-loop Diffusion Policies. By introducing a **Heuristic TD-Error Trigger** and a **Learned RL-Based Controller**, our method effectively interrupts unreliable action sequences and regenerates trajectories when facing environmental disturbances, successfully bridging the train-test distribution gap.

---

## 1. Overview & Motivation

### Why Traditional Replanning Fails in Diffusion Policies

While trajectory-denoising methods (*Diffuser*) and Model Predictive Control (MPC) explicitly model future states and can utilize state-deviation metrics $\|\hat{s}_{t+1} - s_{t+1}\|$ to replan , **Diffusion Policies** condition their action chunks *only* on the current state $s_t$ without a next-state prediction head. This open-loop execution mechanism makes them highly vulnerable to unexpected physical disturbances.

When attempting to introduce adaptive replanning naively, two primary failures collapse the policy:

- **Per-step Replanning (`act_steps = 1`) breaks continuity:** Forcing a rewrite at every step completely destroys trajectory smoothness. Since diffusion models are inherently **multimodal**, the agent erratically switches intents mid-task (e.g., oscillating between a left-side and right-side grasp), resulting in catastrophic jitter.

 
- **Naive Triggers encounter Out-of-Distribution (OOD) states:** Interrupting an action chunk mid-way forces the base policy to generate trajectories from half-executed configurations it never encountered during vanilla open-loop training.



**Core Insight:** To enable safe closed-loop control, we must close this mid-chunk distribution shift during training while preserving multimodal consistency.

---

## 2. Core Methodologies

### Method A: Heuristic TD-Triggered Replan

We introduce a zero-extra-parameter replanning indicator by tracking downside surprises in the environment utilizing the pre-trained diffusion critic $V_\theta$. At each environment step, the TD residual is calculated as:


$$\delta_t = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t)$$



A pooled Exponential Moving Average (EMA) maintains the running mean ($\mu$) and variance ($\sigma^2$) across parallel rollouts. The execution chunk is aborted and resampled when the normalized z-score drops below a downside threshold $\tau$:


$$\text{trigger}(t) = \mathbb{1}\left[z_t < -\tau\right], \quad z_t = \frac{\delta_t - \mu}{\sigma}$$



### Method B: Learned RL-Based Replanner

To learn richer, state-dependent decision boundaries than a fixed z-score threshold, we train a lightweight **Shared-Trunk Actor-Critic Bernoulli Controller** ($\pi_{\mathrm{ctrl}}$).

#### 1. Architecture

The controller processes a joint state-action-critic feature vector:


$$\phi_t = \bigl(s_t, a_t^{\mathrm{queued}}, V_\theta(s_t), \delta_{t-1}, \tfrac{\mathrm{cursor}_t}{H}\bigr)$$



* 
**Policy Head (Actor):** Outputs the Bernoulli logit to choose $u_t \in \{0=\text{continue}, 1=\text{replan}\}$.


* 
**Value Head $V_{\mathrm{ctrl}}(\phi_t)$ (Critic):** Serves as the variance-reducing PPO advantage baseline. A dedicated critic head is mathematically required here because the controller optimizes an **augmented reward** containing a compute penalty, rendering the base diffusion critic $V_\theta$ unaligned.



#### 2. Reward Design

The environment reward is augmented with a Number of Function Evaluations (NFE) penalty to explicitly regulate computational overhead:


$$r^{\mathrm{ctrl}}_t = r^{\mathrm{env}}_t - \lambda \cdot \mathrm{NFE}_t$$

During PPO training, the underlying base diffusion models remain completely **frozen**.

---

## 3. Experimental Results

We evaluate our adaptive replanners on **Robomimic Lift and Can tasks**. To challenge robustness, a horizontal Gaussian velocity impulse ($\sigma_{\mathrm{kick}} = 0.2\text{ m/s}$, $p=0.2$) is injected during physical simulation sub-steps to mimic real-world external disturbances.

### Quantitative Evaluation on Robomimic (Noisy Settings)

| Task | Method | Success Rate $\uparrow$ | Expected Reward $\uparrow$ | Replan Rate |
| --- | --- | --- | --- | --- |
| **Lift** | Vanilla Baseline | 0.820 | 56.82 | 0.250 |
|  | **TD Heuristic Trigger** | 0.855 | 56.59 | 0.255 |
|  | **RL Controller (Ours)** | **0.980** | **142.85** | 0.297 |
|  | *Oracle (Environment Flag)* | 0.965 | 99.535 | 0.340 |
| **Can** | Vanilla Baseline | 0.830 | 144.29 | 0.250 |
|  | **TD Heuristic Trigger** | 0.860 | 149.47 | 0.296 |
|  | **RL Controller (Ours)** | **0.870** | **152.19** | 0.257 |
|  | *Oracle (Environment Flag)* | 0.875 | 152.72 | 0.362 |



Note: In the Lift task, the learned RL Controller successfully discovers closed-loop recovery behaviors that outperform the human-engineered heuristic Oracle baseline.

---

## 4. Setup and Installation

### Environment Setup
1. install dependencies
```
cd D3P
pip install -e ".[robomimic]"
```

2. [Install MuJoCo for Gym and/or Robomimic](installation/install_mujoco.md)
3. Set environment variables for data and logging directory (default is `data/` and `log/`), and set WandB entity (username or team name)
```
source script/set_path.sh
```

---

## Acknowledgements

This project was developed as part of the **NYCU 535514 Reinforcement Learning** curriculum at National Yang Ming Chiao Tung University. We build with reproducing the paper [D3P](https://arxiv.org/pdf/2508.06804) .