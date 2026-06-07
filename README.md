# Adaptive Replanner in Diffusion Policy via Reinforcement Learning

[Website](https://koukanni.github.io/adaptive-replanner/)

## Authors & Contributors
黃冠瑋、鄭芯薇、黃浚瑀

This repository implements an adaptive replanner for closed-loop Diffusion Policies. By introducing a **Heuristic TD-Error Trigger** and a **Learned RL-Based Controller**, our method effectively interrupts unreliable action sequences and regenerates trajectories when facing environmental disturbances, successfully bridging the train-test distribution gap.

---

## 1. Overview & Motivation

### Why Traditional Replanning Fails in Diffusion Policies

While trajectory-denoising methods (*Diffuser*) and Model Predictive Control (MPC) explicitly model future states, allowing them to utilize state-deviation metrics to trigger replanning:

$$\Vert \hat{s}_{t+1} - s_{t+1} \Vert$$

**Diffusion Policies** condition their action chunks *only* on the current state $s_t$ without a next-state prediction head. This open-loop execution mechanism makes them highly vulnerable to unexpected physical disturbances.

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

## 5. Usage - Fine-tuning

<!-- ### Set up pre-trained policy -->

<!-- If you did not set the environment variables for pre-training, we need to set them here for fine-tuning. 
```console
export DPPO_WANDB_ENTITY=<your_wandb_entity>
export DPPO_LOG_DIR=<your_prefered_logging_directory>
``` -->
<!-- First create a directory as the parent directory of the downloaded checkpoints and set the environment variable for it.
```console
export DPPO_LOG_DIR=/path/to/checkpoint
``` -->

Pre-trained policies in DPPO paper can be found [here](https://drive.google.com/drive/folders/1ZlFqmhxC4S8Xh1pzZ-fXYzS5-P8sfpiP?usp=drive_link). Fine-tuning script will download the default checkpoint automatically to the logging directory.
 <!-- or you may manually download other ones (different epochs) or use your own pre-trained policy if you like. -->

 <!-- e.g., `${DPPO_LOG_DIR}/gym-pretrain/hopper-medium-v2_pre_diffusion_mlp_ta4_td20/2024-08-26_22-31-03_42/checkpoint/state_0.pt`. -->

<!-- The checkpoint path follows `${DPPO_LOG_DIR}/<benchmark>/<task>/.../<run>/checkpoint/state_<epoch>.pt`. -->

### Fine-tuning pre-trained policy

All the configs can be found under `cfg/<env>/finetune/`. A new WandB project may be created based on `wandb.project` in the config file; set `wandb=null` in the command line to test without WandB logging.
In this project, we mainly finetuned and evaluated in robomimic env.
<!-- Running them will download the default pre-trained policy. -->
<!-- Running the script will download the default pre-trained policy checkpoint specified in the config (`base_policy_path`) automatically, as well as the normalization statistics, to `DPPO_LOG_DIR`.  -->
- Finetune DPPO:
```console
# Robomimic - lift/can/square/transport
python script/run.py --config-name=ft_ppo_diffusion_mlp \
    --config-dir=cfg/robomimic/finetune/can
```

- Finetune D3P:
```console
# Robomimic - lift/can/square/transport
python script/run.py --config-name=ft_d3p_ppo_diffusion_mlp \
    --config-dir=cfg/robomimic/finetune/can
```

- Finetune D3P under noisy environment: (or, you can manually add noisy configuration in env wrappers)
```console
# Robomimic - lift/can/square/transport
python script/run.py --config-name=ft_d3p_ppo_diffusion_mlp_noisy \
    --config-dir=cfg/robomimic/finetune/can

```

- Train RL controller: you must have trained your base diffusion policy before running this
```console
# Robomimic - lift/can/square/transport
python script/run.py --config-name=ft_replan_controller \
    --config-dir=cfg/robomimic/finetune/can
```

**Note**: In Gym, Robomimic, and D3IL tasks, we run 40, 50, and 50 parallelized MuJoCo environments on CPU, respectively. If you would like to use fewer environments (given limited CPU threads, or GPU memory for rendering), you can reduce `env.n_envs` and increase `train.n_steps`, so the total number of environment steps collected in each iteration (n_envs x n_steps x act_steps) remains roughly the same. Try to set `train.n_steps` a multiple of `env.max_episode_steps / act_steps`, and be aware that we only count episodes finished within an iteration for eval. Furniture-Bench tasks run IsaacGym on a single GPU.

To fine-tune your own pre-trained policy instead, override `base_policy_path` to your own checkpoint, which is saved under `checkpoint/` of the pre-training directory. You can set `base_policy_path=<path>` in the command line when launching fine-tuning.

<!-- **Note**: If you did not download the pre-training [data](https://drive.google.com/drive/folders/1AXZvNQEKOrp0_jk1VLepKh_oHCg_9e3r?usp=drive_link), you need to download the normalization statistics from it for fine-tuning, e.g., `${DPPO_DATA_DIR}/furniture/round_table_low/normalization.pkl`. -->

See [here](cfg/finetuning.md) for details of the experiments in the paper.

---

## Usage - Evaluation
Pre-trained or fine-tuned policies can be evaluated without running the fine-tuning script now. Some example configs are provided under `cfg/{gym/robomimic/furniture}/eval}` including ones below. Set `base_policy_path` to override the default checkpoint, and `ft_denoising_steps` needs to match fine-tuning config (otherwise assumes `ft_denoising_steps=0`, which means evaluating the pre-trained policy).

- Vanilla
```console
python script/run.py --config-name=eval_diffusion_mlp \
    --config-dir=cfg/robomimic/eval/can
```

- D3P
```console
python script/run.py --config-name=eval_d3p_diffusion_mlp \
    --config-dir=cfg/robomimic/eval/can
```

- TD replanner
```console
python script/run.py --config-name=eval_d3p_replan_diffusion_mlp \
    --config-dir=cfg/robomimic/eval/can
```

- RL replanner
```console
python script/run.py --config-name=eval_replan_controller_noisy_mlp \
    --config-dir=cfg/robomimic/eval/can
```

---

## Acknowledgements

This project was developed as part of the **NYCU 535514 Reinforcement Learning** curriculum at National Yang Ming Chiao Tung University. We build with reproducing the paper [D3P](https://arxiv.org/pdf/2508.06804) .