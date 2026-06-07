"""
DPPO fine-tuning.

"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
import math

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_agent import TrainPPOAgent
from util.scheduler import CosineAnnealingWarmupRestarts
from d3p_utils.D3PAdaptor import D3PAdaptor


class TrainD3PPPODiffusionAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Reward horizon --- always set to act_steps for now
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)

        # Eta - between DDIM (=0 for eval) and DDPM (=1 for training)
        self.learn_eta = self.model.learn_eta
        if self.learn_eta:
            self.eta_update_interval = cfg.train.eta_update_interval
            self.eta_optimizer = torch.optim.AdamW(
                self.model.eta.parameters(),
                lr=cfg.train.eta_lr,
                weight_decay=cfg.train.eta_weight_decay,
            )
            self.eta_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.eta_optimizer,
                first_cycle_steps=cfg.train.eta_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.eta_lr,
                min_lr=cfg.train.eta_lr_scheduler.min_lr,
                warmup_steps=cfg.train.eta_lr_scheduler.warmup_steps,
                gamma=1.0,
            )

        # intialize d3p adaptor
        adaptor_mlp_dims = cfg.get("d3p", {}).get("mlp_dims", [256, 512, 1024, 512, 256])
        stride = cfg.denoising_steps // cfg.ft_denoising_steps
        
        self.adaptor = D3PAdaptor(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            output_mean=stride,
            seq_len=cfg.cond_steps,
            chunk_size=self.horizon_steps,
            mlp_dims=adaptor_mlp_dims 
        ).to(self.device)
        
        self.adaptor_optimizer = torch.optim.Adam(
            self.adaptor.parameters(),
            lr=cfg.get("d3p", {}).get("adaptor_lr", 1e-4)
        )
        
        self.gamma_s = cfg.get("d3p", {}).get("gamma_s", 0.95)
        self.alpha_w = cfg.get("d3p", {}).get("alpha", 1.0)
        self.beta_w = cfg.get("d3p", {}).get("beta", 0.2)

        self.e_slow = max(1, cfg.train.update_epochs // 2)
        self._in_stage3 = False



    def run(self):
        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        while self.itr < self.n_train_itr:
            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            obs_trajs = {"state": np.zeros((self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim))}
            _D = self.model.ddim_steps  # denoising steps
            chains_trajs = np.zeros((self.n_steps, self.n_envs, _D + 1, self.horizon_steps, self.action_dim))
            
            k_trajs = np.zeros((self.n_steps, self.n_envs, _D)) # store the k predicted by the adaptor at each denoising step
            k_logprobs_trajs = np.zeros((self.n_steps, self.n_envs, _D))    # store the logprobs of the k predictions at each denoising step (for calculating adaptor loss)
            valid_mask_trajs = np.zeros((self.n_steps, self.n_envs, _D)) # store the valid mask for each denoising step (for calculating adaptor loss)
            total_steps_trajs = np.zeros((self.n_steps, self.n_envs)) # store the total steps at each environment step (for calculating adaptor loss)

            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv

            # Holder
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            if self.save_full_observations:  # state-only
                obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
                )
            indices_trajs = np.zeros((self.n_steps, self.n_envs, _D), dtype=np.int64)

            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"])
                        .float()
                        .to(self.device)
                    }

                    samples, k_samples, k_logprobs, padded_indices, v_mask, stp_t = self.model.forward_d3p(
                        cond=cond,
                        adaptor=self.adaptor,
                        deterministic=eval_mode,
                    )
                    indices_trajs[step] = padded_indices.cpu().numpy()

                    output_venv = (
                        samples.trajectories.cpu().numpy()
                    )  # n_env x horizon x act
                    chains_venv = (
                        samples.chains.cpu().numpy()
                    )  # n_env x denoising x horizon x act
                action_venv = output_venv[:, : self.act_steps]

                # Apply multi-step action
                (
                    obs_venv,
                    reward_venv,
                    terminated_venv,
                    truncated_venv,
                    info_venv,
                ) = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                if self.save_full_observations:  # state-only
                    obs_full_venv = np.array(
                        [info["full_obs"]["state"] for info in info_venv]
                    )  # n_envs x act_steps x obs_dim
                    obs_full_trajs = np.vstack(
                        (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                    )
                obs_trajs["state"][step] = prev_obs_venv["state"]
                chains_trajs[step] = chains_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv

                prev_obs_venv = obs_venv
                # count steps --- not acounting for done within action chunk
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

                # store D3P decision data
                k_trajs[step] = k_samples.cpu().numpy()
                k_logprobs_trajs[step] = k_logprobs.cpu().numpy()
                valid_mask_trajs[step] = v_mask.cpu().numpy()
                total_steps_trajs[step] = stp_t.cpu().numpy()

            # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                if (
                    self.furniture_sparse_reward
                ):  # only for furniture tasks, where reward only occurs in one env step
                    episode_best_reward = episode_reward
                else:
                    episode_best_reward = np.array(
                        [
                            np.max(reward_traj) / self.act_steps
                            for reward_traj in reward_trajs_split
                        ]
                    )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # Update models
            if not eval_mode:
                with torch.no_grad():
                    obs_trajs["state"] = (
                        torch.from_numpy(obs_trajs["state"]).float().to(self.device)
                    )

                    # Calculate value and logprobs - split into batches to prevent out of memory
                    num_split = math.ceil(
                        self.n_envs * self.n_steps / self.logprob_batch_size
                    )
                    obs_ts = [{} for _ in range(num_split)]
                    obs_k = einops.rearrange(
                        obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                    obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                    for i, obs_t in enumerate(obs_ts_k):
                        obs_ts[i]["state"] = obs_t
                    values_trajs = np.empty((0, self.n_envs))
                    for obs in obs_ts:
                        values = self.model.critic(obs).cpu().numpy().flatten()
                        values_trajs = np.vstack(
                            (values_trajs, values.reshape(-1, self.n_envs))
                        )
                    total_env_steps = self.n_steps * self.n_envs
                    indices_k = torch.tensor(indices_trajs, device=self.device).long().view(-1, _D)
                    k_k_floor = torch.floor(torch.tensor(k_trajs, device=self.device).float()).long().clamp(min=1).view(-1, _D)
                    valid_mask_flat = torch.tensor(valid_mask_trajs, device=self.device).float().view(-1)
                    chains_k = torch.tensor(chains_trajs, device=self.device).float().view(-1, _D + 1, self.horizon_steps, self.action_dim)
                    obs_state_k = obs_trajs["state"].view(-1, *obs_trajs["state"].shape[2:])

                    valid_inds = torch.where(valid_mask_flat > 0)[0]
                    logprobs_k_flat = torch.zeros((len(valid_mask_flat), self.horizon_steps, self.action_dim), device=self.device)

                    chunk_size = self.logprob_batch_size
                    for start in range(0, len(valid_inds), chunk_size):
                        end = min(start + chunk_size, len(valid_inds))
                        batch_idx = valid_inds[start:end]
                        
                        b_inds, d_inds = torch.unravel_index(batch_idx, (len(obs_state_k), _D))
                        
                        obs_chunk = {"state": obs_state_k[b_inds]}
                        chains_prev_chunk = chains_k[b_inds, d_inds]
                        chains_next_chunk = chains_k[b_inds, d_inds + 1]
                        indices_chunk = indices_k[b_inds, d_inds]
                        k_chunk = k_k_floor[b_inds, d_inds]
                        
                        # Calculate logprob specifically for the dynamic step
                        logp = self.model.get_logprobs_subsample(
                            obs_chunk, chains_prev_chunk, chains_next_chunk, d_inds,
                            indices_b=indices_chunk, k_b=k_chunk
                        )
                        logprobs_k_flat[batch_idx] = logp

                    logprobs_k     = logprobs_k_flat.view(-1, _D, self.horizon_steps, self.action_dim)
                    logprobs_trajs = logprobs_k.cpu().numpy()

                    # normalize reward with running variance if specified
                    if self.reward_scale_running:
                        reward_trajs_transpose = self.running_reward_scaler(
                            reward=reward_trajs.T, first=firsts_trajs[:-1].T
                        )
                        reward_trajs = reward_trajs_transpose.T

                    # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
                    obs_venv_ts = {
                        "state": torch.from_numpy(obs_venv["state"])
                        .float()
                        .to(self.device)
                    }
                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            nextvalues = (
                                self.model.critic(obs_venv_ts)
                                .reshape(1, -1)
                                .cpu()
                                .numpy()
                            )
                        else:
                            nextvalues = values_trajs[t + 1]
                        nonterminal = 1.0 - terminated_trajs[t]
                        # delta = r + gamma*V(st+1) - V(st)
                        delta = (
                            reward_trajs[t] * self.reward_scale_const
                            + self.gamma * nextvalues * nonterminal
                            - values_trajs[t]
                        )
                        # A = delta_t + gamma*lamdba*delta_{t+1} + ...
                        advantages_trajs[t] = lastgaelam = (
                            delta
                            + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                        )
                    returns_trajs = advantages_trajs + values_trajs

                k_k = torch.tensor(k_trajs, device=self.device).float().reshape(-1, _D)
                logprobs_k_k = torch.tensor(k_logprobs_trajs, device=self.device).float().reshape(-1, _D)
                mask_k = torch.tensor(valid_mask_trajs, device=self.device).float().reshape(-1, _D)
                stp_k = torch.tensor(total_steps_trajs, device=self.device).float().reshape(-1)
                success_k = torch.tensor((reward_trajs > 0).astype(float), device=self.device).reshape(-1) # 簡化版成功標記
                env_advantages_k = torch.tensor(advantages_trajs, device=self.device).float().reshape(-1)


                # k for environment step
                obs_k = {"state": einops.rearrange(obs_trajs["state"], "s e ... -> (s e) ...")}
                chains_k = einops.rearrange(torch.tensor(chains_trajs, device=self.device).float(), "s e t h d -> (s e) t h d")
                returns_k = torch.tensor(returns_trajs, device=self.device).float().reshape(-1)
                values_k = torch.tensor(values_trajs, device=self.device).float().reshape(-1)
                advantages_k = torch.tensor(advantages_trajs, device=self.device).float().reshape(-1)
                logprobs_k = torch.tensor(logprobs_trajs, device=self.device).float()

                indices_k  = torch.tensor(indices_trajs, device=self.device).long().view(-1, _D)
                k_k_floor  = torch.floor(torch.tensor(k_trajs, device=self.device).float()).long().clamp(min=1).view(-1, _D)

                # Update
                total_env_steps = self.n_steps * self.n_envs

                # ---- D3P Adaptor Update (paper Eq. 7 + Eq. 8) ----
                # r_K is the terminal reward at j=0; it already contains the γ_s^(sgn·stp)
                # and γ_s^stp penalty factors from Eq. 7.
                sgn_adv = torch.sign(env_advantages_k)
                term1 = self.alpha_w * env_advantages_k * (self.gamma_s ** (sgn_adv * stp_k))
                term2 = self.beta_w * success_k * (self.gamma_s ** stp_k)
                r_K = term1 + term2                                         # (total_env_steps,)

                # Normalize r_K at env-step level for PPO stability (same spirit as DPPO).
                r_K_norm = (r_K - r_K.mean()) / (r_K.std() + 1e-8)

                # Per-denoising-position discount γ_s^(stp - s - 1).
                # Sparse reward + no V_K baseline ⇒ GAE collapses to MC return with γ_s decay.
                positions  = torch.arange(_D, device=self.device, dtype=torch.float)          # (_D,)
                remaining  = (stp_k.unsqueeze(1) - positions.unsqueeze(0) - 1).clamp(min=0)   # (N, _D)
                per_step_d = self.gamma_s ** remaining                                         # (N, _D)
                adv_per_step_all = r_K_norm.unsqueeze(1) * per_step_d                          # (N, _D)

                adaptor_losses = []
                for update_epoch in range(self.update_epochs):
                    inds_adaptor = torch.randperm(total_env_steps, device=self.device)
                    num_adaptor_batch = max(1, total_env_steps // self.batch_size)

                    for batch in range(num_adaptor_batch):
                        start = batch * self.batch_size
                        end   = start + self.batch_size
                        idx   = inds_adaptor[start:end]

                        adv_batch  = adv_per_step_all[idx]    # (B, _D)
                        mask_batch = mask_k[idx]              # (B, _D)

                        obs_b = obs_k["state"][idx]
                        chains_prev_b = chains_k[idx, :-1]    # (B, _D, h, d)

                        obs_repeat  = obs_b.unsqueeze(1).repeat(1, _D, 1, 1).flatten(0, 1)
                        chains_flat = chains_prev_b.flatten(0, 1)

                        dist_new = self.adaptor({"state": obs_repeat}, chains_flat)
                        new_logprobs_k = dist_new.log_prob(k_k[idx].flatten()).reshape(len(idx), -1)

                        ratio_k = torch.exp(new_logprobs_k - logprobs_k_k[idx])
                        surr1 = ratio_k * adv_batch
                        surr2 = torch.clamp(ratio_k, 1.0 - 0.2, 1.0 + 0.2) * adv_batch
                        adaptor_loss = -(torch.min(surr1, surr2) * mask_batch).sum() / (mask_batch.sum() + 1e-8)

                        self.adaptor_optimizer.zero_grad()
                        adaptor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.adaptor.parameters(), 10.0)
                        self.adaptor_optimizer.step()
                        adaptor_losses.append(adaptor_loss.item())

                # Update policy and critic
                total_denoise_steps = self.n_steps * self.n_envs * _D
                clipfracs = []
                valid_inds_actor = torch.where(mask_k.view(-1) > 0)[0]

                for update_epoch in range(self.update_epochs):
                    flag_break = False
                    inds_actor = valid_inds_actor[torch.randperm(len(valid_inds_actor), device=self.device)]
                    num_actor_batch = max(1, len(inds_actor) // self.batch_size)
                    
                    for batch in range(num_actor_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_actor[start:end] 
                        
                        batch_inds_b, denoising_inds_b = torch.unravel_index(inds_b, (self.n_steps * self.n_envs, _D))
                        
                        obs_b = {"state": obs_k["state"][batch_inds_b]}
                        chains_prev_b = chains_k[batch_inds_b, denoising_inds_b]
                        chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1]
                        returns_b = returns_k[batch_inds_b]
                        values_b = values_k[batch_inds_b]
                        advantages_b = advantages_k[batch_inds_b]
                        logprobs_b = logprobs_k[batch_inds_b, denoising_inds_b]

                        indices_b = indices_k[batch_inds_b, denoising_inds_b]
                        k_b = k_k_floor[batch_inds_b, denoising_inds_b]

                        # Extract the total steps for the items in this batch
                        stp_b = stp_k[batch_inds_b]

                        # get diffusion policy loss
                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                            eta,
                        ) = self.model.loss(
                            obs_b,
                            chains_prev_b,
                            chains_next_b,
                            denoising_inds_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                            reward_horizon=self.reward_horizon,
                            indices_b=indices_b,
                            k_b=k_b,
                            stp_b=stp_b
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.ent_coef
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )
                        clipfracs += [clipfrac]

                        # update policy and critic
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        if self.learn_eta:
                            self.eta_optimizer.zero_grad()
                        loss.backward()
                        if self.itr >= self.n_critic_warmup_itr:
                            if self.max_grad_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.actor_ft.parameters(), self.max_grad_norm
                                )
                            self.actor_optimizer.step()
                            if self.learn_eta and batch % self.eta_update_interval == 0:
                                self.eta_optimizer.step()
                        self.critic_optimizer.step()
                        log.info(
                            f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_actor_batch}"
                        )

                        # Stop gradient update if KL difference reaches target
                        if self.target_kl is not None and approx_kl > self.target_kl:
                            flag_break = True
                            break
                    if flag_break:
                        break

                # Explained variation of future rewards using value function
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )

            # Plot state trajectories (only in D3IL)
            if (
                self.itr % self.render_freq == 0
                and self.n_render > 0
                and self.traj_plotter is not None
            ):
                self.traj_plotter(
                    obs_full_trajs=obs_full_trajs,
                    n_render=self.n_render,
                    max_episode_steps=self.max_episode_steps,
                    render_dir=self.render_dir,
                    itr=self.itr,
                )

            # Update lr, min_sampling_std
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.save_trajs:
                run_results[-1]["obs_full_trajs"] = obs_full_trajs
                run_results[-1]["obs_trajs"] = obs_trajs
                run_results[-1]["chains_trajs"] = chains_trajs
                run_results[-1]["reward_trajs"] = reward_trajs
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=False,
                        )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} | eta {eta:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "loss": loss,
                                "pg loss": pg_loss,
                                "value loss": v_loss,
                                "bc loss": bc_loss,
                                "eta": eta,
                                "approx kl": approx_kl,
                                "ratio": ratio,
                                "clipfrac": np.mean(clipfracs),
                                "explained variance": explained_var,
                                "avg episode reward - train": avg_episode_reward,
                                "num episode - train": num_episode_finished,
                                "diffusion - min sampling std": diffusion_min_sampling_std,
                                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                                "critic lr": self.critic_optimizer.param_groups[0]["lr"],
                                "adaptor loss": np.mean(adaptor_losses) if len(adaptor_losses) > 0 else 0.0,
                                "avg denoising steps": stp_k.mean().item() if 'stp_k' in locals() else self.model.ft_denoising_steps,
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            if not self._in_stage3 and 'stp_k' in locals():
                if stp_k.mean().item() < self.cfg.d3p.zeta2:
                    log.info(f"Entering Stage 3 at itr {self.itr}: avg stp {stp_k.mean().item():.2f} < ζ2 {self.cfg.d3p.zeta2}")
                    self.update_epochs = self.e_slow
                    self._in_stage3 = True
            
            self.itr += 1

    def save_model(self):
        """
        Overrides the parent save_model to also save the D3P adaptor.
        """
        # Let the parent class save the main diffusion model (e.g., state_0.pt)
        super().save_model()
        
        # Define the path for the adaptor
        adaptor_path = os.path.join(self.checkpoint_dir, f"adaptor_{self.itr}.pt")
        
        # Save the adaptor's weights and its optimizer state
        payload = {
            "adaptor": self.adaptor.state_dict(),
            "adaptor_optimizer": self.adaptor_optimizer.state_dict(),
        }
        
        torch.save(payload, adaptor_path)
        log.info(f"Saved D3P Adaptor to {adaptor_path}")