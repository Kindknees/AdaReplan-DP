"""
Train RL replan controller (PPO trigger) on top of a frozen D3P policy.

"""

import os
import pickle
import numpy as np
import torch
import logging
import wandb

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_agent import TrainPPOAgent
from d3p_utils.D3PAdaptor import D3PAdaptor
from d3p_utils.ReplanController import ReplanController


class TrainReplanControllerAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        assert cfg.act_steps == 1, (
            "Replan controller needs act_steps=1 so the wrapper advances one "
            f"action per step; got act_steps={cfg.act_steps}"
        )

        # ---- Freeze the entire diffusion policy + critic ----
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # ---- Load the (frozen) D3P adaptor ----
        d3p_cfg = cfg.get("d3p", {})
        adaptor_mlp_dims = d3p_cfg.get("mlp_dims", [128, 128])
        stride = cfg.denoising_steps // cfg.ft_denoising_steps
        self.adaptor = D3PAdaptor(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            output_mean=stride,
            seq_len=cfg.cond_steps,
            chunk_size=self.horizon_steps,
            mlp_dims=adaptor_mlp_dims,
        ).to(self.device)

        self.use_dynamic_nfe = cfg.controller.get("use_dynamic_nfe", True)
        adaptor_path = cfg.get("adaptor_path", None)
        if self.use_dynamic_nfe:
            assert adaptor_path is not None and os.path.isfile(adaptor_path), (
                f"use_dynamic_nfe=True needs a valid adaptor_path; got {adaptor_path}"
            )
            payload = torch.load(
                adaptor_path, map_location=self.device, weights_only=True
            )
            self.adaptor.load_state_dict(payload["adaptor"])
            log.info(f"Loaded D3P adaptor from {adaptor_path}")
        self.adaptor.eval()
        for p in self.adaptor.parameters():
            p.requires_grad = False

        # ---- The only thing we train: the replan controller ----
        self.chunk_len = int(cfg.controller.get("max_chunk_steps", self.horizon_steps))
        assert self.chunk_len <= self.horizon_steps
        # feature = obs (To*Do) + next_action (Da) + [V, delta, cursor_frac, remain_frac]
        self.feature_dim = self.obs_dim * self.n_cond_step + self.action_dim + 4
        self.controller = ReplanController(
            feature_dim=self.feature_dim,
            mlp_dims=cfg.controller.get("mlp_dims", [128, 128]),
            init_replan_bias=cfg.controller.get("init_replan_bias", -1.0),
        ).to(self.device)
        self.controller_optimizer = torch.optim.Adam(
            self.controller.parameters(),
            lr=cfg.controller.get("lr", 3e-4),
        )

        # ---- Controller PPO / reward hyperparameters ----
        self.ctrl_gamma = cfg.controller.get("gamma", self.gamma)
        self.ctrl_gae_lambda = cfg.controller.get("gae_lambda", self.gae_lambda)
        self.ctrl_clip = cfg.controller.get("clip", 0.2)
        # Entropy bonus with linear annealing. The instantaneous coefficient
        # is `ent_coef` at itr 0 and `ent_coef_final` at `ent_anneal_end_itr`
        # (and beyond). Set ent_coef == ent_coef_final to disable annealing.
        self.ctrl_ent_coef_init = float(cfg.controller.get("ent_coef", 0.01))
        self.ctrl_ent_coef_final = float(
            cfg.controller.get("ent_coef_final", self.ctrl_ent_coef_init)
        )
        self.ctrl_ent_anneal_end_itr = int(
            cfg.controller.get("ent_anneal_end_itr", self.n_train_itr)
        )
        self.ctrl_vf_coef = cfg.controller.get("vf_coef", 0.5)
        self.ctrl_max_grad_norm = cfg.controller.get("max_grad_norm", 10.0)
        self.replan_cost = cfg.controller.get("replan_cost", 0.0)  # lambda
        self.fixed_nfe = self.model.ft_denoising_steps

    def _current_ent_coef(self) -> float:
        """Linearly interpolate ent_coef from init -> final over training."""
        if self.ctrl_ent_anneal_end_itr <= 0:
            return self.ctrl_ent_coef_final
        frac = min(1.0, self.itr / float(self.ctrl_ent_anneal_end_itr))
        return (
            self.ctrl_ent_coef_init
            + frac * (self.ctrl_ent_coef_final - self.ctrl_ent_coef_init)
        )

    def _critic_value(self, obs_np):
        """V_Theta(s) from the frozen diffusion critic. obs_np: (N, To, Do)."""
        cond = {"state": torch.from_numpy(obs_np).float().to(self.device)}
        with torch.no_grad():
            return self.model.critic(cond).cpu().numpy().flatten()

    def _build_feature(self, obs_np, next_action, v_t, delta_prev, cursor, forced):
        """Assemble the controller feature tensor. All inputs are numpy (N, ...)."""
        N = obs_np.shape[0]
        s_flat = obs_np.reshape(N, -1)                       # (N, To*Do)
        na = next_action.copy()
        na[forced] = 0.0                                     # no valid queued action
        cursor_frac = (cursor / self.chunk_len).reshape(N, 1)
        remain_frac = ((self.chunk_len - cursor) / self.chunk_len).reshape(N, 1)
        feat = np.concatenate(
            [
                s_flat,
                na,
                v_t.reshape(N, 1),
                delta_prev.reshape(N, 1),
                cursor_frac,
                remain_frac,
            ],
            axis=1,
        ).astype(np.float32)
        return torch.from_numpy(feat).to(self.device)

    def _sample_chunks(self, obs_np, replan_mask):
        """
        Resample fresh action chunks for the envs in replan_mask.
        Returns (chunks (n_sub, H, A), nfe (n_sub,)) for those envs only.
        """
        idx = np.where(replan_mask)[0]
        if len(idx) == 0:
            return None, None, idx
        cond = {
            "state": torch.from_numpy(obs_np[idx]).float().to(self.device)
        }
        with torch.no_grad():
            if self.use_dynamic_nfe:
                samples, _, _, _, _, stp_t = self.model.forward_d3p(
                    cond=cond, adaptor=self.adaptor, deterministic=True
                )
                chunks = samples.trajectories.cpu().numpy()  # (n_sub, horizon, A)
                nfe = stp_t.cpu().numpy()
            else:
                samples = self.model.forward(cond=cond, deterministic=True)
                chunks = samples.trajectories.cpu().numpy()
                nfe = np.full(len(idx), float(self.fixed_nfe))
        return chunks[:, : self.chunk_len], nfe, idx

    def run(self):
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        N, H, A = self.n_envs, self.chunk_len, self.action_dim

        while self.itr < self.n_train_itr:
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.controller.eval() if eval_mode else self.controller.train()

            # ---- rollout buffers ----
            feats_trajs = np.zeros((self.n_steps, N, self.feature_dim), dtype=np.float32)
            u_trajs = np.zeros((self.n_steps, N), dtype=np.float32)
            logp_trajs = np.zeros((self.n_steps, N), dtype=np.float32)
            val_trajs = np.zeros((self.n_steps, N), dtype=np.float32)
            ctrl_reward_trajs = np.zeros((self.n_steps, N), dtype=np.float32)
            dmask_trajs = np.zeros((self.n_steps, N), dtype=np.float32)
            reward_trajs = np.zeros((self.n_steps, N), dtype=np.float32)
            terminated_trajs = np.zeros((self.n_steps, N), dtype=np.float32)
            nfe_trajs = np.zeros((self.n_steps, N), dtype=np.float32)
            replan_trajs = np.zeros((self.n_steps, N), dtype=np.float32)
            firsts_trajs = np.zeros((self.n_steps + 1, N), dtype=np.float32)

            prev_obs_venv = self.reset_env_all(options_venv=[{} for _ in range(N)])
            firsts_trajs[0] = 1

            # replan bookkeeping
            cursor = np.zeros(N, dtype=np.int64)
            queued_chunks = np.zeros((N, H, A), dtype=np.float32)
            must_replan = np.ones(N, dtype=bool)   # forced at episode start
            delta_prev = np.zeros(N, dtype=np.float32)

            for step in range(self.n_steps):
                if step % 50 == 0:
                    print(f"[itr {self.itr}] step {step}/{self.n_steps}")

                obs_np = prev_obs_venv["state"]                # (N, To, Do)
                v_t = self._critic_value(obs_np)               # (N,)

                # candidate "continue" action (masked out where forced)
                safe_cursor = np.clip(cursor, 0, H - 1)
                next_action = queued_chunks[np.arange(N), safe_cursor]  # (N, A)

                forced = must_replan.copy()
                feat = self._build_feature(
                    obs_np, next_action, v_t, delta_prev, cursor, forced
                )

                with torch.no_grad():
                    dist, value = self.controller(feat)
                    u = dist.sample() if not eval_mode else (dist.probs > 0.5).float()
                    logp = dist.log_prob(u)

                u_np = u.cpu().numpy().astype(bool)
                replan = forced | u_np                         # forced always replans

                # resample fresh chunks for replanning envs
                chunks, nfe_sub, idx = self._sample_chunks(obs_np, replan)
                nfe_step = np.zeros(N, dtype=np.float32)
                if chunks is not None:
                    queued_chunks[idx] = chunks
                    cursor[idx] = 0
                    nfe_step[idx] = nfe_sub
                replan_trajs[step] = replan.astype(np.float32)
                nfe_trajs[step] = nfe_step

                # execute one action
                action_per_env = queued_chunks[np.arange(N), cursor]   # (N, A)
                action_venv = action_per_env[:, None, :]               # (N, 1, A)
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv

                # next-state value + TD residual (feature for the NEXT step)
                v_tp1 = self._critic_value(obs_venv["state"])
                nonterminal = (~terminated_venv).astype(np.float32)
                delta = reward_venv + self.ctrl_gamma * v_tp1 * nonterminal - v_t

                # controller reward: env reward minus compute cost
                ctrl_reward = reward_venv - self.replan_cost * nfe_step

                # store transition
                feats_trajs[step] = feat.cpu().numpy()
                u_trajs[step] = u_np.astype(np.float32)
                logp_trajs[step] = logp.cpu().numpy()
                val_trajs[step] = value.cpu().numpy()
                ctrl_reward_trajs[step] = ctrl_reward
                dmask_trajs[step] = (~forced).astype(np.float32)   # credit only free choices
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv.astype(np.float32)
                firsts_trajs[step + 1] = done_venv

                # advance cursor / decide next forced replan
                cursor = cursor + 1
                must_replan = (cursor >= H) | done_venv
                delta_prev = np.where(done_venv, 0.0, delta).astype(np.float32)
                cursor = np.where(done_venv, 0, cursor)

                prev_obs_venv = obs_venv
                cnt_train_step += N if not eval_mode else 0

            # ---- episode aggregation (mirrors D3P agent) ----
            episodes_start_end = []
            for env_ind in range(N):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start, end = env_steps[i], env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_split = [
                    reward_trajs[s : e + 1, ei] for ei, s, e in episodes_start_end
                ]
                num_episode_finished = len(reward_split)
                episode_reward = np.array([np.sum(r) for r in reward_split])
                episode_best_reward = np.array(
                    [np.max(r) / self.act_steps for r in reward_split]
                )
                avg_episode_reward = float(np.mean(episode_reward))
                avg_best_reward = float(np.mean(episode_best_reward))
                success_rate = float(
                    np.mean(
                        episode_best_reward >= self.best_reward_threshold_for_success
                    )
                )
            else:
                num_episode_finished = 0
                avg_episode_reward = avg_best_reward = success_rate = 0.0
                log.info("[WARNING] No episode completed within the iteration!")

            # ---- controller PPO update ----
            ctrl_losses, pg_losses, v_losses, ent_losses = [], [], [], []
            if not eval_mode:
                # bootstrap value at the final step
                obs_np = prev_obs_venv["state"]
                v_t = self._critic_value(obs_np)
                safe_cursor = np.clip(cursor, 0, H - 1)
                next_action = queued_chunks[np.arange(N), safe_cursor]
                final_feat = self._build_feature(
                    obs_np, next_action, v_t, delta_prev, cursor, must_replan
                )
                with torch.no_grad():
                    _, final_value = self.controller(final_feat)
                final_value = final_value.cpu().numpy()

                # GAE over the controller reward stream
                advantages = np.zeros_like(ctrl_reward_trajs)
                lastgae = np.zeros(N, dtype=np.float32)
                for t in reversed(range(self.n_steps)):
                    nextval = (
                        final_value if t == self.n_steps - 1 else val_trajs[t + 1]
                    )
                    nonterm = 1.0 - terminated_trajs[t]
                    delta = (
                        ctrl_reward_trajs[t]
                        + self.ctrl_gamma * nextval * nonterm
                        - val_trajs[t]
                    )
                    lastgae = (
                        delta
                        + self.ctrl_gamma * self.ctrl_gae_lambda * nonterm * lastgae
                    )
                    advantages[t] = lastgae
                returns = advantages + val_trajs

                # flatten and keep only free-choice decisions
                feats_f = torch.from_numpy(feats_trajs.reshape(-1, self.feature_dim)).to(self.device)
                u_f = torch.from_numpy(u_trajs.reshape(-1)).to(self.device)
                logp_f = torch.from_numpy(logp_trajs.reshape(-1)).to(self.device)
                adv_f = torch.from_numpy(advantages.reshape(-1)).to(self.device)
                ret_f = torch.from_numpy(returns.reshape(-1)).to(self.device)
                mask_f = torch.from_numpy(dmask_trajs.reshape(-1)).to(self.device)

                valid_inds = torch.where(mask_f > 0)[0]
                # normalize advantages over the valid set
                if len(valid_inds) > 1:
                    adv_valid = adv_f[valid_inds]
                    adv_f = adv_f.clone()
                    adv_f[valid_inds] = (adv_valid - adv_valid.mean()) / (
                        adv_valid.std() + 1e-8
                    )

                ent_coef_now = self._current_ent_coef()
                for _ in range(self.update_epochs):
                    perm = valid_inds[torch.randperm(len(valid_inds), device=self.device)]
                    n_batch = max(1, len(perm) // self.batch_size)
                    for b in range(n_batch):
                        idx = perm[b * self.batch_size : (b + 1) * self.batch_size]
                        dist, value = self.controller(feats_f[idx])
                        new_logp = dist.log_prob(u_f[idx])
                        ratio = torch.exp(new_logp - logp_f[idx])

                        a = adv_f[idx]
                        surr1 = ratio * a
                        surr2 = torch.clamp(
                            ratio, 1 - self.ctrl_clip, 1 + self.ctrl_clip
                        ) * a
                        pg_loss = -torch.min(surr1, surr2).mean()
                        v_loss = 0.5 * ((value - ret_f[idx]) ** 2).mean()
                        ent = dist.entropy().mean()
                        loss = (
                            pg_loss
                            + self.ctrl_vf_coef * v_loss
                            - ent_coef_now * ent
                        )

                        self.controller_optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.controller.parameters(), self.ctrl_max_grad_norm
                        )
                        self.controller_optimizer.step()
                        ctrl_losses.append(loss.item())
                        pg_losses.append(pg_loss.item())
                        v_losses.append(v_loss.item())
                        ent_losses.append(ent.item())

            # ---- logging ----
            avg_nfe = float(nfe_trajs.mean())
            replan_rate = float(replan_trajs.mean())
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            run_results.append({"itr": self.itr, "step": cnt_train_step})
            if self.itr % self.log_freq == 0:
                t = timer()
                if eval_mode:
                    log.info(
                        f"eval {self.itr}: success {success_rate:.4f} | reward "
                        f"{avg_episode_reward:.2f} | avg NFE {avg_nfe:.3f} | "
                        f"replan {replan_rate:.3f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg NFE - eval": avg_nfe,
                                "replan rate - eval": replan_rate,
                            },
                            step=self.itr,
                            commit=True,
                        )
                else:
                    log.info(
                        f"train {self.itr}: ctrl_loss "
                        f"{np.mean(ctrl_losses) if ctrl_losses else 0:.4f} | reward "
                        f"{avg_episode_reward:.2f} | avg NFE {avg_nfe:.3f} | "
                        f"replan {replan_rate:.3f} | t {t:.1f}s"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "controller loss": np.mean(ctrl_losses) if ctrl_losses else 0,
                                "controller pg loss": np.mean(pg_losses) if pg_losses else 0,
                                "controller v loss": np.mean(v_losses) if v_losses else 0,
                                "controller entropy": np.mean(ent_losses) if ent_losses else 0,
                                "avg episode reward - train": avg_episode_reward,
                                "avg NFE - train": avg_nfe,
                                "replan rate - train": replan_rate,
                                "num episode - train": num_episode_finished,
                            },
                            step=self.itr,
                            commit=True,
                        )
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)

            self.itr += 1

    def save_model(self):
        """Save only the controller (base policy + adaptor are frozen on disk)."""
        path = os.path.join(self.checkpoint_dir, f"controller_{self.itr}.pt")
        torch.save(
            {
                "controller": self.controller.state_dict(),
                "controller_optimizer": self.controller_optimizer.state_dict(),
            },
            path,
        )
        log.info(f"Saved replan controller to {path}")
