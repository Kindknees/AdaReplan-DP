"""
Evaluate a trained RL replan controller (PPO trigger) on top of a frozen D3P policy.

"""

import os
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.eval.eval_agent import EvalAgent
from d3p_utils.D3PAdaptor import D3PAdaptor
from d3p_utils.ReplanController import ReplanController


class EvalReplanControllerAgent(EvalAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # ---- D3P adaptor (frozen) ----
        d3p_cfg = cfg.get("d3p", {})
        adaptor_mlp_dims = d3p_cfg.get("mlp_dims", [128, 128])
        stride = cfg.denoising_steps // cfg.ft_denoising_steps
        self.adaptor = D3PAdaptor(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            output_mean=stride,
            seq_len=cfg.cond_steps,
            chunk_size=cfg.horizon_steps,
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
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # ---- Replan controller (frozen, loaded from training run) ----
        self.chunk_len = int(cfg.controller.get("max_chunk_steps", cfg.horizon_steps))
        assert self.chunk_len <= cfg.horizon_steps
        self.feature_dim = self.obs_dim * self.n_cond_step + self.action_dim + 4
        self.controller = ReplanController(
            feature_dim=self.feature_dim,
            mlp_dims=cfg.controller.get("mlp_dims", [128, 128]),
            init_replan_bias=cfg.controller.get("init_replan_bias", -1.0),
        ).to(self.device)

        controller_path = cfg.controller_path
        assert controller_path is not None and os.path.isfile(controller_path), (
            f"controller_path must point to a saved controller_*.pt; got {controller_path}"
        )
        ctrl_payload = torch.load(
            controller_path, map_location=self.device, weights_only=True
        )
        self.controller.load_state_dict(ctrl_payload["controller"])
        self.controller.eval()
        for p in self.controller.parameters():
            p.requires_grad = False
        log.info(f"Loaded replan controller from {controller_path}")

        # ---- Eval-time decision mode ----
        # "deterministic" (default): replan iff Bernoulli prob > threshold (default 0.5).
        # "stochastic": sample u ~ Bernoulli; useful only for measuring decision noise.
        self.decision_mode = cfg.controller.get("decision_mode", "deterministic")
        self.decision_threshold = float(
            cfg.controller.get("decision_threshold", 0.5)
        )
        self.gamma = float(cfg.controller.get("gamma", 0.999))  # for delta feature only

        assert cfg.act_steps == 1, (
            "Replan eval requires act_steps=1 so the wrapper advances one action "
            f"per step; got act_steps={cfg.act_steps}"
        )

    # ------------------------------------------------------------------ #
    def _critic_value(self, obs_np):
        cond = {"state": torch.from_numpy(obs_np).float().to(self.device)}
        with torch.no_grad():
            return self.model.critic(cond).cpu().numpy().flatten()

    def _build_feature(self, obs_np, next_action, v_t, delta_prev, cursor, forced):
        N = obs_np.shape[0]
        s_flat = obs_np.reshape(N, -1)
        na = next_action.copy()
        na[forced] = 0.0
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
        idx = np.where(replan_mask)[0]
        if len(idx) == 0:
            return None, None, idx
        cond = {"state": torch.from_numpy(obs_np[idx]).float().to(self.device)}
        with torch.no_grad():
            if self.use_dynamic_nfe:
                samples, _, _, _, _, stp_t = self.model.forward_d3p(
                    cond=cond, adaptor=self.adaptor, deterministic=True
                )
                chunks = samples.trajectories.cpu().numpy()
                nfe = stp_t.cpu().numpy()
            else:
                samples = self.model.forward(cond=cond, deterministic=True)
                chunks = samples.trajectories.cpu().numpy()
                nfe = np.full(len(idx), float(self.model.ft_denoising_steps))
        return chunks[:, : self.chunk_len], nfe, idx

    # ------------------------------------------------------------------ #
    def run(self):
        timer = Timer()
        N, H, A = self.n_envs, self.chunk_len, self.action_dim

        options_venv = [{} for _ in range(N)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )

        firsts_trajs = np.zeros((self.n_steps + 1, N))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, N))

        cursor = np.zeros(N, dtype=np.int64)
        queued_chunks = np.zeros((N, H, A), dtype=np.float32)
        must_replan = np.ones(N, dtype=bool)
        delta_prev = np.zeros(N, dtype=np.float32)

        nfe_trajs = np.zeros((self.n_steps, N))
        replan_trajs = np.zeros((self.n_steps, N))      # 1 if env i resampled
        trigger_trajs = np.zeros((self.n_steps, N))     # 1 if controller chose replan (not forced)
        prob_trajs = np.zeros((self.n_steps, N))        # Bernoulli probability for diagnostics

        if self.save_full_observations:
            obs_full_trajs = np.empty((0, N, self.obs_dim))
            obs_full_trajs = np.vstack(
                (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
            )

        for step in range(self.n_steps):
            if step % 20 == 0:
                print(f"Processed step {step} of {self.n_steps}")

            obs_np = prev_obs_venv["state"]
            v_t = self._critic_value(obs_np)

            safe_cursor = np.clip(cursor, 0, H - 1)
            next_action = queued_chunks[np.arange(N), safe_cursor]
            forced = must_replan.copy()

            feat = self._build_feature(
                obs_np, next_action, v_t, delta_prev, cursor, forced
            )
            with torch.no_grad():
                dist, _ = self.controller(feat)
                probs = dist.probs.cpu().numpy()
                if self.decision_mode == "stochastic":
                    u = dist.sample().cpu().numpy().astype(bool)
                else:
                    u = probs > self.decision_threshold

            prob_trajs[step] = probs
            voluntary = (~forced) & u
            trigger_trajs[step] = voluntary.astype(np.float32)
            replan = forced | u                          # forced always replans

            chunks, nfe_sub, idx = self._sample_chunks(obs_np, replan)
            nfe_step = np.zeros(N, dtype=np.float32)
            if chunks is not None:
                queued_chunks[idx] = chunks
                cursor[idx] = 0
                nfe_step[idx] = nfe_sub
            replan_trajs[step] = replan.astype(np.float32)
            nfe_trajs[step] = nfe_step

            action_per_env = queued_chunks[np.arange(N), cursor]
            action_venv = action_per_env[:, None, :]
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            done_venv = terminated_venv | truncated_venv

            v_tp1 = self._critic_value(obs_venv["state"])
            nonterm = (~terminated_venv).astype(np.float32)
            delta = reward_venv + self.gamma * v_tp1 * nonterm - v_t

            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = done_venv
            if self.save_full_observations:
                obs_full_venv = np.array(
                    [info["full_obs"]["state"] for info in info_venv]
                )
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                )

            cursor = cursor + 1
            must_replan = (cursor >= H) | done_venv
            delta_prev = np.where(done_venv, 0.0, delta).astype(np.float32)
            cursor = np.where(done_venv, 0, cursor)
            prev_obs_venv = obs_venv

        # ---- Episode aggregation (mirrors eval_d3p_replan_agent) ----
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
            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [np.max(r) / self.act_steps for r in reward_split]
                )
            avg_episode_reward = float(np.mean(episode_reward))
            avg_best_reward = float(np.mean(episode_best_reward))
            success_rate = float(
                np.mean(episode_best_reward >= self.best_reward_threshold_for_success)
            )
        else:
            num_episode_finished = 0
            avg_episode_reward = 0.0
            avg_best_reward = 0.0
            success_rate = 0.0
            log.info("[WARNING] No episode completed within the iteration!")

        if self.traj_plotter is not None:
            self.traj_plotter(
                obs_full_trajs=obs_full_trajs,
                n_render=self.n_render,
                max_episode_steps=self.max_episode_steps,
                render_dir=self.render_dir,
                itr=0,
            )

        avg_nfe = float(nfe_trajs.mean())                 # mean NFE per env-step
        replan_rate = float(replan_trajs.mean())          # any replan (forced or voluntary)
        trigger_rate = float(trigger_trajs.mean())        # voluntary replans only
        avg_prob = float(prob_trajs.mean())
        nfe_hist, nfe_edges = np.histogram(
            nfe_trajs.ravel(),
            bins=np.arange(0, self.model.ddim_steps + 2) - 0.5,
        )

        time = timer()
        log.info(
            f"eval: ep {num_episode_finished:4d} | success {success_rate:6.4f} "
            f"| reward {avg_episode_reward:8.4f} (best/step {avg_best_reward:8.4f}) "
            f"| avg NFE {avg_nfe:6.3f} | replan {replan_rate:5.3f} "
            f"| trigger {trigger_rate:5.3f} | avg p(replan) {avg_prob:.3f}"
        )
        np.savez(
            self.result_path,
            num_episode=num_episode_finished,
            eval_success_rate=success_rate,
            eval_episode_reward=avg_episode_reward,
            eval_best_reward=avg_best_reward,
            avg_nfe=avg_nfe,
            replan_rate=replan_rate,
            trigger_rate=trigger_rate,
            avg_prob=avg_prob,
            nfe_hist=nfe_hist,
            nfe_edges=nfe_edges,
            decision_mode=self.decision_mode,
            decision_threshold=self.decision_threshold,
            time=time,
        )
