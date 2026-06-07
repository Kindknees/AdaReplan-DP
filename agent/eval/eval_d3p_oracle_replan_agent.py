import os
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.eval.eval_agent import EvalAgent
from d3p_utils.D3PAdaptor import D3PAdaptor


class EvalD3POracleReplanAgent(EvalAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # --- D3P adaptor ---
        d3p_cfg = cfg.get("d3p", {})
        adaptor_mlp_dims = d3p_cfg.get("mlp_dims", [256, 512, 1024, 512, 256])
        stride = cfg.denoising_steps // cfg.ft_denoising_steps
        self.adaptor = D3PAdaptor(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            output_mean=stride,
            seq_len=cfg.cond_steps,
            chunk_size=cfg.horizon_steps,
            mlp_dims=adaptor_mlp_dims,
        ).to(self.device)

        adaptor_path = cfg.get("adaptor_path", None)
        assert adaptor_path is not None and os.path.isfile(adaptor_path), (
            f"adaptor_path must point to a saved adaptor_*.pt; got {adaptor_path}"
        )
        payload = torch.load(adaptor_path, map_location=self.device, weights_only=True)
        self.adaptor.load_state_dict(payload["adaptor"])
        self.adaptor.eval()
        log.info(f"Loaded D3P adaptor from {adaptor_path}")

        # --- Oracle replan config ---
        oracle_cfg = cfg.get("oracle", {})
        self.trigger_on_kick = bool(oracle_cfg.get("trigger_on_kick", True))
        self.trigger_on_action_noise = bool(
            oracle_cfg.get("trigger_on_action_noise", True)
        )
        self.chunk_len = int(oracle_cfg.get("max_chunk_steps", cfg.horizon_steps))
        assert self.chunk_len <= cfg.horizon_steps, (
            "max_chunk_steps cannot exceed horizon_steps (chunk length)"
        )

        warm_cfg = oracle_cfg.get("warm_start", {})
        self.warm_start_enabled = bool(warm_cfg.get("enabled", False))
        self.warm_start_t_frac = float(warm_cfg.get("t_frac", 0.5))
        assert 0.0 <= self.warm_start_t_frac <= 1.0, (
            "oracle.warm_start.t_frac must be in [0, 1]"
        )

        assert cfg.act_steps == 1, (
            "Oracle replan agent requires act_steps=1 so the env advances one "
            f"action per step; got act_steps={cfg.act_steps}"
        )

    def run(self):
        timer = Timer()
        N = self.n_envs
        H = self.chunk_len
        A = self.action_dim

        options_venv = [{} for _ in range(N)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )

        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, N))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, N))

        # Replan bookkeeping
        cursor = np.zeros(N, dtype=np.int64)
        queued_chunks = np.zeros((N, H, A), dtype=np.float32)
        nfe_trajs = np.zeros((self.n_steps, N))
        replan_trajs = np.zeros((self.n_steps, N))           # any replan
        oracle_fire_trajs = np.zeros((self.n_steps, N))      # oracle fired (subset of replan)
        forced_replan_trajs = np.zeros((self.n_steps, N))    # cursor>=H or done (subset of replan)
        kick_trajs = np.zeros((self.n_steps, N), dtype=np.float32)
        noise_trajs = np.zeros((self.n_steps, N), dtype=np.float32)

        # Pending-oracle flag: set when the just-completed step had a
        # disturbance, consumed on the NEXT step's chunk-sample decision.
        # Mirrors the TD trigger's "post-step trigger → next-step replan"
        # timing so chunk indices stay aligned.
        pending_oracle = np.zeros(N, dtype=bool)

        # Warm-start buffers (only used when warm_start_enabled). See
        # EvalD3PReplanAgent for the algorithm; identical here.
        warm_start_chunk = np.zeros((N, H, A), dtype=np.float32)
        pending_warm = np.zeros(N, dtype=bool)
        if self.warm_start_enabled:
            ddim_steps = int(self.model.ddim_steps)
            k_warm = int(round((1.0 - self.warm_start_t_frac) * ddim_steps))
            k_warm = max(0, min(k_warm, ddim_steps - 1))
            t_warm_ddpm = int(self.model.ddim_t[k_warm].item())
            log.info(
                f"[warm-start] enabled, t_frac={self.warm_start_t_frac:.2f} "
                f"-> k_warm={k_warm}/{ddim_steps}, ddpm_t={t_warm_ddpm}"
            )
        else:
            k_warm = 0
            t_warm_ddpm = 0

        if self.save_full_observations:
            obs_full_trajs = np.empty((0, N, self.obs_dim))
            obs_full_trajs = np.vstack(
                (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
            )

        for step in range(self.n_steps):
            if step % 20 == 0:
                print(f"Processed step {step} of {self.n_steps}")

            need_chunk = cursor == 0
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }

            if need_chunk.any():
                warm_start_x = None
                warm_start_indices = None
                if self.warm_start_enabled and pending_warm.any():
                    warm_envs = pending_warm & need_chunk
                    if warm_envs.any():
                        warm_start_x = torch.randn(
                            (N, H, A), device=self.device
                        )
                        warm_start_indices = torch.zeros(
                            N, dtype=torch.long, device=self.device
                        )
                        warm_envs_t = torch.from_numpy(warm_envs).to(self.device)
                        warm_chunks_t = torch.from_numpy(
                            warm_start_chunk[warm_envs]
                        ).float().to(self.device)
                        t_b = torch.full(
                            (int(warm_envs.sum()),),
                            t_warm_ddpm,
                            dtype=torch.long,
                            device=self.device,
                        )
                        noised = self.model.q_sample(warm_chunks_t, t_b)
                        warm_start_x[warm_envs_t] = noised
                        warm_start_indices[warm_envs_t] = k_warm

                with torch.no_grad():
                    samples, _, _, _, _, stp_t = self.model.forward_d3p(
                        cond=cond,
                        adaptor=self.adaptor,
                        deterministic=True,
                        warm_start_x=warm_start_x,
                        warm_start_indices=warm_start_indices,
                    )
                new_chunks = samples.trajectories.cpu().numpy()
                stp_np = stp_t.cpu().numpy()

                if self.warm_start_enabled:
                    pending_warm[need_chunk] = False

                queued_chunks[need_chunk] = new_chunks[need_chunk, :H]
                nfe_trajs[step] = np.where(need_chunk, stp_np, 0.0)
            else:
                nfe_trajs[step] = 0.0
            replan_trajs[step] = need_chunk.astype(np.float32)
            oracle_fire_trajs[step] = (need_chunk & pending_oracle).astype(np.float32)
            forced_replan_trajs[step] = (need_chunk & ~pending_oracle).astype(np.float32)
            pending_oracle[need_chunk] = False

            # ---- Pop one action per env from the queue ----
            action_per_env = queued_chunks[np.arange(N), cursor]
            action_venv = action_per_env[:, None, :]

            # ---- Step env, observe r, s', disturbance flags ----
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            done_venv = terminated_venv | truncated_venv

            def _info_any(d, k):
                if k not in d:
                    return False
                v = d[k]
                if isinstance(v, np.ndarray):
                    return bool(np.any(v))
                return bool(v)
            kick_per_env = np.array(
                [_info_any(info_venv[i], "kick_fired") for i in range(N)],
                dtype=bool,
            )
            noise_per_env = np.array(
                [_info_any(info_venv[i], "action_noise_fired") for i in range(N)],
                dtype=bool,
            )
            kick_trajs[step] = kick_per_env.astype(np.float32)
            noise_trajs[step] = noise_per_env.astype(np.float32)

            # ---- Oracle trigger: did a disturbance just happen? ----
            # only the configured sources count. obs_delay is excluded
            # by construction — see __init__ docstring.
            oracle_fire = np.zeros(N, dtype=bool)
            if self.trigger_on_kick:
                oracle_fire |= kick_per_env
            if self.trigger_on_action_noise:
                oracle_fire |= noise_per_env
            oracle_fire &= ~done_venv

            # ---- Decide chunk-buffer state for next step ----
            actually_changes_action = (
                oracle_fire & ((cursor + 1) < H) & ~done_venv
            )

            # Warm-start: stash old chunk's remaining actions when oracle fires
            # mid-chunk, for next step's warm-started forward_d3p.
            if self.warm_start_enabled:
                for i in np.where(actually_changes_action)[0]:
                    c = int(cursor[i])
                    n_warm = H - c - 1
                    remaining = queued_chunks[i, c + 1 : H]
                    last_action = queued_chunks[i, H - 1]
                    pad = np.broadcast_to(last_action, (H - n_warm, A))
                    warm_start_chunk[i] = np.concatenate([remaining, pad], axis=0)
                pending_warm[actually_changes_action] = True

            pending_oracle[actually_changes_action] = True

            next_cursor = cursor + 1
            forced = (next_cursor >= H) | done_venv | oracle_fire
            next_cursor[forced] = 0
            cursor = next_cursor

            # ---- Standard bookkeeping ----
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = done_venv
            if self.save_full_observations:
                obs_full_venv = np.array(
                    [info["full_obs"]["state"] for info in info_venv]
                )
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                )
            prev_obs_venv = obs_venv

        # ---- Episode aggregation (mirrors EvalDiffusionAgent) ----
        episodes_start_end = []
        for env_ind in range(N):
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
            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [np.max(r) / self.act_steps for r in reward_trajs_split]
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

        avg_nfe = float(nfe_trajs.mean())
        replan_rate = float(replan_trajs.mean())
        oracle_replan_rate = float(oracle_fire_trajs.mean())
        forced_replan_rate = float(forced_replan_trajs.mean())
        kick_rate = float(kick_trajs.mean())
        noise_rate = float(noise_trajs.mean())
        n_oracle_replan = int(oracle_fire_trajs.sum())
        n_kick = int(kick_trajs.sum())
        n_noise = int(noise_trajs.sum())
        nfe_hist, nfe_edges = np.histogram(
            nfe_trajs.ravel(),
            bins=np.arange(0, self.model.ddim_steps + 2) - 0.5,
        )

        time = timer()
        log.info(
            f"eval (oracle): ep {num_episode_finished:4d} | success {success_rate:6.4f} "
            f"| reward {avg_episode_reward:8.4f} (best/step {avg_best_reward:8.4f}) "
            f"| avg NFE {avg_nfe:6.3f}"
        )
        log.info(
            f"  replan rate {replan_rate:5.3f} = oracle {oracle_replan_rate:5.3f} "
            f"+ forced {forced_replan_rate:5.3f} | "
            f"disturbance rate kick {kick_rate:5.3f} noise {noise_rate:5.3f}"
        )
        log.info(
            f"  counts: oracle replans {n_oracle_replan} | kicks {n_kick} | "
            f"action-noise events {n_noise}"
        )

        # Per-episode arrays for downstream significance testing.
        episode_reward_arr = (
            episode_reward if len(episodes_start_end) > 0 else np.zeros(0, dtype=np.float32)
        )
        episode_best_reward_arr = (
            episode_best_reward if len(episodes_start_end) > 0 else np.zeros(0, dtype=np.float32)
        )
        episode_env_ind_arr = np.array(
            [e[0] for e in episodes_start_end], dtype=np.int64
        )
        episode_start_step_arr = np.array(
            [e[1] for e in episodes_start_end], dtype=np.int64
        )

        np.savez(
            self.result_path,
            num_episode=num_episode_finished,
            eval_success_rate=success_rate,
            eval_episode_reward=avg_episode_reward,
            eval_best_reward=avg_best_reward,
            # Per-episode raw data for downstream statistical tests.
            episode_reward=episode_reward_arr,
            episode_best_reward=episode_best_reward_arr,
            episode_env_ind=episode_env_ind_arr,
            episode_start_step=episode_start_step_arr,
            best_reward_threshold_for_success=float(self.best_reward_threshold_for_success),
            avg_nfe=avg_nfe,
            replan_rate=replan_rate,
            oracle_replan_rate=oracle_replan_rate,
            forced_replan_rate=forced_replan_rate,
            n_oracle_replan=n_oracle_replan,
            n_kick=n_kick,
            n_noise=n_noise,
            kick_trajs=kick_trajs,
            noise_trajs=noise_trajs,
            oracle_fire_trajs=oracle_fire_trajs,
            forced_replan_trajs=forced_replan_trajs,
            replan_trajs=replan_trajs.astype(np.float32),
            nfe_hist=nfe_hist,
            nfe_edges=nfe_edges,
            time=time,
        )
