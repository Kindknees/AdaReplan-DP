import os
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.eval.eval_agent import EvalAgent
from d3p_utils.D3PAdaptor import D3PAdaptor


class EvalD3PReplanAgent(EvalAgent):

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

        # --- Replan trigger config ---
        replan_cfg = cfg.replan
        self.tau = float(replan_cfg.tau)                  # z-score threshold (downside)
        self.ema_alpha = float(replan_cfg.ema_alpha)      # EMA decay
        self.warmup_steps = int(replan_cfg.warmup_steps)  # samples before trigger is allowed
        self.gamma = float(replan_cfg.gamma)              # discount used in TD residual
        self.chunk_len = int(replan_cfg.get("max_chunk_steps", cfg.horizon_steps))
        assert self.chunk_len <= cfg.horizon_steps, (
            "max_chunk_steps cannot exceed horizon_steps (chunk length)"
        )

        self.diagnose_replan_l1 = bool(
            replan_cfg.get("diagnose_replan_l1", False)
        )

        warm_cfg = replan_cfg.get("warm_start", {})
        self.warm_start_enabled = bool(warm_cfg.get("enabled", False))

        self.warm_start_t_frac = float(warm_cfg.get("t_frac", 0.5))
        assert 0.0 <= self.warm_start_t_frac <= 1.0, (
            "replan.warm_start.t_frac must be in [0, 1]"
        )

        assert cfg.act_steps == 1, (
            "Replan agent requires act_steps=1 at the wrapper level so the env "
            "advances one action per step; got act_steps="
            f"{cfg.act_steps}"
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
        replan_trajs = np.zeros((self.n_steps, N))   # 1 if env i resampled this step
        trigger_trajs = np.zeros((self.n_steps, N))  # 1 if TD trigger fired (subset of replan)
        delta_log = np.zeros((self.n_steps, N))      # for diagnostics / histograms
        kick_trajs = np.zeros((self.n_steps, N), dtype=np.float32)
        noise_trajs = np.zeros((self.n_steps, N), dtype=np.float32)

        ema_mu = 0.0
        ema_var = 1.0
        n_seen = 0

        counterfactual_action = np.zeros((N, A), dtype=np.float32)
        pending_compare = np.zeros(N, dtype=bool)
        l1_replan_log = []
        l1_perdim_log = []
        # Track ALL executed actions so we can compute the empirical per-dim std
        # of the policy's output distribution, and use it to normalize the L1.
        executed_actions_log = []

        warm_start_chunk = np.zeros((N, H, A), dtype=np.float32)
        pending_warm = np.zeros(N, dtype=bool)
        # Pre-compute warm-start DDIM index and DDPM timestep (constant across run).
        # k_warm = how many denoising steps to *skip*; larger = closer to old chunk.
        if self.warm_start_enabled:
            ddim_steps = int(self.model.ddim_steps)
            k_warm = int(round((1.0 - self.warm_start_t_frac) * ddim_steps))
            k_warm = max(0, min(k_warm, ddim_steps - 1))   # keep at least 1 denoising step
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

            # ---- Sample fresh chunks ONLY for envs whose cursor is at 0 ----
  
            need_chunk = cursor == 0
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                v_t = self.model.critic(cond).cpu().numpy().flatten()

            if need_chunk.any():
                warm_start_x = None
                warm_start_indices = None
                if self.warm_start_enabled and pending_warm.any():
                    warm_envs = pending_warm & need_chunk  # subset that triggered
                    if warm_envs.any():
                        # Default everyone to pure noise / index 0, then override
                        # the warm subset with q_sample'd old chunks at k_warm.
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

                if self.diagnose_replan_l1 and pending_compare.any():
                    compare_envs = pending_compare & need_chunk
                    for i in np.where(compare_envs)[0]:
                        diff = np.abs(
                            new_chunks[i, 0] - counterfactual_action[i]
                        )
                        l1_replan_log.append(float(diff.sum()))
                        l1_perdim_log.append(diff.astype(np.float32).copy())
                    pending_compare[need_chunk] = False

                queued_chunks[need_chunk] = new_chunks[need_chunk, :H]
                nfe_trajs[step] = np.where(need_chunk, stp_np, 0.0)
            else:
                nfe_trajs[step] = 0.0
            replan_trajs[step] = need_chunk.astype(np.float32)

            action_per_env = queued_chunks[np.arange(N), cursor]
            action_venv = action_per_env[:, None, :]
            if self.diagnose_replan_l1:
                executed_actions_log.append(action_per_env.copy())

            # ---- Step env, observe r, s' ----
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            done_venv = terminated_venv | truncated_venv

            # Capture per-env disturbance flags from wrappers (set in
            # ObjectKickWrapper / ActionNoiseWrapper). MultiStep aggregates info
            # over inner sub-steps; we treat "any sub-step fired" as True.
            def _info_any(d, k):
                if k not in d:
                    return False
                v = d[k]
                if isinstance(v, np.ndarray):
                    return bool(np.any(v))
                return bool(v)
            kick_trajs[step] = np.array(
                [_info_any(info_venv[i], "kick_fired") for i in range(N)],
                dtype=np.float32,
            )
            noise_trajs[step] = np.array(
                [_info_any(info_venv[i], "action_noise_fired") for i in range(N)],
                dtype=np.float32,
            )

            # ---- V(s_{t+1}) and TD residual ----
            with torch.no_grad():
                cond_next = {
                    "state": torch.from_numpy(obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                v_tp1 = self.model.critic(cond_next).cpu().numpy().flatten()

            nonterminal = (~done_venv).astype(np.float32)
            delta = reward_venv + self.gamma * v_tp1 * nonterminal - v_t
            delta_log[step] = delta

            # ---- Score against OLD pooled EMA, then update ----
            sigma = max(np.sqrt(ema_var), 1e-6)
            z = (delta - ema_mu) / sigma
            valid = ~done_venv
            trigger = (
                (z < -self.tau)
                & valid
                & (n_seen >= self.warmup_steps)
            )
            trigger_trajs[step] = trigger.astype(np.float32)

            # Sequentially update the pooled EMA with each valid delta this step
            for d in delta[valid]:
                ema_mu = (1.0 - self.ema_alpha) * ema_mu + self.ema_alpha * d
                ema_var = (
                    (1.0 - self.ema_alpha) * ema_var
                    + self.ema_alpha * (d - ema_mu) ** 2
                )
                n_seen += 1

            # ---- Update cursor for next env-step ----
            # Compute "trigger that actually changes behavior": fires AND chunk
            # still has remaining actions AND env didn't terminate.
            actually_changes_action = (
                trigger & ((cursor + 1) < H) & ~done_venv
            )

            # Diagnostic: stash the would-have-been action for L1 comparison.
            if self.diagnose_replan_l1:
                for i in np.where(actually_changes_action)[0]:
                    counterfactual_action[i] = queued_chunks[i, cursor[i] + 1].copy()
                pending_compare[actually_changes_action] = True

            if self.warm_start_enabled:
                for i in np.where(actually_changes_action)[0]:
                    c = int(cursor[i])
                    n_warm = H - c - 1  
                    remaining = queued_chunks[i, c + 1 : H]      
                    last_action = queued_chunks[i, H - 1]  
                    pad = np.broadcast_to(last_action, (H - n_warm, A))
                    warm_start_chunk[i] = np.concatenate([remaining, pad], axis=0)
                pending_warm[actually_changes_action] = True

            next_cursor = cursor + 1
            forced = (next_cursor >= H) | done_venv | trigger
            next_cursor[forced] = 0
            cursor = next_cursor

            # ---- Standard rollout bookkeeping ----
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
                # act_steps == 1 here so divisor is 1 (kept for parity).
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
        trigger_rate = float(trigger_trajs.mean())
        nfe_hist, nfe_edges = np.histogram(
            nfe_trajs.ravel(),
            bins=np.arange(0, self.model.ddim_steps + 2) - 0.5,
        )
        delta_hist, delta_edges = np.histogram(delta_log.ravel(), bins=51)

        # Diagnostic: L1 between replanned action and would-have-been-queued action.
        if self.diagnose_replan_l1 and len(l1_replan_log) > 0:
            l1_arr = np.asarray(l1_replan_log)
            l1_mean = float(l1_arr.mean())
            l1_median = float(np.median(l1_arr))
            l1_std = float(l1_arr.std())
            l1_p10 = float(np.percentile(l1_arr, 10))
            l1_p90 = float(np.percentile(l1_arr, 90))
            l1_hist, l1_edges = np.histogram(l1_arr, bins=40)
            perdim_arr = np.stack(l1_perdim_log, axis=0)              
            perdim_mean = perdim_arr.mean(axis=0)                   
            arm_l1_mean = float(perdim_arr[:, :-1].sum(axis=1).mean())
            gripper_l1_mean = float(perdim_arr[:, -1].mean())    
            log.info(
                f"replan L1 diag: n={len(l1_arr):5d} | "
                f"mean {l1_mean:.4f} | median {l1_median:.4f} | std {l1_std:.4f} | "
                f"p10 {l1_p10:.4f} | p90 {l1_p90:.4f}"
            )
            log.info(
                f"replan L1 split: arm (sum dim 0..{A-2}) {arm_l1_mean:.4f} | "
                f"gripper (dim {A-1}) {gripper_l1_mean:.4f} | "
                f"per-dim mean " + np.array2string(
                    perdim_mean, precision=4, separator=", "
                )
            )

            executed_all = np.concatenate(executed_actions_log, axis=0)  # (T*N, A)
            action_std = executed_all.std(axis=0) + 1e-6                 # (A,)
            perdim_arr_std = perdim_arr / action_std                     # (n, A)
            l1_std_norm = perdim_arr_std.sum(axis=1)                     # (n,)
            l1_std_norm_mean = float(l1_std_norm.mean())
            arm_l1_std_mean = float(perdim_arr_std[:, :-1].sum(axis=1).mean())
            gripper_l1_std_mean = float(perdim_arr_std[:, -1].mean())
            log.info(
                f"replan L1 (σ-normed): total {l1_std_norm_mean:.4f} | "
                f"arm {arm_l1_std_mean:.4f} | gripper {gripper_l1_std_mean:.4f} | "
                f"action σ per dim " + np.array2string(
                    action_std, precision=4, separator=", "
                )
            )
        else:
            l1_arr = np.zeros(0, dtype=np.float32)
            l1_mean = l1_median = l1_std = l1_p10 = l1_p90 = 0.0
            arm_l1_mean = gripper_l1_mean = 0.0
            perdim_mean = np.zeros(A, dtype=np.float32)
            perdim_arr = np.zeros((0, A), dtype=np.float32)
            l1_hist, l1_edges = np.zeros(0), np.zeros(0)
            action_std = np.zeros(A, dtype=np.float32)
            l1_std_norm_mean = arm_l1_std_mean = gripper_l1_std_mean = 0.0

        # ---- Trigger-vs-disturbance correlation ----
        # Did the TD trigger fire at the same env-step that a disturbance
        # (object kick / action noise) actually occurred? Precision = "of all
        # triggers, how many were on disturbance steps". Recall = "of all
        # disturbance steps, how many did we catch with a trigger". Compared
        # against the chance baseline (lift = overlap / expected_if_independent)
        # this tells you whether the trigger has any timing signal.
        trigger_bool = trigger_trajs > 0
        kick_bool = kick_trajs > 0
        noise_bool = noise_trajs > 0
        disturbance_bool = kick_bool | noise_bool
        n_total = trigger_bool.size
        n_trigger = int(trigger_bool.sum())
        n_kick = int(kick_bool.sum())
        n_noise = int(noise_bool.sum())
        n_dist = int(disturbance_bool.sum())
        overlap_kick = int((trigger_bool & kick_bool).sum())
        overlap_dist = int((trigger_bool & disturbance_bool).sum())
        # Precision / recall vs kick events
        prec_kick = overlap_kick / max(n_trigger, 1)
        rec_kick = overlap_kick / max(n_kick, 1)
        # Precision / recall vs any disturbance
        prec_dist = overlap_dist / max(n_trigger, 1)
        rec_dist = overlap_dist / max(n_dist, 1)
        # Chance-level expectation if trigger and disturbance were independent
        expected_overlap_kick = (n_trigger / max(n_total, 1)) * n_kick
        expected_overlap_dist = (n_trigger / max(n_total, 1)) * n_dist
        lift_kick = overlap_kick / max(expected_overlap_kick, 1e-9)
        lift_dist = overlap_dist / max(expected_overlap_dist, 1e-9)

        if n_kick > 0:
            log.info(
                f"trigger-vs-kick: trigger={n_trigger} kick={n_kick} overlap={overlap_kick} "
                f"| precision {prec_kick:.3f} | recall {rec_kick:.3f} "
                f"| lift {lift_kick:.2f}× chance (chance overlap {expected_overlap_kick:.1f})"
            )
        if n_noise > 0:
            log.info(
                f"trigger-vs-any-dist: trigger={n_trigger} dist={n_dist} overlap={overlap_dist} "
                f"| precision {prec_dist:.3f} | recall {rec_dist:.3f} "
                f"| lift {lift_dist:.2f}× chance"
            )
        if n_kick == 0 and n_noise == 0:
            log.info("[trigger-vs-disturbance] no disturbance events recorded "
                     "(wrappers disabled or info flags missing); skipping correlation")

        # ---- Windowed overlap: account for TD detection lag ----
        # TD residual takes >=1 env-step to register a disturbance because V(s')
        # only exists after the step. Mild kicks may need 2-3 steps to propagate
        # into V. Compare trigger@t with kick@(t-Δ..t) across Δ = 0..3.
        # For each window size, also compute the chance baseline assuming
        # independence (kick rate, trigger rate, total step budget known).
        def _windowed_overlap(trigger_2d, kick_2d, window):
            """Per-env: did kick@t get followed by ANY trigger in [t..t+window]?"""
            T, n_env = kick_2d.shape
            if window == 0:
                catch = (trigger_2d & kick_2d).sum(axis=0)   # per-env
                return int(catch.sum())
            # Cumulative "trigger in upcoming `window` steps starting at row t"
            # Slide trigger up by 0..window and OR them.
            trig_or = trigger_2d.copy()
            for w in range(1, window + 1):
                shifted = np.zeros_like(trigger_2d)
                shifted[: T - w] = trigger_2d[w:]
                trig_or = trig_or | shifted
            return int((trig_or & kick_2d).sum())
        if n_kick > 0:
            windows = [0, 1, 2, 3]
            log.info(
                f"trigger-vs-kick lag analysis "
                f"(does trigger fire within Δ steps AFTER a kick?):"
            )
            for w in windows:
                catch = _windowed_overlap(trigger_bool, kick_bool, w)
                p_trig = n_trigger / max(n_total, 1)
                p_catch_chance = 1.0 - (1.0 - p_trig) ** (w + 1)
                chance = p_catch_chance * n_kick
                lift_w = catch / max(chance, 1e-9)
                recall_w = catch / max(n_kick, 1)
                log.info(
                    f"  Δ={w}: kick caught by trigger {catch:6d}/{n_kick} "
                    f"(recall {recall_w:.3f}) | chance {chance:7.1f} "
                    f"| lift {lift_w:.2f}×"
                )

        time = timer()
        log.info(
            f"eval: ep {num_episode_finished:4d} | success {success_rate:6.4f} "
            f"| reward {avg_episode_reward:8.4f} (best/step {avg_best_reward:8.4f}) "
            f"| avg NFE {avg_nfe:6.3f} | replan {replan_rate:5.3f} | trigger {trigger_rate:5.3f} "
            f"| EMA δ μ={ema_mu:+.4f} σ={np.sqrt(ema_var):.4f}"
        )

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
            trigger_rate=trigger_rate,
            ema_mu=ema_mu,
            ema_var=ema_var,
            nfe_hist=nfe_hist,
            nfe_edges=nfe_edges,
            delta_hist=delta_hist,
            delta_edges=delta_edges,
            tau=self.tau,
            time=time,
            l1_replan_mean=l1_mean,
            l1_replan_median=l1_median,
            l1_replan_std=l1_std,
            l1_replan_p10=l1_p10,
            l1_replan_p90=l1_p90,
            l1_replan_hist=l1_hist,
            l1_replan_edges=l1_edges,
            l1_replan_samples=l1_arr,
            l1_replan_perdim_mean=perdim_mean,
            l1_replan_perdim_samples=perdim_arr,
            l1_replan_arm_mean=arm_l1_mean,
            l1_replan_gripper_mean=gripper_l1_mean,
            action_std_perdim=action_std,
            l1_replan_std_norm_mean=l1_std_norm_mean,
            l1_replan_arm_std_mean=arm_l1_std_mean,
            l1_replan_gripper_std_mean=gripper_l1_std_mean,
            # Trigger-vs-disturbance correlation
            trigger_trajs=trigger_trajs.astype(np.float32),
            kick_trajs=kick_trajs,
            noise_trajs=noise_trajs,
            n_trigger=n_trigger,
            n_kick=n_kick,
            n_noise=n_noise,
            n_disturbance=n_dist,
            overlap_trigger_kick=overlap_kick,
            overlap_trigger_disturbance=overlap_dist,
            precision_vs_kick=prec_kick,
            recall_vs_kick=rec_kick,
            precision_vs_disturbance=prec_dist,
            recall_vs_disturbance=rec_dist,
            lift_vs_kick=lift_kick,
            lift_vs_disturbance=lift_dist,
        )
