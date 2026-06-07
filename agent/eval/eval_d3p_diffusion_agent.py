import os
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.eval.eval_agent import EvalAgent
from d3p_utils.D3PAdaptor import D3PAdaptor


class EvalD3PDiffusionAgent(EvalAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

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

    def run(self):
        timer = Timer()

        options_venv = [{} for _ in range(self.n_envs)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )

        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))
        nfe_trajs = np.zeros((self.n_steps, self.n_envs))  # dynamic denoising-step count, NFE = Number of Function Evaluations
        if self.save_full_observations:
            obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
            obs_full_trajs = np.vstack(
                (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
            )

        for step in range(self.n_steps):
            if step % 10 == 0:
                print(f"Processed step {step} of {self.n_steps}")

            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                samples, _k, _k_logp, _idx, _vmask, stp_t = self.model.forward_d3p(
                    cond=cond,
                    adaptor=self.adaptor,
                    deterministic=True,
                )
                output_venv = samples.trajectories.cpu().numpy()
                nfe_trajs[step] = stp_t.cpu().numpy()
            action_venv = output_venv[:, : self.act_steps]

            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = terminated_venv | truncated_venv
            if self.save_full_observations:
                obs_full_venv = np.array(
                    [info["full_obs"]["state"] for info in info_venv]
                )
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                )

            prev_obs_venv = obs_venv

        # Episode bookkeeping (mirrors EvalDiffusionAgent)
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
            if self.furniture_sparse_reward:
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
            num_episode_finished = 0
            avg_episode_reward = 0
            avg_best_reward = 0
            success_rate = 0
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
        nfe_hist, nfe_edges = np.histogram(
            nfe_trajs.ravel(),
            bins=np.arange(0, self.model.ddim_steps + 2) - 0.5,
        )

        time = timer()
        log.info(
            f"eval: num episode {num_episode_finished:4d} | success rate {success_rate:8.4f} "
            f"| avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f} "
            f"| avg NFE {avg_nfe:6.3f} (max {self.model.ddim_steps})"
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
            avg_nfe=avg_nfe,
            nfe_hist=nfe_hist,
            nfe_edges=nfe_edges,
            time=time,
            # Per-episode raw data.
            episode_reward=episode_reward_arr,
            episode_best_reward=episode_best_reward_arr,
            episode_env_ind=episode_env_ind_arr,
            episode_start_step=episode_start_step_arr,
            best_reward_threshold_for_success=float(self.best_reward_threshold_for_success),
        )
