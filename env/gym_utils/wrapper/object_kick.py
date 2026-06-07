"""
Object-kick wrapper — inject a Gaussian velocity impulse into a target object
joint AFTER each underlying env step. Models a transient external disturbance
("wind gust" / "bump") on the manipulated object.

Conceptually stronger than ActionNoiseWrapper for testing adaptive replanning:
action noise only shakes the arm (chunk smoothness absorbs it), whereas a
velocity kick on the object directly perturbs world state — the next obs no
longer matches the policy's planned chunk, so the policy must actually
re-plan to recover.

Place AFTER the env-specific wrapper (e.g. RobomimicLowdim) and either before
or after ActionNoiseWrapper (they don't conflict — noise mutates the action
before step, kick mutates the sim state after step). MUST be inside MultiStep
so the kick fires per physical sub-step:

    env:
      wrappers:
        robomimic_lowdim:
          ...
        object_kick:
          p: 0.05               # per-step probability of a kick
          impulse_sigma: 0.3    # impulse stddev in m/s (per axis)
          target_joint_name: cube_joint0   # robosuite task-specific
          apply_z: False        # default: kick only in xy plane
          clip_magnitude: 1.0   # optional |impulse| cap per axis
        action_noise:
          ...
        multi_step:
          ...

CLI overrides:
    env.wrappers.object_kick.p=0.1
    env.wrappers.object_kick.impulse_sigma=0.5

At impulse_sigma=0 or p=0 the wrapper is a transparent passthrough.

Why velocity kick (not position teleport):
  - Physically self-consistent (momentum transfer, then dynamics resolves)
  - If object is firmly grasped, gripper contact force absorbs the kick →
    natural "grasp gating" without any if-else logic
  - No MuJoCo constraint-violation warnings (position teleport can fight
    with contact constraints)

The wrapper's RNG is decorrelated from the env / policy RNG via seed offset.
"""

import gym
import numpy as np


def _find_sim(env, max_depth: int = 16):
    """Walk down env.env until something exposes a `.sim` (MjSim) attribute."""
    cur = env
    for _ in range(max_depth):
        if hasattr(cur, "sim"):
            return cur.sim
        if not hasattr(cur, "env"):
            break
        cur = cur.env
    return None


class ObjectKickWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        p: float = 0.0,
        impulse_sigma: float = 0.0,
        target_joint_name: str = "cube_joint0",
        apply_z: bool = False,
        clip_magnitude=None,
        seed=None,
        seed_offset: int = 13513,
    ):
        super().__init__(env)
        self.p = float(p)
        assert 0.0 <= self.p <= 1.0, f"object_kick.p must be in [0, 1]; got {p}"
        self.impulse_sigma = float(impulse_sigma)
        self.target_joint_name = str(target_joint_name)
        self.apply_z = bool(apply_z)
        self.clip_magnitude = (
            float(clip_magnitude) if clip_magnitude is not None else None
        )
        self._seed_offset = int(seed_offset)
        self._rng = np.random.default_rng(seed)
        # Lazy — robosuite sim may not be constructed until first reset/step.
        self._sim = None
        self._qvel_start = None

    def seed(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed + self._seed_offset)
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None

    def _ensure_init(self):
        if self._sim is not None and self._qvel_start is not None:
            return
        sim = _find_sim(self.env)
        if sim is None:
            raise AttributeError(
                "ObjectKickWrapper could not locate a MjSim object by walking "
                "env.env... — make sure this wrapper is composed over an env "
                "that ultimately exposes a `.sim` attribute (e.g. robosuite)."
            )
        try:
            addr = sim.model.get_joint_qvel_addr(self.target_joint_name)
        except Exception as e:
            try:
                available = list(sim.model.joint_names)
            except Exception:
                available = "<unavailable>"
            raise ValueError(
                f"ObjectKickWrapper: joint '{self.target_joint_name}' not "
                f"found in the underlying sim. Available joints: {available}"
            ) from e
        # mujoco-py returns (start, end) for multi-DOF joints, int for 1-DOF.
        if isinstance(addr, tuple):
            self._qvel_start = int(addr[0])
        else:
            self._qvel_start = int(addr)
        self._sim = sim

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        kick_fired = False
        # Skip work cheaply if no kick can occur.
        if self.impulse_sigma > 0.0 and self.p > 0.0 and self._rng.random() < self.p:
            self._ensure_init()
            n_dims = 3 if self.apply_z else 2
            impulse = self._rng.standard_normal(n_dims).astype(np.float64) * self.impulse_sigma
            if self.clip_magnitude is not None:
                impulse = np.clip(impulse, -self.clip_magnitude, self.clip_magnitude)
            s = self._qvel_start
            self._sim.data.qvel[s : s + n_dims] += impulse
            kick_fired = True
        # Expose for downstream diagnostics (e.g. trigger vs disturbance correlation).
        info["kick_fired"] = kick_fired
        return obs, reward, done, info
