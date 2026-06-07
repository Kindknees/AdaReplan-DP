import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from model.common.mlp import MLP


class ReplanController(nn.Module):
    """
    Learned replan trigger for DPPO/D3P.

    At every environment step the controller observes a small feature vector
    (current obs, the action it is *about* to execute, the critic value V(s_t),
    the previous-step TD residual delta, and how stale the current chunk is),
    and emits a Bernoulli decision:

        u = 0  -> continue (execute the next queued action, 0 NFE)
        u = 1  -> replan   (discard the queue, resample a fresh chunk, pay NFE)

    It is a small actor-critic: a shared MLP trunk feeds a policy logit head
    (Bernoulli) and a value head V_psi used as the PPO/GAE baseline. The value
    head is the controller's *own* baseline; it is distinct from the frozen
    diffusion critic V_Theta, which is only used to build the delta feature.
    """

    def __init__(
        self,
        feature_dim,
        mlp_dims=[128, 128],
        init_replan_bias=-1.0,
    ):
        super().__init__()

        dim_list = [feature_dim] + list(mlp_dims)
        self.trunk = MLP(
            dim_list=dim_list,
            activation_type="Mish",
            out_activation_type="Mish",
            use_layernorm=False,
        )
        self.policy_head = nn.Linear(mlp_dims[-1], 1)
        self.value_head = nn.Linear(mlp_dims[-1], 1)

        # Bias the trigger toward "continue" at init so early rollouts don't
        # collapse to replanning every step (which would just reproduce the
        # act_steps=1 closed-loop baseline). sigmoid(init_replan_bias) is the
        # initial replan probability, e.g. -1.0 -> ~0.27.
        nn.init.zeros_(self.policy_head.weight)
        nn.init.constant_(self.policy_head.bias, init_replan_bias)

    def forward(self, feat):
        """
        feat: (B, feature_dim)
        returns: (Bernoulli dist over {0,1}, value (B,))
        """
        h = self.trunk(feat)
        logit = self.policy_head(h).squeeze(-1)
        value = self.value_head(h).squeeze(-1)
        dist = Bernoulli(logits=logit)
        return dist, value
