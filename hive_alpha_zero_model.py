"""
This module exports the `HiveAlphaZeroModel` class for predicting the output of
policy and value functions given a Hive board state.
"""

import numpy as np

from torch import nn
import torch.nn.functional as F

"""
@todo:
   - l2 kernel regularization (available in Keras as layer parameter, in torch param for optimizer)
   - play with (decrease?) hyperparameters (n_filters, n_res_layers, value_fc_size)
       - also batch_norm_kwargs - we're using keras defaults; maybe change to torch defaults?
   - change action shape from 22x22x7 to 22x22x6?
"""

class ResidualBlock(nn.Module):
    """A single block of residual layers."""
    def __init__(self, n_filters: int, filter_size: int, **batch_norm_kwargs) -> None:
        super().__init__()

        self.conv_1 = nn.Conv2d(n_filters, n_filters, kernel_size=filter_size, padding="same", bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(n_filters, **batch_norm_kwargs)

        self.conv_2 = nn.Conv2d(n_filters, n_filters, kernel_size=filter_size, padding="same", bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(n_filters, **batch_norm_kwargs)

    def forward(self, x):
        """Forward the input of the residual block."""
        residual = x

        x = self.batch_norm_1(self.conv_1(x))
        x = F.relu(x)

        x = self.batch_norm_2(self.conv_2(x))

        x += residual

        x = F.relu(x)

        return x

class HiveAlphaZeroModel(nn.Module):
    """Class for predicting the output of policy and value functions given a Hive board state."""
    def __init__(
        self,
        state_shape=(23,23,18),
        action_prob_shape=(22,22,7),
        policy_n_filters=2,
        value_n_filters=4,
        n_filters=256,
        input_filter_size=5,
        filter_size=3,
        n_res_layers=7,
        value_fc_size=256
    ):
        super().__init__()

        self._build(state_shape, action_prob_shape,
                    policy_n_filters, value_n_filters, n_filters,
                    input_filter_size, filter_size,
                    n_res_layers,
                    value_fc_size)

    def _build(
        self,
        state_shape: tuple[int, int, int],
        action_prob_shape: tuple[int, int, int],
        policy_n_filters: int,
        value_n_filters: int,
        n_filters: int,
        input_filter_size: int,
        filter_size: int,
        n_res_layers: int,
        value_fc_size: int
    ):
        """Build the torch model."""
        state_n_channels = state_shape[2]

        # keras defaults
        batch_norm_kwargs = dict(eps=1e-3, momentum=0.99)

        self.input_block = nn.Sequential(
            nn.Conv2d(state_n_channels, n_filters, padding="same", kernel_size=input_filter_size, bias=False),
            nn.BatchNorm2d(n_filters, **batch_norm_kwargs),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList((
            ResidualBlock(n_filters, filter_size, **batch_norm_kwargs)
            for _ in range(n_res_layers)
        ))

        state_plane_size = np.prod(state_shape[:2])

        policy_n_inputs = policy_n_filters * state_plane_size

        policy_n_outputs = np.prod(action_prob_shape)

        self.policy_head = nn.Sequential(
            nn.Conv2d(n_filters, policy_n_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_n_filters, **batch_norm_kwargs),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_n_inputs, policy_n_outputs)
        )

        value_n_inputs = value_n_filters * state_plane_size

        self.value_head = nn.Sequential(
            nn.Conv2d(n_filters, value_n_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_n_filters, **batch_norm_kwargs),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(value_n_inputs, value_fc_size),
            nn.ReLU(),
            nn.Linear(value_fc_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        """Predict the output of policy and value functions given a state input or a batch of inputs."""
        x = self.input_block(x)

        value = self.value_head(x)
        policy = self.policy_head(x)

        return policy, value
