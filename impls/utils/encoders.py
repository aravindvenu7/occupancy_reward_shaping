import functools
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP


class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out

class ResnetDecoderStack(nn.Module):
    """Symmetric ResNet-style upsampling stack."""
    num_features: int
    num_blocks: int
    upsample: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        # Upsample if needed
        if self.upsample:
            # Nearest neighbor upsampling by (2,2) in spatial dims
            B, H, W, C = x.shape
            x = jnp.repeat(x, 2, axis=1)
            x = jnp.repeat(x, 2, axis=2)
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)
        # Residual blocks
        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input
        return conv_out

class ImpalaDecoder(nn.Module):
    output_shape: Sequence[int]                # (H, W, C) of original image
    num_blocks: int = 2
    stack_sizes: Sequence[int] = (32, 32, 16)
    mlp_hidden_dims: Sequence[int] = (2048,)
    dropout_rate: float = None
    layer_norm: bool = False

    def setup(self):
        # Reverse stacks for decoder (mirrored structure)
        self.stack_blocks = [
            ResnetDecoderStack(num_features=n, num_blocks=self.num_blocks, upsample=(i > -1))
            for i, n in enumerate(self.stack_sizes)
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)
        
    @nn.compact
    def __call__(self, z, train=True):
        z = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(z)
        # Calculate starting spatial dimensions
        H, W, C = self.output_shape
        num_stacks = len(self.stack_sizes)
        init_H = H // (2 ** num_stacks) #8
        init_W = W // (2 ** num_stacks) #8

        proj_dim = self.stack_sizes[0]
        # Reshape to spatial feature map
        x = z.reshape((*z.shape[:-1], init_H, init_W, proj_dim))
        for i, stack in enumerate(self.stack_blocks):
            x = stack(x)
            if self.dropout_rate is not None:
                x = self.dropout(x, deterministic=not train)

        x = nn.relu(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        # Final conv to reconstruct original image (C channels)
        x = nn.Conv(features=C, kernel_size=(3, 3), strides=1, padding='SAME')(x)
        x = nn.sigmoid(x) * 255.0
        return x

        
class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out


class GCEncoder(nn.Module):
    """Helper module to handle inputs to goal-conditioned networks.

    It takes in observations (s) and goals (g) and returns the concatenation of `state_encoder(s)`, `goal_encoder(g)`,
    and `concat_encoder([s, g])`. It ignores the encoders that are not provided. This way, the module can handle both
    early and late fusion (or their variants) of state and goal information.
    """

    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    concat_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded=False):
        """Returns the representations of observations and goals.

        If `goal_encoded` is True, `goals` is assumed to be already encoded representations. In this case, either
        `goal_encoder` or `concat_encoder` must be None.
        """
        reps = []
        if self.state_encoder is not None:
            reps.append(self.state_encoder(observations))
        if goals is not None:
            if goal_encoded:
                # Can't have both goal_encoder and concat_encoder in this case.
                assert self.goal_encoder is None or self.concat_encoder is None
                reps.append(goals)
            else:
                if self.goal_encoder is not None:
                    reps.append(self.goal_encoder(goals))
                if self.concat_encoder is not None:
                    reps.append(self.concat_encoder(jnp.concatenate([observations, goals], axis=-1)))
        reps = jnp.concatenate(reps, axis=-1)
        return reps


encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
}


decoder_modules = {
    'impala_small': functools.partial(ImpalaDecoder, num_blocks=1),
}

