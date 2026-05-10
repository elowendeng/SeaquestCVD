# agents/network.py

"""
CNN network, supporting dynamic input channels
"""

import equinox as eqx
import jax
import jax.numpy as jnp


class CNNPolicy(eqx.Module):
    """
    CNN policy for Atari games
    Input: (channels, height, width)
    """
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    fc: eqx.nn.Linear
    action_head: eqx.nn.Linear
    value_head: eqx.nn.Linear
    
    def __init__(self, action_dim: int = 18, input_channels: int = 12, key=None):
        """
        Args:
            action_dim: Dimension of action space (Seaquest=18)
            input_channels: Number of input channels (4 frames × 3 RGB = 12)
            key: JAX random seed
        """
        keys = jax.random.split(key, 7)
        self.conv1 = eqx.nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, 
                                    padding=0, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=4, stride=2, 
                                    padding=0, key=keys[1])
        self.conv3 = eqx.nn.Conv2d(64, 64, kernel_size=3, stride=1, 
                                    padding=0, key=keys[2])  
        # Calculate the flattened dimension: 84x84. The input undergoes three layers of convolution.
        # conv1: (84-8)/4+1=20, conv2: (20-4)/2+1=9, conv3: (9-3)/1+1=7 -> 64*7*7=3136
        self.fc = eqx.nn.Linear(3136, 512, key=keys[3])
        # Value head uses a smaller initialization.
        self.action_head = eqx.nn.Linear(512, action_dim, key=keys[4])
        self.value_head = eqx.nn.Linear(512, 1, key=keys[5])
    
    def __call__(self, x):
        """Forward propagation"""
        # x shape: (channels, height, width)
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.relu(self.conv3(x))
        x = x.reshape(-1)
        x = jax.nn.relu(self.fc(x))
        logits = self.action_head(x)
        value = self.value_head(x).squeeze()
        return logits, value

    def batch_forward(self, x):
        """Batch forward propagation (for training)"""
        # x shape: (batch, H, W, C) or (batch, C, H, W)
        if len(x.shape) == 4:
            # If it is in the NHWC format, convert it to NCHW
            if x.shape[-1] in [12, 24]:  # The last dimension is the channel
                x = jnp.transpose(x, (0, 3, 1, 2))
        logits, values = jax.vmap(self)(x)
        return logits, values
