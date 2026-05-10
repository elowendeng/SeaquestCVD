# agents/ppo.py

"""
PPO Agent
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple
import pickle
import optax

from agents.network import CNNPolicy


class PPOAgent:
    """PPO Agent with JAX+Equinox"""
    
    def __init__(
        self,
        action_dim: int = 18,
        input_channels: int = 12,
        learning_rate: float = 2.5e-4,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        seed: int = 42
    ):
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        key = jax.random.PRNGKey(seed)
        self.network = CNNPolicy(action_dim, input_channels, key)
        
        # Create an optimizer using optax
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(self.network, eqx.is_array))

    def get_action_deterministic(self, obs: np.ndarray) -> int:
        """Deterministic action (for evaluation)"""
        # Handling input in HWC format
        if len(obs.shape) == 3:
            # If it is in HWC format (where the last dimension is the channel), convert it to CHW format
            # Standard training: 12 channels, Difference map training: 24 channels
            if obs.shape[-1] in [3, 12, 24]:  # Add 24-channel support
                obs = jnp.transpose(obs, (2, 0, 1))
            else:
                obs = jnp.array(obs)
        else:
            obs = jnp.array(obs)
    
        logits, _ = self.network(obs)
        return int(jnp.argmax(logits))

    def get_action_and_value(self, obs: np.ndarray, key) -> Tuple[int, float, float]:
        """Obtain actions, log probabilities and values"""
        if len(obs.shape) == 3:
            if obs.shape[-1] in [3, 12, 24]:  # Add 24-channel support
                obs = jnp.transpose(obs, (2, 0, 1))
            else:
                obs = jnp.array(obs)
        else:
            obs = jnp.array(obs)
    
        logits, value = self.network(obs)
        probs = jax.nn.softmax(logits)
        action = jax.random.categorical(key, logits)
        log_prob = jnp.log(probs[action] + 1e-8)
    
        return int(action), float(log_prob), float(value)
    
    @eqx.filter_jit
    def _update_batch(
        self,
        model,
        opt_state,
        obs,
        actions,
        old_log_probs,
        advantages,
        returns
    ):
        """Single batch PPO update"""
        
        def loss_fn(model):
            logits, values = model.batch_forward(obs)
            
            # Policy loss with clipping
            new_log_probs = jax.nn.log_softmax(logits)
            action_log_probs = jnp.sum(new_log_probs * jax.nn.one_hot(actions, self.action_dim), axis=-1)
            
            ratio = jnp.exp(action_log_probs - old_log_probs)
            clip_adv = jnp.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            # The policy_loss should be positive, so a negative sign should be used.
            policy_loss = -jnp.minimum(ratio * advantages, clip_adv).mean()
            
            # Value loss (Huber loss is more stable)
            value_error = returns - values
            value_loss = 0.5 * jnp.square(value_error).mean()
            
            # Entropy bonus
            probs = jax.nn.softmax(logits)
            entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1).mean()
            entropy_loss = -self.entropy_coef * entropy
            
            total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
            
            # return (total_loss, aux). Where "aux" is a tuple consisting of three losses.
            return total_loss, (policy_loss, value_loss, entropy)
        
        grad_fn = eqx.filter_grad(loss_fn, has_aux=True)
        grads, aux = grad_fn(model)
        
        # Gradient clipping
        def clip_grad(g):
            if g is None:
                return None
            return jnp.clip(g, -self.max_grad_norm, self.max_grad_norm)
        
        grads = jax.tree_util.tree_map(clip_grad, grads)
        
        # Using optax for updates
        updates, new_opt_state = self.optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        new_model = eqx.apply_updates(model, updates)
        
        # aux is (policy_loss, value_loss, entropy)
        return new_model, new_opt_state, aux
    
    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        num_epochs: int = 4,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """Update strategy"""
        # Change input format
        if len(obs.shape) == 4:  # (batch, H, W, C)
            obs = jnp.transpose(obs, (0, 3, 1, 2))
        else:
            obs = jnp.array(obs)
        
        actions = jnp.array(actions)
        old_log_probs = jnp.array(old_log_probs)
        advantages = jnp.array(advantages)
        returns = jnp.array(returns)
        
        # Normalization advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Normalization returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        n_samples = len(obs)
        indices = np.arange(n_samples)
        
        total_losses = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        n_updates = 0
        
        for _ in range(num_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                
                # Call _update_batch
                self.network, self.opt_state, aux = self._update_batch(
                    self.network, self.opt_state,
                    obs[batch_idx], actions[batch_idx], old_log_probs[batch_idx],
                    advantages[batch_idx], returns[batch_idx]
                )
                
                # aux is (policy_loss, value_loss, entropy)
                policy_loss = float(aux[0])
                value_loss = float(aux[1])
                entropy = float(aux[2])
                
                total_losses['policy_loss'] += policy_loss
                total_losses['value_loss'] += value_loss
                total_losses['entropy'] += entropy
                n_updates += 1
        
        if n_updates > 0:
            for k in total_losses:
                total_losses[k] /= n_updates
        total_losses['total_loss'] = total_losses['policy_loss'] + total_losses['value_loss']
        
        return total_losses
    
    def save(self, path: str):
        """save model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'network': self.network,
                'opt_state': self.opt_state,
                'action_dim': self.action_dim,
            }, f)

    def load(self, path: str):
        """load model with validation"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
    
        # Verify whether the network structure is compatible
        loaded_network = data['network']
    
        # Check if the input channels are matched.
        loaded_channels = loaded_network.conv1.weight.shape[1]  # The input channels of the first layer of convolution
        expected_channels = self.network.conv1.weight.shape[1]
    
        if loaded_channels != expected_channels:
            raise ValueError(
                f"Channel mismatch! Loaded model expects {loaded_channels} input channels, "
                f"but current agent is configured for {expected_channels} channels. "
                f"Please create agent with input_channels={loaded_channels}"
            )
    
        self.network = loaded_network
        self.opt_state = data['opt_state']
        self.action_dim = data['action_dim']
        # Recreate the optimizer
        self.optimizer = optax.adam(self.learning_rate)
