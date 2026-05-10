# agents/__init__.py

from agents.network import CNNPolicy
from agents.ppo import PPOAgent

__all__ = [
    'CNNPolicy',
    'PPOAgent',
]
