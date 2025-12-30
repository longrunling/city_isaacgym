import numpy as np
import torch
import gym
from gym import spaces
from Envs.utils import flatten_observations, flatten_observation_spaces

class EnvWrapperSKRL():
    """An env wrapper that flattens the observation dictionary to an array."""
    def __init__(self, gym_env):
        """Initializes the wrapper."""
        self._gym_env = gym_env
        self.observation_space = self._flatten_observation_spaces(self._gym_env.observation_space)
        self.action_space = self._gym_env.action_space
        # self.key_sequence = ["state","occ_map"]

    def __getattr__(self, attr):
        return getattr(self._gym_env, attr)

    def _flatten_observation_spaces(self, observation_spaces):
        flat_observation_space = flatten_observation_spaces(
            observation_spaces=observation_spaces, 
            key_sequence=["state","occ_map"]
        )
        return flat_observation_space

    def _flatten_observation(self, input_observation):
        """Flatten the dictionary to an array."""
        return flatten_observations(
            observation_dict=input_observation, 
            key_sequence=["state","occ_map"]
        )

    def reset(self):
        observation = self._gym_env.reset()
        return {'obs': self._flatten_observation(observation)}

    def step(self, action):
        """Steps the wrapped environment.

        Args:
          action: Numpy array. The input action from an NN agent.

        Returns:
          The tuple containing the flattened observation, the reward, the epsiode
            end indicator.
        """
        observation_dict, reward, done, info = self._gym_env.step(action)
        for key in info['episode']:
            info['episode'][key] = torch.tensor(info['episode'][key])
        return {'obs': self._flatten_observation(observation_dict)}, reward, done, info

    def render(self, mode='human'):
        return self._gym_env.render(mode)

    def close(self):
        self._gym_env.close()