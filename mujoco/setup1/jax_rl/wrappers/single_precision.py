import gym
import numpy as np


class SinglePrecision(gym.ObservationWrapper):
    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.astype(np.float32)
