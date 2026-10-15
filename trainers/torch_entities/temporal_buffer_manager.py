from collections import deque
import numpy as np


class TemporalBufferManager:
    def __init__(self, window_size, state_dim):
        if window_size < 1 or state_dim < 1:
            raise ValueError("window_size and state_dim must be positive")
        self.window_size = window_size
        self.state_dim = state_dim
        self.agent_buffers = {}

    def update(self, agent_id, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.state_dim,):
            raise ValueError("Observation does not match temporal state_dim")
        self.agent_buffers.setdefault(agent_id, deque(maxlen=self.window_size)).append(
            observation.copy()
        )

    def get_window(self, agent_id):
        window = np.zeros((self.window_size, self.state_dim), dtype=np.float32)
        history = self.agent_buffers.get(agent_id)
        if history:
            window[-len(history):] = np.stack(history)
        return window

    def reset(self, agent_id):
        self.agent_buffers.pop(agent_id, None)
