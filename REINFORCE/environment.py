from sclee_env import ENVIRONMENT as ScleeEnv

import numpy as np


class Environment:

    def __init__(self):
        self.make()
        self.action_dict = {0: "l", 1: "r", 2: "u", 3: "d"}

    def initialize_game(self):
        self._state, _ = self.env.reset()
        self.env.reward = -0.02
        return self.state

    def _step(self, action):
        self.reward, self._state, self.done, self.info, _ = self.env.step(
            action)

    @property
    def action_space(self):
        return self.env.actions

    @property
    def action_space_size(self):
        return self.env.num_actions

    @property
    def state(self):
        s = np.zeros(self.env.num_states)
        s[self._state] = 1
        return s

    def act(self, action):
        self._step(action)
        self.env.current_state = self._state
        return self.state, self.reward, self.done

    def make(self):
        self.env = ScleeEnv()
        self._state = None
        self.reward = 0
        self.done = True
        self.info = None