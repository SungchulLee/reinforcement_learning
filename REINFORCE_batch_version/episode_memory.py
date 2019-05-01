from collections import deque

import numpy as np


class Episode:

    def __init__(self, config):
        self.config = config
        self.ep = deque()

    def initialize(self, state):
        self.ep.append(state)

    def add(self, action, reward, state):
        self.ep.extend([action, reward, state])

    def get_batch(self):
        self.ep.pop()
        n = len(self.ep) // 3
        S = []
        A = np.zeros(n)
        V = np.zeros(n)
        v = 0
        for i in range(n):
            v *= self.config.df
            v += self.ep.pop()
            V[i] = v

            A[i] = self.ep.pop()
            S.append(self.ep.pop())

        if len(self.ep) != 0:
            raise ValueError

        return S, A, V
