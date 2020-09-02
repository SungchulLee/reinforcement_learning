import numpy as np


class MarkovChain:

    def __init__(self, transition_probs, initial_distribution, np_seed=1):
        self.P = transition_probs
        self.pi = initial_distribution
        np.random.seed(np_seed)

        self.num_states = len(initial_distribution)
        self.states = np.arange(self.num_states)
        self.current_state = None

    def reset(self):
        self.current_state = np.random.choice(self.states, p=self.pi)

    def step(self):
        self.current_state = np.random.choice(self.states,
                                              p=self.P[self.current_state, :])
