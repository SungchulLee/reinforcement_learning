import numpy as np


class MDP:

    def __init__(self, transition_probs, initial_distribution, policy, np_seed=1):
        self.transition_probs = transition_probs
        self.initial_distribution = initial_distribution
        self.policy = policy
        np.random.seed(np_seed)

        self.num_states = len(initial_distribution)
        self.states = np.arange(self.num_states)
        self.num_actions = transition_probs.shape[1]
        self.actions = np.arange(self.num_actions)
        self.current_state = None
        self.current_action = None

    def reset(self):
        self.current_state = np.random.choice(self.states, p=self.initial_distribution)

    def step(self):
        self.current_action = np.random.choice(self.actions, p=self.policy[self.current_state, :])
        p = self.transition_probs[self.current_state, self.current_action, :]
        self.current_state = np.random.choice(self.states, p=p)
