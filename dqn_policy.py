import numpy as np


class POLICY:
    def __init__(self, n_states=11, n_actions=4, policy_choice='random'):
        """
        policy_choice = 'random', 'bad', or 'optimal'
        """
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.policy_choice = policy_choice

        self.policy = np.empty((self.n_states, self.n_actions))
        if self.policy_choice == 'random':
            self.policy = 0.25 * np.ones((self.n_states, self.n_actions))
        if self.policy_choice == 'bad':
            self.policy[0, :] = [0, 1, 0, 0]
            self.policy[1, :] = [0, 1, 0, 0]
            self.policy[2, :] = [0, 1, 0, 0]
            self.policy[3, :] = [0, 1, 0, 0]
            self.policy[4, :] = [0, 0, 0, 1]
            self.policy[5, :] = [0, 1, 0, 0]
            self.policy[6, :] = [0, 1, 0, 0]
            self.policy[7, :] = [0, 1, 0, 0]
            self.policy[8, :] = [0, 1, 0, 0]
            self.policy[9, :] = [0, 0, 1, 0]
            self.policy[10, :] = [0, 0, 1, 0]
        if self.policy_choice == 'optimal':
            self.policy[0, :] = [0, 1, 0, 0]
            self.policy[1, :] = [0, 1, 0, 0]
            self.policy[2, :] = [0, 1, 0, 0]
            self.policy[3, :] = [1, 0, 0, 0]
            self.policy[4, :] = [0, 0, 1, 0]
            self.policy[5, :] = [0, 0, 1, 0]
            self.policy[6, :] = [1, 0, 0, 0]
            self.policy[7, :] = [0, 0, 1, 0]
            self.policy[8, :] = [1, 0, 0, 0]
            self.policy[9, :] = [1, 0, 0, 0]
            self.policy[10, :] = [1, 0, 0, 0]
        if self.policy_choice == 'optimal_?':
            self.policy[0, :] = [0, 1, 0, 0]
            self.policy[1, :] = [0, 1, 0, 0]
            self.policy[2, :] = [0, 1, 0, 0]
            self.policy[3, :] = [1, 0, 0, 0]
            self.policy[4, :] = [0, 0, 1, 0]
            self.policy[5, :] = [0, 0, 1, 0]
            self.policy[6, :] = [1, 0, 0, 0]
            self.policy[7, :] = [0, 0, 1, 0]
            self.policy[8, :] = [0, 1, 0, 0]
            self.policy[9, :] = [0, 0, 1, 0]
            self.policy[10, :] = [1, 0, 0, 0]