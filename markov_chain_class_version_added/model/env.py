import numpy as np


class TransProb:

    def __init__(self, trans_prob_name='random', np_seed=1):
        self.np_seed = np_seed

        if trans_prob_name == 'random':
            np.random.seed(self.np_seed)
            N_STATES = 4
            self.transition_probs = np.random.normal(0., 1., (N_STATES, N_STATES))
            self.transition_probs = np.exp(self.transition_probs)
            self.transition_probs = self.transition_probs / np.sum(self.transition_probs, axis=1).reshape((N_STATES, 1))

        elif trans_prob_name == 'homogeneous':
            np.random.seed(self.np_seed)
            N_STATES = 4
            self.transition_probs = np.empty((N_STATES, N_STATES))
            l = 0.4
            r = 0.4
            s = 1 - l - r
            #                                 0   1   2   3
            self.transition_probs[0, :] = [s + l, r, 0, 0]
            self.transition_probs[1, :] = [l, s, r, 0]
            self.transition_probs[2, :] = [0, l, s, r]
            self.transition_probs[3, :] = [0, 0, l, s + r]

        elif trans_prob_name == 'non-homogeneous':
            np.random.seed(self.np_seed)
            N_STATES = 4
            self.transition_probs = np.empty((N_STATES, N_STATES))
            #                                 0   1    2    3
            self.transition_probs[0, :] = [0.1, 0.9,   0,   0]
            self.transition_probs[1, :] = [0.1, 0.6, 0.3,   0]
            self.transition_probs[2, :] = [  0, 0.5, 0.4, 0.1]
            self.transition_probs[3, :] = [  0,   0, 0.7, 0.3]

        self.num_states = self.transition_probs.shape[0]
