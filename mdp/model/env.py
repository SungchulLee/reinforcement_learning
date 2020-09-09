import numpy as np


class TransProb:

    def __init__(self, np_seed=1):
        np.random.seed(np_seed)

        # transition probabilities
        N_STATES = 11
        N_ACTIONS = 4
        P = np.empty((N_STATES, N_ACTIONS, N_STATES))

        #                0   1   2   3   4   5   6   7   8   9  10
        P[0, 0, :] = [.9, 0, 0, 0, .1, 0, 0, 0, 0, 0, 0]
        P[0, 1, :] = [.1, .8, 0, 0, .1, 0, 0, 0, 0, 0, 0]
        P[0, 2, :] = [.9, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        P[0, 3, :] = [.1, .1, 0, 0, .8, 0, 0, 0, 0, 0, 0]

        #                0   1   2   3   4   5   6   7   8   9  10
        P[1, 0, :] = [.8, .2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        P[1, 1, :] = [0, .2, .8, 0, 0, 0, 0, 0, 0, 0, 0]
        P[1, 2, :] = [.1, .8, .1, 0, 0, 0, 0, 0, 0, 0, 0]
        P[1, 3, :] = [.1, .8, .1, 0, 0, 0, 0, 0, 0, 0, 0]

        #                0   1   2   3   4   5   6   7   8   9  10
        P[2, 0, :] = [0, .8, .1, 0, 0, .1, 0, 0, 0, 0, 0]
        P[2, 1, :] = [0, 0, .1, .8, 0, .1, 0, 0, 0, 0, 0]
        P[2, 2, :] = [0, .1, .8, .1, 0, 0, 0, 0, 0, 0, 0]
        P[2, 3, :] = [0, .1, 0, .1, 0, .8, 0, 0, 0, 0, 0]

        #                0   1   2   3   4   5   6   7   8   9  10
        P[3, 0, :] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        P[3, 1, :] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        P[3, 2, :] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        P[3, 3, :] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

        #                0   1   2   3   4   5   6   7   8   9  10
        P[4, 0, :] = [.1, 0, 0, 0, .8, 0, 0, .1, 0, 0, 0]
        P[4, 1, :] = [.1, 0, 0, 0, .8, 0, 0, .1, 0, 0, 0]
        P[4, 2, :] = [.8, 0, 0, 0, .2, 0, 0, 0, 0, 0, 0]
        P[4, 3, :] = [0, 0, 0, 0, .2, 0, 0, .8, 0, 0, 0]

        #                0   1   2   3   4   5   6   7   8   9  10
        P[5, 0, :] = [0, 0, .1, 0, 0, .8, 0, 0, 0, .1, 0]
        P[5, 1, :] = [0, 0, .1, 0, 0, 0, .8, 0, 0, .1, 0]
        P[5, 2, :] = [0, 0, .8, 0, 0, .1, .1, 0, 0, 0, 0]
        P[5, 3, :] = [0, 0, 0, 0, 0, .1, .1, 0, 0, .8, 0]

        #                0   1   2   3   4   5   6   7   8   9  10
        P[6, 0, :] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        P[6, 1, :] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        P[6, 2, :] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        P[6, 3, :] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

        #                0   1   2   3   4   5   6   7   8   9  10
        P[7, 0, :] = [0, 0, 0, 0, .1, 0, 0, .9, 0, 0, 0]
        P[7, 1, :] = [0, 0, 0, 0, .1, 0, 0, .1, .8, 0, 0]
        P[7, 2, :] = [0, 0, 0, 0, .8, 0, 0, .1, .1, 0, 0]
        P[7, 3, :] = [0, 0, 0, 0, 0, 0, 0, .9, .1, 0, 0]

        #                0   1   2   3   4   5   6   7   8   9  10
        P[8, 0, :] = [0, 0, 0, 0, 0, 0, 0, .8, .2, 0, 0]
        P[8, 1, :] = [0, 0, 0, 0, 0, 0, 0, 0, .2, .8, 0]
        P[8, 2, :] = [0, 0, 0, 0, 0, 0, 0, .1, .8, .1, 0]
        P[8, 3, :] = [0, 0, 0, 0, 0, 0, 0, .1, .8, .1, 0]

        #                0   1   2   3   4   5   6   7   8   9  10
        P[9, 0, :] = [0, 0, 0, 0, 0, .1, 0, 0, .8, .1, 0]
        P[9, 1, :] = [0, 0, 0, 0, 0, .1, 0, 0, 0, .1, .8]
        P[9, 2, :] = [0, 0, 0, 0, 0, .8, 0, 0, .1, 0, .1]
        P[9, 3, :] = [0, 0, 0, 0, 0, 0, 0, 0, .1, .8, .1]

        #                0   1   2   3   4   5   6   7   8   9  10
        P[10, 0, :] = [0, 0, 0, 0, 0, 0, .1, 0, 0, .8, .1]
        P[10, 1, :] = [0, 0, 0, 0, 0, 0, .1, 0, 0, 0, .9]
        P[10, 2, :] = [0, 0, 0, 0, 0, 0, .8, 0, 0, .1, .1]
        P[10, 3, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, .1, .9]

        self.transition_probs = P

        self.num_states = self.transition_probs.shape[0]


class Policy:

    def __init__(self, policy='random', np_seed=1):
        np.random.seed(np_seed)

        if policy == 'random':
            N_STATES = 11
            N_ACTIONS = 4
            self.policy = 0.25 * np.ones((N_STATES, N_ACTIONS))
        elif policy == 'bad':
            N_STATES = 11
            N_ACTIONS = 4
            self.policy = np.empty((N_STATES, N_ACTIONS))
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
        elif policy == 'optimal':
            N_STATES = 11
            N_ACTIONS = 4
            self.policy = np.empty((N_STATES, N_ACTIONS))
            self.policy[0, :] = [0, 1, 0, 0]
            self.policy[1, :] = [0, 1, 0, 0]
            self.policy[2, :] = [0, 1, 0, 0]
            self.policy[3, :] = [0, 1, 0, 0]
            self.policy[4, :] = [0, 0, 1, 0]
            self.policy[5, :] = [0, 0, 1, 0]
            self.policy[6, :] = [0, 0, 1, 0]
            self.policy[7, :] = [0, 0, 1, 0]
            self.policy[8, :] = [1, 0, 0, 0]
            self.policy[9, :] = [1, 0, 0, 0]
            self.policy[10, :] = [1, 0, 0, 0]
        elif policy == 'noizy_optimal':
            # optimal policy + noise
            # we use optimal policy with probability 1/(1+ep)
            # we use random policy with probability ep/(1+ep)
            ep = 0.1
            self.policy = np.empty((N_STATES, N_ACTIONS))
            self.policy[0, :] = [0, 1, 0, 0]
            self.policy[1, :] = [0, 1, 0, 0]
            self.policy[2, :] = [0, 1, 0, 0]
            self.policy[3, :] = [0, 1, 0, 0]
            self.policy[4, :] = [0, 0, 1, 0]
            self.policy[5, :] = [0, 0, 1, 0]
            self.policy[6, :] = [0, 0, 1, 0]
            self.policy[7, :] = [0, 0, 1, 0]
            self.policy[8, :] = [1, 0, 0, 0]
            self.policy[9, :] = [1, 0, 0, 0]
            self.policy[10, :] = [1, 0, 0, 0]
            self.policy = self.policy + (ep / 4) * np.ones((N_STATES, N_ACTIONS))
            self.policy = self.policy / np.sum(self.policy, axis=1).reshape((N_STATES, 1))
