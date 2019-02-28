import numpy as np
import matplotlib.pyplot as plt

from env import ENVIRONMENT
from dqn_policy import POLICY
import dqn_ops as ops


class MDP(ENVIRONMENT):
    def __init__(self, policy, battery_consumption=-0.02, gamma=0.99,
                 num_simulations=1000):
        super().__init__(policy=policy, battery_consumption=battery_consumption, gamma=gamma)
        self.num_simulations = int(num_simulations)

    def run_one_simulation(self, is_printing=True):
        self.current_state, self.done = self.reset()
        
        while not self.done:
            action = np.random.choice(self.actions, 
                                      p=self.policy[self.current_state, :])
            prob = self.policy[self.current_state, action] 
            self.reward, next_state, self.done, self.info, self.final_reward = self.step(action, prob)
            
            if is_printing: 
                msg = "s: {:2}, a: {}, r: {:5.2f}, s1: {:2}, done: {:1}, info: {}, final_reward: {}"
                print(msg.format(self.current_state, action, self.reward, next_state, \
                                 self.done, self.info, self.final_reward))
                
            self.current_state = next_state
            
        # after game over (while loop broken), report win or lose
        if self.current_state == self.win_state:
            return 1 # report win
        elif self.current_state == self.lose_state:
            return 0 # report lose
            
    def run_many_simulations(self):
        # run simulation self.num_simulations times and
        # record win(1) or lose(0) for each simulation
        simulation_history = []
        for _ in range(self.num_simulations):
            win_or_lose = self.run_one_simulation(is_printing=False)  
            simulation_history.append(win_or_lose)

        # compute running success rate
        x = np.array(simulation_history)
        running_success_rate = np.cumsum(x)/(np.arange(self.num_simulations)+1)

        # print overall success rate
        print("Number of simulations : {}".format(self.num_simulations))
        print("Success rate          : {}".format(running_success_rate[-1]))
        
        # plot running success rate
        plt.plot(running_success_rate)
        plt.title("Running Success Rate")
        plt.show()
        
        
class IPE_V(MDP):
    def __init__(self, policy, battery_consumption=-0.02, gamma=0.99, num_iteration=100):
        super().__init__(policy, battery_consumption=battery_consumption, gamma=gamma)
        self.num_iteration = int(num_iteration)

    def run_ipe(self):
        self.V_history = np.zeros((self.num_iteration, self.num_states))       
        self.V = np.zeros(self.num_states)
        self.V[self.win_state] = 1.
        self.V[self.lose_state] = -1.
        for i in range(self.num_iteration):
            for s in range(self.num_states):
                if (s!=self.win_state) and (s!=self.lose_state):
                    self.V[s] = self.reward + self.gamma * \
                                sum([self.policy[s, a] * \
                                    sum([self.P[s, a, s1] * self.V[s1] \
                                        for s1 in range(self.num_states)]) \
                                    for a in range(self.num_actions)])
            self.V_history[i] = self.V

        return self.V, self.V_history
    
    def plot(self):
        fig = plt.figure(figsize=(16, 12))
        fig.subplots_adjust(hspace=0.1, wspace=0.3)
        fig.suptitle("Iterative_Policy_Evaluation for V", fontsize=16)

        for s in range(self.num_states):
            ax = fig.add_subplot(3, 4, s + 1)
            ax.plot(self.V_history[:, s], label="state {}".format(str(s)))
            ax.legend(loc='upper right')
            ax.set_ylim([-1.1, 1.1])

        plt.show()


class IPE_Q(IPE_V):

    def run_ipe(self):
        self.Q_history = np.zeros((self.num_iteration, \
                                   self.num_states, self.num_actions))
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.Q[self.win_state, :] = 1
        self.Q[self.lose_state, :] = -1
        for i in range(self.num_iteration):
            for s in range(self.num_states):
                if (s!=self.win_state) and (s!=self.lose_state):
                    for a in range(self.num_actions):
                        self.Q[s, a] = self.reward + self.gamma * \
                                       sum([self.P[s, a, s1] * \
                                           sum([self.policy[s1, a1] * self.Q[s1, a1] \
                                                for a1 in range(self.num_actions)]) \
                                           for s1 in range(self.num_states)])
            self.Q_history[i] = self.Q

        return self.Q, self.Q_history
    
    def plot(self):
        fig = plt.figure(figsize=(16, 12))
        fig.subplots_adjust(hspace=0.1, wspace=0.3)
        fig.suptitle("Iterative_Policy_Evaluation for Q", fontsize=16)

        for s in range(self.num_states):
            for a in range(self.num_actions):
                ax = fig.add_subplot(11, 4, 4*s + a + 1)
                ax.plot(self.Q_history[:, s, a], \
                        label="state {}, action {}".format(str(s), str(a)))
                ax.legend(loc='lower right')
                ax.set_ylim([-1.1, 1.1])

        plt.show()
        
        
class VI_V(ENVIRONMENT):
    def __init__(self, battery_consumption=-0.02, gamma=0.99, num_iteration=10):
        super().__init__(battery_consumption=battery_consumption, gamma=gamma)
        self.num_iteration = int(num_iteration)

    def run_value_iteration(self):
        self.V_history = np.zeros((self.num_iteration, self.num_states))
        self.V = np.zeros(self.num_states)
        self.V[self.win_state] = 1.
        self.V[self.lose_state] = -1.
        for i in range(self.num_iteration):
            for s in range(self.num_states):
                if (s!=self.win_state) and (s!=self.lose_state):
                    self.V[s] = max([self.reward + self.gamma * \
                                    sum([self.P[s, a, s1] * self.V[s1] \
                                         for s1 in range(self.num_states)]) \
                                    for a in range(self.num_actions)])
            self.V_history[i] = self.V

        # find Q
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.Q[self.win_state, :] = 1.
        self.Q[self.lose_state, :] = -1.
        for s in range(self.num_states):
            if (s!=self.win_state) and (s!=self.lose_state):
                for a in range(self.num_actions):
                    self.Q[s, a] = self.reward + self.gamma * \
                                   sum([self.P[s, a, s1] * self.V[s1] \
                                        for s1 in range(self.num_states)])

        # find optimal policy and update self.policy
        # 0 ---> [1 0 0 0]
        # 1 ---> [0 1 0 0]
        # 2 ---> [0 0 1 0]
        # 3 ---> [0 0 0 1]
        self.optimal_policy = ops.one_hot_encode(np.argmax(self.Q, axis=1), self.num_actions)

        return self.V, self.V_history, self.Q, self.optimal_policy
    
    def plot(self):
        fig = plt.figure(figsize=(16, 12))
        fig.subplots_adjust(hspace=0.1, wspace=0.3)
        fig.suptitle("Iterative_Policy_Evaluation for V", fontsize=16)

        for s in range(self.num_states):
            ax = fig.add_subplot(3, 4, s + 1)
            ax.plot(self.V_history[:, s], label="state {}".format(str(s)))
            ax.legend(loc='upper right')
            ax.set_ylim([-1.1, 1.1])

        plt.show()


class VI_Q(ENVIRONMENT):
    def __init__(self, battery_consumption=-0.02, gamma=0.99, num_iteration=10):
        super().__init__(battery_consumption=battery_consumption, gamma=gamma)
        self.num_iteration = int(num_iteration)

    def run_value_iteration(self):
        self.Q_history = np.zeros((self.num_iteration, self.num_states, self.num_actions))
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.Q[self.win_state, :] = 1.
        self.Q[self.lose_state, :] = -1.
        for i in range(self.num_iteration):
            for s in range(self.num_states):
                if (s!=self.win_state) and (s!=self.lose_state):
                    for a in range(self.num_actions):
                        self.Q[s, a] = self.reward + self.gamma * \
                                       sum([self.P[s, a, s1] * \
                                            max([self.Q[s1, a1] \
                                                 for a1 in range(self.num_actions)]) \
                                            for s1 in range(self.num_states)])
            self.Q_history[i] = self.Q

        # find optimal policy and update self.policy
        self.optimal_policy = ops.one_hot_encode(np.argmax(self.Q, axis=1), self.num_actions)

        return self.Q, self.Q_history, self.optimal_policy
    
    def plot(self):
        fig = plt.figure(figsize=(16, 12))
        fig.subplots_adjust(hspace=0.1, wspace=0.3)
        fig.suptitle("Iterative_Policy_Evaluation for Q", fontsize=16)

        for s in range(self.num_states):
            for a in range(self.num_actions):
                ax = fig.add_subplot(11, 4, 4*s + a + 1)
                ax.plot(self.Q_history[:, s, a], \
                        label="state {}, action {}".format(str(s), str(a)))
                ax.legend(loc='lower right')
                ax.set_ylim([-1.1, 1.1])

        plt.show()
    
    
class PI(ENVIRONMENT):
    def __init__(self, policy, battery_consumption=-0.01, gamma=0.99, num_iteration=10):
        super().__init__(policy=policy, battery_consumption=battery_consumption, gamma=gamma)
        self.num_iteration = int(num_iteration)

    def run_value_iteration(self):
        self.Q_history = np.zeros((self.num_iteration, self.num_states, self.num_actions))
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.Q[self.win_state, :] = 1
        self.Q[self.lose_state, :] = -1
        for i in range(self.num_iteration):
            for s in range(self.num_states):
                if (s!=self.win_state) and (s!=self.lose_state):
                    for a in range(self.num_actions):
                        self.Q[s, a] = self.reward + self.gamma * \
                                       sum([self.P[s, a, s1] * \
                                            sum([self.policy[s1, a1] * self.Q[s1, a1] \
                                                 for a1 in range(self.num_actions)]) \
                                            for s1 in range(self.num_states)])
            self.Q_history[i] = self.Q

        return self.Q, self.Q_history

    def run_policy_iteration(self):
        self.policy_history = np.zeros((self.num_iteration, self.num_states)).astype(np.int32)

        for i in range(self.num_iteration):

            # policy evaluation
            self.Q, _ = self.run_value_iteration()  # (11,4)

            # policy improvement
            m = np.argmax(self.Q, axis=1).astype(np.int32)  # (11,)
            self.policy = ops.one_hot_encode(m, self.num_actions).astype(np.float32) # (11,4)
            self.policy_history[i, :] = m  

        return self.policy, self.policy_history

    def plot(self):
        fig = plt.figure(figsize=(16, 12))
        fig.subplots_adjust(hspace=0.1, wspace=0.3)
        fig.suptitle("Policy Iteration", fontsize=16)

        for s in range(self.num_states):
            ax = fig.add_subplot(3, 4, s + 1)
            ax.plot(self.policy_history[:, s], label="state {}".format(str(s)))
            ax.legend(loc='upper right')
            ax.set_yticks([0, 1, 2, 3])
            ax.set_ylabel('action')

        plt.show()