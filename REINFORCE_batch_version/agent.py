import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from episode_memory import Episode


class Agent():

    def __init__(self, sess, config, env):
        self.sess = sess
        self.config = config
        self.env = env
        self.ep = Episode(config)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self._build_net()

    def print_pi(self):
        for i in range(11):
            s = np.zeros(11)
            s[i] = 1
            pi = self.sess.run(self.pi, {self.S_in: [s]})[0]
            print(pi)

    def train(self):
        self.print_pi()
        rewards = []
        avg_n = 1
        for i in range(self.config.max_ep):
            if i % avg_n == 0:
                total = 0
            ep_reward = 0.
            done = False
            state = self.env.initialize_game()
            self.ep.initialize(state)
            while not done:
                action = self.choose_action()
                state, reward, done = self.env.act(action)
                ep_reward += reward
                self.after_act(action, reward, state, done)
            total += ep_reward
            if i % avg_n == 0:
                rewards.append(total / avg_n)
            print("Episode: {}, Reward: {}".format(i, ep_reward))
        self.print_pi()
        plt.plot(rewards)
        plt.show()

    def after_act(self, action, reward, state, done):
        self.ep.add(action, reward, state)
        if done:
            self.update_params()

    def update_params(self):
        S, A, V = self.ep.get_batch()
        self.sess.run(self.train_op, {self.S_in: S, self.A_in: A, self.V_in: V})

    def choose_action(self):
        pi = self.sess.run(self.pi, {self.S_in: [self.env.state]})[0]
        a = np.random.choice(self.env.action_space, p=pi)
        # a = [1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 0][self.env._state]
        return a

    def _build_net(self):
        with tf.variable_scope("policy"):
            self.S_in = tf.placeholder(
                tf.float32, (None,) + self.config.input_shape, name="state")
            self.pi = self._fc_net(self.S_in)

        self.loss, self.train_op = self._loss_and_train_op()

        self.sess.run(tf.global_variables_initializer())
        print("Network Built")

    def _fc_net(self, s):
        initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        activation_fn = tf.nn.relu

        h = s
        for i, dim in enumerate(self.config.fc_archi):
            h = tf.layers.dense(
                inputs=h,
                units=dim,
                activation=activation_fn,
                kernel_initializer=initializer,
                name="fc{}".format(i + 1))
        h = tf.layers.dense(
            inputs=h,
            units=self.env.action_space_size,
            kernel_initializer=initializer)
        pi = tf.nn.softmax(h, name="pi")
        return pi

    def _loss_and_train_op(self):
        with tf.variable_scope("optimizer"):
            self.A_in = tf.placeholder(tf.int32, (None,), name="action")
            a_one_hot = tf.one_hot(
                self.A_in, self.env.action_space_size, 1., 0., name="a_one_hot")
            self.V_in = tf.placeholder(tf.float32, (None,), name="return")
            self.pi_s_a = tf.reduce_sum(
                tf.multiply(a_one_hot, self.pi), axis=1, name="pi_s_a")

            loss = tf.reduce_sum(-self.V_in * tf.log(self.pi_s_a), name="loss")

            train_op = tf.train.GradientDescentOptimizer(
                self.config.lr).minimize(loss)

        return loss, train_op