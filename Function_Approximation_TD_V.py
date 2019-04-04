import numpy as np
import tensorflow as tf

# set parameters ###############################################################
epoch = 10000
lr_rate = 1e-2
copy_period = 1
# set parameters ###############################################################

# state
states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_STATES = len(states)

# action
actions = [0, 1, 2, 3]
N_ACTIONS = len(actions)

# transition probabilities
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

# rewards
R = -0.02 * np.ones((N_STATES, N_ACTIONS))
R[3, :] = 1.
R[6, :] = -1.

# discount factor
gamma = 0.99

# policy
if 0:
    # bad policy
    policy = np.empty((N_STATES, N_ACTIONS))
    policy[0, :] = [0, 1, 0, 0]
    policy[1, :] = [0, 1, 0, 0]
    policy[2, :] = [0, 1, 0, 0]
    policy[3, :] = [0, 1, 0, 0]
    policy[4, :] = [0, 0, 0, 1]
    policy[5, :] = [0, 1, 0, 0]
    policy[6, :] = [0, 1, 0, 0]
    policy[7, :] = [0, 1, 0, 0]
    policy[8, :] = [0, 1, 0, 0]
    policy[9, :] = [0, 0, 1, 0]
    policy[10, :] = [0, 0, 1, 0]
elif 0:
    # random policy
    policy = 0.25 * np.ones((N_STATES, N_ACTIONS))
elif 0:
    # optimal policy
    policy = np.empty((N_STATES, N_ACTIONS))
    policy[0, :] = [0, 1, 0, 0]
    policy[1, :] = [0, 1, 0, 0]
    policy[2, :] = [0, 1, 0, 0]
    policy[3, :] = [0, 1, 0, 0]
    policy[4, :] = [0, 0, 1, 0]
    policy[5, :] = [0, 0, 1, 0]
    policy[6, :] = [0, 0, 1, 0]
    policy[7, :] = [0, 0, 1, 0]
    policy[8, :] = [1, 0, 0, 0]
    policy[9, :] = [1, 0, 0, 0]
    policy[10, :] = [1, 0, 0, 0]
elif 1:
    # optimal policy + noise
    # we use optimal policy with probability 1/(1+ep)
    # we use random policy with probability ep/(1+ep)
    ep = 0.1
    policy = np.empty((N_STATES, N_ACTIONS))
    policy[0, :] = [0, 1, 0, 0]
    policy[1, :] = [0, 1, 0, 0]
    policy[2, :] = [0, 1, 0, 0]
    policy[3, :] = [0, 1, 0, 0]
    policy[4, :] = [0, 0, 1, 0]
    policy[5, :] = [0, 0, 1, 0]
    policy[6, :] = [0, 0, 1, 0]
    policy[7, :] = [0, 0, 1, 0]
    policy[8, :] = [1, 0, 0, 0]
    policy[9, :] = [1, 0, 0, 0]
    policy[10, :] = [1, 0, 0, 0]
    policy = policy + (ep / 4) * np.ones((N_STATES, N_ACTIONS))
    policy = policy / np.sum(policy, axis=1).reshape((N_STATES, 1))

" TD "

state = tf.placeholder(tf.int32, shape=[None], name='state')
action = tf.placeholder(tf.int32, shape=[None], name='action')
reward = tf.placeholder(tf.float32, shape=[None], name='reward')
next_state = tf.placeholder(tf.int32, shape=[None], name='next_state')

# https://www.tensorflow.org/api_docs/python/tf/one_hot
state_one_hot = tf.one_hot(state, N_STATES)
action_one_hot = tf.one_hot(action, N_ACTIONS)
next_state_one_hot = tf.one_hot(next_state, N_STATES)

# Previously
# V = np.zeros(N_STATES)
# V[3] = 1.
# V[6] = -1.

# state s: one hot encode
# V[s]: [s as one hot encode] @ W

with tf.variable_scope('main_net') as scope:
    W = tf.get_variable(name='W', \
                        shape=[N_STATES, 1], \
                        dtype=tf.float32, \
                        initializer=tf.random_uniform_initializer(-1.0, 1.0))

with tf.variable_scope('target_net') as scope:
    W_target = tf.get_variable(name='W_target', \
                               shape=[N_STATES, 1], \
                               dtype=tf.float32, \
                               initializer=tf.random_uniform_initializer(-1.0, 1.0))

# variable copy
from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'main_net')
# print(from_vars)
# [<tf.Variable 'main_net/W:0' shape=(11, 1) dtype=float32_ref>]

to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_net')
# print(to_vars)
# [<tf.Variable 'target_net/W_target:0' shape=(11, 1) dtype=float32_ref>]

copy_ops = [to_vars[i].assign(from_vars[i]) for i in range(len(from_vars))]

# TD
# V(S_t)\leftarrow V(S_t)+\alpha(R_{t+1}+\gamma V(S_{t+1})-V(S_t))
# V[s] += alpha * (R[s,a] + gamma * V[s1] - V[s])

# TD target and TD error
TD_target = reward + gamma * tf.matmul(next_state_one_hot, W_target)
V = tf.matmul(state_one_hot, W)
TD_error = TD_target - V

loss = tf.reduce_mean(tf.square(TD_error))
opt = tf.train.GradientDescentOptimizer(learning_rate=lr_rate)
train_ops = opt.minimize(loss, \
                         var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                                    'main_net'))

" Session "
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    gradient_update_number = 0

    for _ in range(epoch):

        # indicate game is not over yet
        done = False

        # choose initial state randomly, not from 3 or 6
        s = np.random.choice([0, 1, 2, 4, 5, 7, 8, 9, 10])

        while not done:

            if gradient_update_number % copy_period == 0:
                sess.run(copy_ops)

            # choose action using current policy
            a = np.random.choice(actions, p=policy[s, :])

            # choose next state using transition probabilities
            s1 = np.random.choice(states, p=P[s, a, :])

            feed_dict = {state: [s], action: [a], reward: [R[s, a]], next_state: [s1]}
            sess.run(train_ops, feed_dict=feed_dict)
            # _, loss_now = sess.run([train_ops, loss], feed_dict=feed_dict)
            # print(loss_now)

            gradient_update_number += 1

            if (s1 == 3) or (s1 == 6):
                # if game is over,
                # ready to break while loop by letting done = True
                done = True
            else:
                # if game is not over, continue playing game
                s = s1

    V_final = []
    for s in range(N_STATES):
        feed_dict = {state: [s]}
        V_now = sess.run(V, feed_dict=feed_dict)
        V_final.append(V_now[0][0])
    print(V_final)