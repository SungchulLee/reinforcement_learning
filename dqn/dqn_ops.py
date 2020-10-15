import tensorflow as tf
import numpy as np


def variable_copy(from_scope, to_scope):
    """
    It assigns variables of "from_scope" to variables of "to_scope".
    :param from_scope: string, for example, Qnet.name
    :param to_scope: string, for example, Qnet_target.name
    :return:
    """

    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    # [<tf.Variable 'qnet/weight1:0' shape=(11, 4) dtype=float32_ref>]

    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    # [<tf.Variable 'target_qnet/weight1:0' shape=(11, 4) dtype=float32_ref>]

    copy_ops = [to_vars[i].assign(from_vars[i]) for i in range(len(from_vars))]

    return copy_ops


def one_hot_encode(states, num_states):
    """
    :param states: list or 1D numpy array, ie: [1,2,5,7]
    :param num_states: integer; number of states
    :return: numpy 2D array using one-hot encoding.
    """
    return np.eye(num_states)[states].astype(np.float32)


if __name__ == '__main__':
    states = [3]
    num_states = 11
    encoding = one_hot_encode(states, num_states)
    print(encoding)
    print()

    states = [3, 5]
    num_states = 11
    encoding = one_hot_encode(states, num_states)
    print(encoding)
    print()

    states = np.array([3, 5])
    num_states = 11
    encoding = one_hot_encode(states, num_states)
    print(encoding)
