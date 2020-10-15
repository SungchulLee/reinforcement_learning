import tensorflow as tf


class QNet:
    def __init__(self, num_states=11, num_actions=4, name=None):
        self._num_states = int(num_states)
        self._num_actions = int(num_actions) 
        self._name = name

    def __call__(self, inputs, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        return self._q_graph(inputs)

    def _q_graph(self, inputs):
        w1 = tf.get_variable(name='weight1',
                             shape=[self._num_states, self._num_actions],
                             dtype=tf.float32,
                             initializer=tf.random_uniform_initializer(0.0, 1.0))

        outputs = tf.matmul(inputs, w1)
        return outputs

    @property
    def name(self):
        return self._name