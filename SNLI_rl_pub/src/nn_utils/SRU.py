import tensorflow as tf
from src.nn_utils.nn import bn_dense_layer
from src.nn_utils.rnn import dynamic_rnn, bw_dynamic_rnn
from src.nn_utils.rnn_cell import SwitchableDropoutWrapper


def bi_sru_recurrent_network(
        rep_tensor, rep_mask, is_train=None, keep_prob=1., wd=0.,
        scope=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    with tf.variable_scope(scope or 'bi_sru_recurrent_network'):
        U_d = bn_dense_layer([rep_tensor], 6 * ivec, True, 0., 'get_frc', 'linear',
                           False, wd, keep_prob, is_train)  # bs, sl, 6vec
        U_d_fw, U_d_bw = tf.split(U_d, 2, 2)
        with tf.variable_scope('forward'):
            U_fw = tf.concat([rep_tensor, U_d_fw], -1)
            fw_SRUCell = SwitchableDropoutWrapper(SRUCell(ivec, tf.nn.tanh), is_train, keep_prob)
            fw_output, _ = dynamic_rnn(
                fw_SRUCell, U_fw, tf.reduce_sum(tf.cast(rep_mask, tf.int32), -1),
                dtype=tf.float32, scope='forward_sru')  # bs, sl, vec

        with tf.variable_scope('backward'):
            U_bw = tf.concat([rep_tensor, U_d_bw], -1)
            bw_SRUCell = SwitchableDropoutWrapper(SRUCell(ivec, tf.nn.tanh), is_train, keep_prob)
            bw_output, _ = bw_dynamic_rnn(
                bw_SRUCell, U_bw, tf.reduce_sum(tf.cast(rep_mask, tf.int32), -1),
                dtype=tf.float32, scope='backward_sru')  # bs, sl, vec

        all_output = tf.concat([fw_output, bw_output], -1)  # bs, sl, 2vec
        return all_output


class SRUCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, activation=None):
        super(SRUCell, self).__init__()
        self.num_units = num_units
        self.activation = activation or tf.nn.tanh

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, inputs, state, scope=None):
        """

        :param inputs: [bs,4*vec]
        :param state: [bs, vec]
        :return:
        """
        x_t, x_dt, f_t, r_t = tf.split(inputs, 4, 1)
        f_t = tf.nn.sigmoid(f_t)
        r_t = tf.nn.sigmoid(r_t)
        c_t = f_t * state + (1 - f_t) * x_dt
        h_t = r_t * self.activation(c_t) + (1 - r_t) * x_t
        return h_t, c_t







