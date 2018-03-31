import tensorflow as tf

from resan.utils.nn import bn_dense_layer
from resan.utils.general import VERY_NEGATIVE_NUMBER


def binary_entropy(probs, labels):
    with tf.name_scope('binary_entropy'):
        labels = tf.cast(labels, tf.float32)
        return - (labels * tf.log(tf.clip_by_value(probs, 1e-10, 1.0)) +
                  (1.0 - labels) * tf.log(tf.clip_by_value(1.0 - probs, 1e-10, 1.0)))


def sequence_conditional_feature(rep_tensor, rep_mask):
    rep_tensor_pooling = pooling_along_time(rep_tensor, rep_mask, 'mean', True, True)
    rep_tensor_new = tf.concat([rep_tensor, rep_tensor_pooling, rep_tensor * rep_tensor_pooling], -1)
    return rep_tensor_new


def sequence_conditional_feature_v2(rep_tensor, rep_mask):
    rep_tensor_pooling = pooling_along_time(rep_tensor, rep_mask, 'mean', True, True)
    rep_tensor_new = tf.concat([rep_tensor, rep_tensor * rep_tensor_pooling], -1)
    return rep_tensor_new


def generate_mask_with_rl(rep_tensor, rep_mask, is_mat=False, scope=None,
                          keep_prob=1., is_train=None, wd=0., activation='elu',
                          disable_rl=False, global_step=None, mode='train', start_only_rl=0, hn=300):
    scope = scope or 'generate_mask_with_rl'
    with tf.variable_scope(scope):
        disable_rl_tf = tf.logical_and(tf.logical_or(
            tf.constant(disable_rl, tf.bool, [], name='disable_rl'),
            tf.less_equal(global_step, start_only_rl)
        ), tf.constant(mode == 'train', tf.bool, []))

        logpa, actions, percentage = \
            tf.cond(disable_rl_tf,
                    lambda: generate_mask_with_rl_fake(rep_tensor, rep_mask, is_mat, scope+'_fake'),
                    lambda: generate_mask_with_rl_real(rep_tensor, rep_mask, is_mat, scope+'_real',
                                                       keep_prob, is_train, wd, activation, hn)
                    )
        return logpa, actions, percentage


def generate_mask_with_rl_real(rep_tensor, rep_mask, is_mat=False,scope=None,
                               keep_prob=1., is_train=None, wd=0., activation='elu', hn=None):
    """

    :param rep_tensor: 3d tensor
    :param rep_mask: 2d tensor
    :param is_mat: [True|False]
    :param start_rl:
    :param end_rl_gain:
    :param scope:
    :param keep_prob:
    :param is_train:
    :param wd:
    :param activation:
    :param global_step:
    :return:
    """

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = hn or rep_tensor.get_shape()[2]

    with tf.variable_scope(scope or 'generate_mask_with_rl_real'):
        if is_mat:
            rep_row = tf.tile(tf.expand_dims(rep_tensor, 1), [1, sl, 1, 1])
            rep_col = tf.tile(tf.expand_dims(rep_tensor, 2), [1, 1, sl, 1])
            rep_h0 = tf.concat([rep_row, rep_col], -1)
        else:
            rep_h0 = rep_tensor

        rep_h1 = bn_dense_layer([rep_h0], ivec, True, 0., 'dense_rep_mat_h1',
                                    activation, False, wd, keep_prob, is_train)
        rep_h2 = bn_dense_layer([rep_h1], 1, True, 0., 'dense_rep_mat_h2',
                                    'linear', False, wd, keep_prob, is_train)
        rep_h2 = tf.squeeze(rep_h2, 3 if is_mat else 2)  # bs,sl,sl / bs,sl
        rep_prob = tf.nn.sigmoid(rep_h2)

        # sampling
        # Here, need a dynamic policy to add the random

        # todo:text
        # mode_is_train = tf.constant(mode == 'train', tf.bool, [], 'mode_is_train')
        # random_values = tf.cond(
        #     tf.logical_and(mode_is_train, is_train),
        #     lambda: tf.random_uniform([bs, sl, sl] if is_mat else [bs, sl]),
        #     lambda: tf.ones([bs, sl, sl] if is_mat else [bs, sl], tf.float32) * 0.5
        # )
        random_values = tf.random_uniform([bs, sl, sl] if is_mat else [bs, sl])

        # if global_step is not None:
        #     policy_rep_prob = tf.cond(tf.logical_and(mode_is_train,
        #                                              tf.less(global_step,
        #                                                      tf.constant(int(x2), tf.int32))),
        #                               lambda: rep_prob + prob_gain,
        #                               lambda: rep_prob)
        #
        # else:
        #     policy_rep_prob = rep_prob

        policy_rep_prob = rep_prob

        actions = tf.less_equal(random_values, policy_rep_prob)

        actions = tf.stop_gradient(actions)

        if is_mat:
            rep_mask_new = tf.logical_and(
                tf.expand_dims(rep_mask, 1),
                tf.expand_dims(rep_mask, 2)
            )
        else:
            rep_mask_new = rep_mask

        actions = tf.logical_and(actions, rep_mask_new)

        # log p(a)
        logpa = - binary_entropy(rep_prob, actions) * tf.cast(rep_mask_new, tf.float32)
        if is_mat:
            logpa = - tf.reshape(logpa, [bs, sl * sl])

        # percentage
        actions_flat = tf.reshape(actions, [bs, -1])
        rep_mask_new_flat = tf.reshape(rep_mask_new, [bs, -1])

        percentage = tf.reduce_sum(tf.cast(actions_flat, tf.float32), -1) / \
                     tf.reduce_sum(tf.cast(rep_mask_new_flat, tf.float32), -1)

        return logpa, actions, percentage


def generate_mask_with_rl_fake(rep_tensor, rep_mask, is_mat=False, scope=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    with tf.name_scope(scope or 'generate_mask_with_rl_fake'):
        logpa = tf.ones([bs, sl*sl] if is_mat else [bs, sl], tf.float32)
        if is_mat:
            actions = tf.logical_and(
                tf.expand_dims(rep_mask, 1),
                tf.expand_dims(rep_mask, 2)
            )
        else:
            actions = rep_mask

        percentage = tf.ones([bs], tf.float32)

        return logpa, actions, percentage


def reduce_data_rep_max_len(rep_tensor, data_filter, scope=None):
    with tf.name_scope(scope or 'reduce_data_rep_max_len'):
        # lens: batch_size, original_max_len, hiddn_dim
        bs, oml, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
        ivec = rep_tensor.get_shape().as_list()[2]  # hidden_dim_int
        lens = tf.reduce_sum(tf.cast(data_filter, tf.int32), -1)  # [bs] new lengths for each sample
        nml = tf.reduce_max(lens) # new_max_len
        compensate_lens = nml - lens  # [bs] number of padding(False in mask) for each sample

        range = tf.tile(tf.expand_dims(tf.range(nml, 0, -1), 0), [bs, 1])
        range = range - tf.tile(tf.expand_dims(compensate_lens, 1), [1, nml])
        mask_new = tf.greater(range, 0)  # new mask for data
        # -- generate new data --
        where_old = tf.where(data_filter)
        where_new = tf.where(mask_new)
        gather = tf.gather_nd(rep_tensor, where_old)
        scatter = tf.scatter_nd(indices=where_new, updates=gather, shape=[tf.cast(bs, tf.int64),
                                                                          tf.cast(nml, tf.int64),
                                                                          ivec])
        # old index
        old_idx = tf.tile(tf.expand_dims(tf.range(oml), 0), [bs, 1])
        gather_old_idx = tf.gather_nd(old_idx, where_old)
        scatter_old_idx = tf.scatter_nd(indices=where_new, updates=gather_old_idx,
                                        shape=[tf.cast(bs, tf.int64), tf.cast(nml, tf.int64)])
        scatter_old_idx = tf.where(
            mask_new,
            scatter_old_idx,
            - tf.ones_like(scatter_old_idx, tf.int32)
        )

        return scatter, mask_new, scatter_old_idx





# ---- back up -----
def generate_mask_with_rl_real_bk(rep_tensor, rep_mask, is_mat=False, start_rl=10000, end_rl_gain=20000, scope=None,
                               keep_prob=1., is_train=None, wd=0., activation='elu',
                               global_step=None, mode='train'):
    """

    :param rep_tensor: 3d tensor
    :param rep_mask: 2d tensor
    :param is_mat: [True|False]
    :param start_rl:
    :param end_rl_gain:
    :param scope:
    :param keep_prob:
    :param is_train:
    :param wd:
    :param activation:
    :param global_step:
    :return:
    """
    x1 = start_rl
    x2 = end_rl_gain

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]

    with tf.variable_scope(scope or 'generate_mask_with_rl_real'):
        if is_mat:
            rep_row = tf.tile(tf.expand_dims(rep_tensor, 1), [1, sl, 1, 1])
            rep_col = tf.tile(tf.expand_dims(rep_tensor, 2), [1, 1, sl, 1])
            rep_h0 = tf.concat([rep_row, rep_col], -1)
        else:
            rep_h0 = rep_tensor

        rep_h1 = bn_dense_layer([rep_h0], ivec, True, 0., 'dense_rep_mat_h1',
                                    activation, False, wd, keep_prob, is_train)
        rep_h2 = bn_dense_layer([rep_h1], 1, True, 0., 'dense_rep_mat_h2',
                                    'linear', False, wd, keep_prob, is_train)
        rep_h2 = tf.squeeze(rep_h2, 3 if is_mat else 2)  # bs,sl,sl / bs,sl
        rep_prob = tf.nn.sigmoid(rep_h2)

        # sampling
        # Here, need a dynamic policy to add the random
        random_values = tf.random_uniform([bs, sl, sl] if is_mat else [bs, sl])
        mode_is_train = tf.constant(mode == 'train', tf.bool, [], 'mode_is_train')

        assert x1 <= x2

        if global_step is not None:
            prob_gain = 1. if x1 == x2 else tf.nn.relu(tf.cast(global_step, tf.float32)/(x1 - x2) +
                                                       (x2 / (x2 - x1)))

            policy_rep_prob = tf.cond(tf.logical_and(mode_is_train,
                                                     tf.less(global_step,
                                                             tf.constant(int(x2), tf.int32))),
                                      lambda: rep_prob + prob_gain,
                                      lambda: rep_prob)

        else:
            policy_rep_prob = rep_prob

        actions = tf.less_equal(random_values, policy_rep_prob)
        actions = tf.stop_gradient(actions)

        if is_mat:
            rep_mask_new = tf.logical_and(
                tf.expand_dims(rep_mask, 1),
                tf.expand_dims(rep_mask, 2)
            )
        else:
            rep_mask_new = rep_mask

        actions = tf.logical_and(actions, rep_mask_new)

        # log p(a)
        logpa = - binary_entropy(rep_prob, actions) * tf.cast(rep_mask_new, tf.float32)
        if is_mat:
            logpa = - tf.reshape(logpa, [bs, sl * sl])

        # percentage
        actions_flat = tf.reshape(actions, [bs, -1])
        rep_mask_new_flat = tf.reshape(rep_mask_new, [bs, -1])

        percentage = tf.reduce_sum(tf.cast(actions_flat, tf.float32), -1) / \
                     tf.reduce_sum(tf.cast(rep_mask_new_flat, tf.float32), -1)

        return logpa, actions, percentage


def pooling_along_time(rep_tensor, rep_mask, pooling_method='mean',
                       keep_dim=False, keep_len=False, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    with tf.name_scope(name or '%s_pooling_along_time' % pooling_method):
        rep_mask_tiled = tf.tile(tf.expand_dims(rep_mask, 2), [1, 1, ivec])
        denominator = tf.reduce_sum(tf.cast(rep_mask, tf.int32), 1, keep_dims=True)  # bs, 1
        denominator_tiled = tf.tile(denominator, [1, ivec])
        if pooling_method == 'mean':
            rep_tensor_zero = tf.where(rep_mask_tiled, rep_tensor, tf.zeros_like(rep_tensor))  # bs,sl,vec
            rep_sum = tf.reduce_sum(rep_tensor_zero, 1)  # bs, vec
            denominator = tf.where(tf.equal(denominator, 0), tf.ones_like(denominator), denominator)
            res = rep_sum / tf.cast(denominator, tf.float32)
        elif pooling_method == 'max':
            rep_tensor_ng = tf.where(rep_mask_tiled, rep_tensor, tf.ones_like(rep_tensor) * VERY_NEGATIVE_NUMBER)
            rep_max = tf.reduce_max(rep_tensor_ng, 1) # bs, vdc
            res = tf.where(tf.equal(denominator_tiled, 0), tf.zeros_like(rep_max), rep_max)
        else:
            raise RuntimeError('no pooling method %s' % pooling_method)

        if keep_dim:
            res = tf.expand_dims(res, 1)
            if keep_len:
                res = tf.tile(res, [1, sl, 1])
        return res