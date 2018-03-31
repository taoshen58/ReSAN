import tensorflow as tf
from src.nn_utils.general import VERY_NEGATIVE_NUMBER


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