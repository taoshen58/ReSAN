import tensorflow as tf
from resan.utils.nn import bn_dense_layer
from resan.utils.general import exp_mask_for_high_rank
from resan.resa import reinforced_self_attention


def reinforced_self_attention_network(
        rep_tensor, rep_mask, dep_selection, head_selection, direction=None, hn=None, keep_unselected=True,
        scope=None, keep_prob=1., is_train=None, wd=0., activation='elu'
):
    with tf.variable_scope(scope or 'reinforced_self_attention_network'):
        resa_result = reinforced_self_attention(
            rep_tensor, rep_mask, dep_selection, head_selection, hn, keep_unselected,
            'reinforced_self_attention', keep_prob, is_train, wd, activation)
        output = multi_dimensional_attention(
            resa_result, rep_mask, 'multi_dim_attn', keep_prob, is_train, wd, activation
        )
        return output


def multi_dimensional_attention(rep_tensor, rep_mask, scope=None,
                                keep_prob=1., is_train=None, wd=0., activation='elu',
                                tensor_dict=None, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'multi_dimensional_attention'):
        map1 = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1', activation,
                              False, wd, keep_prob, is_train)
        map2 = bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', 'linear',
                              False, wd, keep_prob, is_train)
        map2_masked = exp_mask_for_high_rank(map2, rep_mask)

        soft = tf.nn.softmax(map2_masked, 1)  # bs,sl,vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs, vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = soft

        return attn_output


