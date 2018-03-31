import tensorflow as tf

from resan.utils.nn import bn_dense_layer, dropout, linear
from resan.utils.general import exp_mask_for_high_rank, mask_for_high_rank
from resan.rl_nn import reduce_data_rep_max_len


def reinforced_self_attention(
        rep_tensor, rep_mask, dep_selection, head_selection,
        hn=None, keep_unselected=True,
        scope=None, keep_prob=1., is_train=None, wd=0., activation='elu'
):
    with tf.variable_scope(scope or 'reinforced_self_attention'):
        fw_result = directional_attention_with_selections(
            rep_tensor, rep_mask, dep_selection, head_selection,
            'forward', hn, keep_unselected,
            'forward_resa', keep_prob, is_train, wd, activation
        )
        bw_result = directional_attention_with_selections(
            rep_tensor, rep_mask, dep_selection, head_selection,
            'backward', hn, keep_unselected,
            'backward_resa', keep_prob, is_train, wd, activation
        )
        return tf.concat([fw_result, bw_result], -1)


def directional_attention_with_selections(
        rep_tensor, rep_mask, dep_selection, head_selection, direction=None, hn=None, keep_unselected=True,
        scope=None, keep_prob=1., is_train=None, wd=0., activation='elu'):

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    org_ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or org_ivec

    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
        # non-linear
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        # ensure the seletion is right
        dep_selection = tf.logical_and(rep_mask, dep_selection)
        head_selection = tf.logical_and(rep_mask, head_selection)
        rep_dep_tensor, rep_dep_mask, dep_org_idx = reduce_data_rep_max_len(rep_map, dep_selection)
        rep_head_tensor,rep_head_mask, head_org_idx = reduce_data_rep_max_len(rep_map, head_selection)
        sl_dep, sl_head = tf.shape(rep_dep_tensor)[1], tf.shape(rep_head_tensor)[1]

        if keep_unselected:
            unhead_selection = tf.logical_and(rep_mask, tf.logical_not(head_selection))
            rep_unhead_tensor, rep_unhead_mask, unhead_org_idx = reduce_data_rep_max_len(rep_map, unhead_selection)
            sl_unhead = tf.shape(rep_unhead_tensor)[1]

        attn_result = tf.cond(
            tf.equal(sl_head, 0),
            lambda: tf.zeros([bs, 0, hn], tf.float32),
            lambda: self_attention_for_selected_head(
                head_selection, head_org_idx, sl_head, rep_head_mask,
                dep_selection, dep_org_idx, sl_dep, rep_dep_mask,
                rep_map, rep_dep_tensor, keep_prob, is_train, direction, ivec
            )
        )

        if keep_unselected:
            input_idx = tf.tile(tf.expand_dims(tf.range(sl), 0), [bs, 1])
            pooling_result = tf.cond(
                tf.equal(sl_unhead, 0),
                lambda: tf.zeros([bs, 0, hn], tf.float32),
                lambda: mean_pooling_for_unselected_head(
                    unhead_org_idx, sl_unhead, rep_unhead_mask,
                    input_idx, sl, rep_mask, rep_map, None)  # todo: point !
            )

        with tf.variable_scope('output'):
            if keep_unselected:
                range_head = tf.tile(tf.expand_dims(tf.range(bs), -1), [1, sl_head])
                scatter_attn = tf.cond(
                    tf.equal(sl_head, 0),
                    lambda: tf.zeros([bs, sl+1, hn], tf.float32),
                    lambda: tf.scatter_nd(
                        tf.stack([range_head, head_org_idx], -1), attn_result, [bs, sl+1, hn])
                )

                range_unhead = tf.tile(tf.expand_dims(tf.range(bs), -1), [1, sl_unhead])
                scatter_pooling = tf.cond(
                    tf.equal(sl_unhead, 0),
                    lambda: tf.zeros([bs, sl+1, hn], tf.float32),
                    lambda: tf.scatter_nd(
                        tf.stack([range_unhead, unhead_org_idx], -1), pooling_result, [bs, sl+1, hn])
                )

                self_attn_input = rep_map
                context_features = tf.add(scatter_attn[:, :-1], scatter_pooling[:, :-1], 'context_features')
                output_mask = rep_mask
            else:
                self_attn_input = rep_head_tensor
                context_features = attn_result
                output_mask = rep_head_mask

            # context fusion gate
            o_bias = tf.get_variable('o_bias', [ivec], tf.float32, tf.constant_initializer(0.))
            fusion_gate = tf.nn.sigmoid(
                linear(self_attn_input, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                linear(context_features, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * self_attn_input + (1 - fusion_gate) * context_features

        return output, output_mask


def self_attention_for_selected_head(
        head_selection, head_org_idx, sl_head, rep_head_mask,
        dep_selection, dep_org_idx, sl_dep, rep_dep_mask,
        rep_map, rep_dep_tensor, keep_prob, is_train, direction, ivec
):
    # data for self-attention
    rep_map_dp = dropout(rep_map, keep_prob, is_train)
    rep_dep_tensor_dp, _, _ = reduce_data_rep_max_len(rep_map_dp, dep_selection)
    rep_head_tensor_dp, _, _ = reduce_data_rep_max_len(rep_map_dp, head_selection)

    # mask generation
    dep_idxs = tf.tile(tf.expand_dims(dep_org_idx, 1), [1, sl_head, 1])
    head_idxs = tf.tile(tf.expand_dims(head_org_idx, 2), [1, 1, sl_dep])

    if direction is None:
        direct_mask = tf.not_equal(head_idxs, dep_idxs)  # [bs, slh, sld]
    else:
        if direction == 'forward':
            direct_mask = tf.greater(head_idxs, dep_idxs)  # [bs, slh, sld]
        else:
            direct_mask = tf.less(head_idxs, dep_idxs)  # [bs, slh, sld]
    # [bs, slh, slh]
    rep_mask_tile = tf.logical_and(tf.expand_dims(rep_dep_mask, 1), tf.expand_dims(rep_head_mask, 2))
    attn_mask = tf.logical_and(direct_mask, rep_mask_tile)  # [bs, slh, sld]

    # tensor tile
    rep_map_tile = tf.tile(tf.expand_dims(rep_dep_tensor, 1), [1, sl_head, 1, 1])  # bs,slh,sld,vec
    with tf.variable_scope('attention'):  # bs,sl,sl,vec
        f_bias = tf.get_variable('f_bias', [ivec], tf.float32, tf.constant_initializer(0.))
        dependent = linear(rep_dep_tensor_dp, ivec, False, scope='linear_dependent')  # bs,sld,vec
        dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sld,vec
        head = linear(rep_head_tensor_dp, ivec, False, scope='linear_head')  # bs,slh,vec
        head_etd = tf.expand_dims(head, 2)  # bs,slh,1,vec

        logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)  # bs,slh,sld,vec
        logits_masked = exp_mask_for_high_rank(logits, attn_mask)  # bs,slh,sld,vec
        attn_score = tf.nn.softmax(logits_masked, 2)  # bs,slh,sld,vec
        attn_score = mask_for_high_rank(attn_score, attn_mask)
        attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,slh,vec -> head_org_idx
    return attn_result


def mean_pooling_for_unselected_head(
        unhead_org_idx, sl_unhead, rep_unhead_mask,
        dep_org_idx, sl_dep, rep_dep_mask,
        rep_dep_tensor, direction
):
    with tf.name_scope('pooling_for_un_head'):
        undep_idxs = tf.tile(tf.expand_dims(dep_org_idx, 1), [1, sl_unhead, 1])  # [bs, sluh, sld]
        unhead_idxs = tf.tile(tf.expand_dims(unhead_org_idx, 2), [1, 1, sl_dep])  # [bs, sluh, sld]
        if direction is None:
            direct_mask_un = tf.not_equal(unhead_idxs, undep_idxs)  # [bs, sluh, sld]
        else:
            if direction == 'forward':
                direct_mask_un = tf.greater(unhead_idxs, undep_idxs)  # [bs, sluh, sld]
            else:
                direct_mask_un = tf.less(unhead_idxs, undep_idxs)  # [bs, sluh, sld]

        # [bs, sluh, sld]
        rep_mask_tile_un = tf.logical_and(tf.expand_dims(rep_dep_mask, 1), tf.expand_dims(rep_unhead_mask, 2))
        pooling_mask = tf.logical_and(direct_mask_un, rep_mask_tile_un)  # [bs, sluh, sld]

        # data for pooling
        pooling_data = tf.tile(tf.expand_dims(rep_dep_tensor, 1), [1, sl_unhead, 1, 1])  # bs,sluh,sld,hn
        # execute mean pooling based on pooling_mask[bs, sluh, sld] and pooling_data[bs,sluh,sld,hn]
        pooling_data = mask_for_high_rank(pooling_data, pooling_mask)  # [bs,sluh,sld,hn]
        pooling_data_sum = tf.reduce_sum(pooling_data, -2)  # [bs,sluh,hn]
        pooling_den = tf.reduce_sum(tf.cast(pooling_mask, tf.int32), -1, keep_dims=True)  # [bs,sluh]
        pooling_den = tf.where(tf.equal(pooling_den, 0), tf.ones_like(pooling_den), pooling_den)

        pooling_result = pooling_data_sum / tf.cast(pooling_den, tf.float32)
        return pooling_result


def scaled_tanh(x, scale=5.):
    return scale * tf.nn.tanh(1./scale * x)