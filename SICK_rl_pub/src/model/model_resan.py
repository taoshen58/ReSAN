"""
_v4
do not have direction in pooling
do not have dependents
"""

from configs import cfg
from src.utils.record_log import _logger
import tensorflow as tf

from src.model.model_template import ModelTemplate
from src.nn_utils.nn import bn_dense_layer, dropout, linear
from src.nn_utils.general import exp_mask_for_high_rank, mask_for_high_rank
from src.nn_utils.rl.nn import reduce_data_rep_max_len, sequence_conditional_feature, generate_mask_with_rl
from src.nn_utils.integration_func import generate_embedding_mat, multi_dimensional_attention
from src.nn_utils.resa import directional_attention_with_selections


class ModelResan(ModelTemplate):
    def __init__(self, token_emb_mat, glove_emb_mat, tds, cds, tl, scope):
        super(ModelResan, self).__init__(token_emb_mat, glove_emb_mat, tds, cds, tl, scope)
        self.disable_rl = False
        self.update_tensor_add_ema_and_opt()

    def build_network(self):
        _logger.add()
        _logger.add('building %s neural network structure...' % cfg.network_type)

        tds, cds = self.tds, self.cds
        tl = self.tl
        tel, cel, cos, ocd, fh = self.tel, self.cel, self.cos, self.ocd, self.fh
        hn = self.hn
        bs, sl1, sl2 = self.bs, self.sl1, self.sl2

        with tf.variable_scope('emb'):
            token_emb_mat = generate_embedding_mat(tds, tel, init_mat=self.token_emb_mat,
                                                   extra_mat=self.glove_emb_mat, extra_trainable=self.finetune_emb,
                                                   scope='gene_token_emb_mat')
            s1_emb = tf.nn.embedding_lookup(token_emb_mat, self.sent1_token)  # bs,sl1,tel
            s2_emb = tf.nn.embedding_lookup(token_emb_mat, self.sent2_token)  # bs,sl2,tel
            self.tensor_dict['s1_emb'] = s1_emb
            self.tensor_dict['s2_emb'] = s2_emb

        with tf.variable_scope('hard_network'):
            # for sentence 1
            s1_emb_new = sequence_conditional_feature(s1_emb, self.sent1_token_mask)
            s1_logpa_dep, s1_act_dep, s1_percentage_dep = generate_mask_with_rl(
                s1_emb_new, self.sent1_token_mask, False, 'generate_mask_with_rl_dep',
                cfg.dropout, self.is_train, cfg.wd, 'relu',
                self.disable_rl, self.global_step, cfg.mode, cfg.start_only_rl, hn
            )  # [bs, sl] & [bs, sl]
            s1_logpa_head, s1_act_head, s1_percentage_head = generate_mask_with_rl(
                s1_emb_new, self.sent1_token_mask, False, 'generate_mask_with_rl_head',
                cfg.dropout, self.is_train, cfg.wd, 'relu',
                self.disable_rl, self.global_step, cfg.mode, cfg.start_only_rl, hn
            )  # [bs, sl] & [bs, sl]
            s1_logpa = tf.concat([s1_logpa_dep, s1_logpa_head], -1)
            s1_act = tf.logical_and(tf.expand_dims(s1_act_dep, 1), tf.expand_dims(s1_act_head, 2))
            s1_percentage = s1_percentage_dep * s1_percentage_head

            tf.get_variable_scope().reuse_variables()
            # for sentence 2
            s2_emb_new = sequence_conditional_feature(s2_emb, self.sent2_token_mask)
            s2_logpa_dep, s2_act_dep, s2_percentage_dep = generate_mask_with_rl(
                s2_emb_new, self.sent2_token_mask, False, 'generate_mask_with_rl_dep',
                cfg.dropout, self.is_train, cfg.wd, 'relu',
                self.disable_rl, self.global_step, cfg.mode, cfg.start_only_rl, hn
            )  # [bs, sl] & [bs, sl]
            s2_logpa_head, s2_act_head, s2_percentage_head = generate_mask_with_rl(
                s2_emb_new, self.sent2_token_mask, False, 'generate_mask_with_rl_head',
                cfg.dropout, self.is_train, cfg.wd, 'relu',
                self.disable_rl, self.global_step, cfg.mode, cfg.start_only_rl, hn
            )  # [bs, sl] & [bs, sl]
            s2_logpa = tf.concat([s2_logpa_dep, s2_logpa_head], -1)
            s2_act = tf.logical_and(tf.expand_dims(s2_act_dep, 1), tf.expand_dims(s2_act_head, 2))
            s2_percentage = s2_percentage_dep * s2_percentage_head

        keep_unselected = True
        with tf.variable_scope('ct_attn'):
            s1_fw, s1_token_mask_new = directional_attention_with_selections(
                s1_emb, self.sent1_token_mask, s1_act_dep, s1_act_head,'forward', hn, keep_unselected,
                'dir_attn_fw', cfg.dropout, self.is_train, cfg.wd, 'relu'
            )
            s1_bw, _ = directional_attention_with_selections(
                s1_emb, self.sent1_token_mask, s1_act_dep, s1_act_head, 'backward', hn, keep_unselected,
                'dir_attn_bw', cfg.dropout, self.is_train, cfg.wd, 'relu'
            )

            s1_seq_rep = tf.concat([s1_fw, s1_bw], -1)

            tf.get_variable_scope().reuse_variables()

            s2_fw, s2_token_mask_new = directional_attention_with_selections(
                s2_emb, self.sent2_token_mask, s2_act_dep, s2_act_head, 'forward', hn, keep_unselected,
                'dir_attn_fw', cfg.dropout, self.is_train, cfg.wd, 'relu'
            )
            s2_bw, _ = directional_attention_with_selections(
                s2_emb, self.sent2_token_mask, s2_act_dep, s2_act_head, 'backward', hn, keep_unselected,
                'dir_attn_bw', cfg.dropout, self.is_train, cfg.wd, 'relu'
            )
            s2_seq_rep = tf.concat([s2_fw, s2_bw], -1)

        with tf.variable_scope('sentence_enc'):
            s1_rep = multi_dimensional_attention(
                s1_seq_rep, s1_token_mask_new, 'multi_dimensional_attention',
                cfg.dropout, self.is_train, cfg.wd, 'relu',
                tensor_dict=self.tensor_dict, name='s1_attn')
            tf.get_variable_scope().reuse_variables()
            s2_rep = multi_dimensional_attention(
                s2_seq_rep, s2_token_mask_new, 'multi_dimensional_attention',
                cfg.dropout, self.is_train, cfg.wd, 'relu',
                tensor_dict=self.tensor_dict, name='s2_attn')

        with tf.variable_scope('output'):
            out_rep = tf.concat([s1_rep * s2_rep, tf.abs(s1_rep - s2_rep)], -1)
            out_rep_map = bn_dense_layer(
                out_rep, hn, True, 0., 'out_rep_map', 'relu', False, cfg.wd, cfg.dropout, self.is_train)
            if cfg.use_mse and cfg.mse_logits:
                logits = tf.nn.sigmoid(
                    linear(
                        out_rep_map, 1, True, 0., scope='logits', squeeze=True,
                        wd=cfg.wd, input_keep_prob=cfg.dropout, is_train=self.is_train)
                ) * 2. + 3.
            else:
                logits = linear([out_rep_map], self.output_class, True, 0., scope='logits', squeeze=False,
                                wd=cfg.wd, input_keep_prob=cfg.dropout, is_train=self.is_train)
        return logits, (s1_act, s1_logpa), (s2_act, s2_logpa), (s1_percentage, s2_percentage)  # logits

