from configs import cfg
from src.utils.record_log import _logger
import tensorflow as tf

from src.model.model_template import ModelTemplate
from src.nn_utils.nn import linear, bn_dense_layer
from src.nn_utils.integration_func import generate_embedding_mat, multi_dimensional_attention, \
    directional_attention_with_dense


class ModelResanBase(ModelTemplate):
    def __init__(self, token_emb_mat, glove_emb_mat, tds, cds, tl, scope):
        super(ModelResanBase, self).__init__(token_emb_mat, glove_emb_mat, tds, cds, tl, scope)
        self.disable_rl = True
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
            # s1_act, s1_logpa, s2_act, s2_logpa, choose_percentage
            s1_act = self.sent1_token_mask
            s1_logpa = tf.cast(s1_act, tf.float32)

            s2_act = self.sent2_token_mask
            s2_logpa = tf.cast(s2_act, tf.float32)

            s1_percentage = tf.ones([bs], tf.float32)
            s2_percentage = tf.ones([bs], tf.float32)

        with tf.variable_scope('ct_attn'):
            s1_fw = directional_attention_with_dense(
                s1_emb, self.sent1_token_mask, 'forward', 'dir_attn_fw',
                cfg.dropout, self.is_train, cfg.wd, 'relu',
                tensor_dict=self.tensor_dict, name='s1_fw_attn')
            s1_bw = directional_attention_with_dense(
                s1_emb, self.sent1_token_mask, 'backward', 'dir_attn_bw',
                cfg.dropout, self.is_train, cfg.wd, 'relu',
                tensor_dict=self.tensor_dict, name='s1_bw_attn')

            s1_seq_rep = tf.concat([s1_fw, s1_bw], -1)

            tf.get_variable_scope().reuse_variables()

            s2_fw = directional_attention_with_dense(
                s2_emb, self.sent2_token_mask, 'forward', 'dir_attn_fw',
                cfg.dropout, self.is_train, cfg.wd, 'relu',
                tensor_dict=self.tensor_dict, name='s2_fw_attn')
            s2_bw = directional_attention_with_dense(
                s2_emb, self.sent2_token_mask, 'backward', 'dir_attn_bw',
                cfg.dropout, self.is_train, cfg.wd, 'relu',
                tensor_dict=self.tensor_dict, name='s2_bw_attn')
            s2_seq_rep = tf.concat([s2_fw, s2_bw], -1)

        with tf.variable_scope('sentence_enc'):
            s1_rep = multi_dimensional_attention(
                s1_seq_rep, self.sent1_token_mask, 'multi_dimensional_attention',
                cfg.dropout, self.is_train, cfg.wd, 'relu',
                tensor_dict=self.tensor_dict, name='s1_attn')
            tf.get_variable_scope().reuse_variables()
            s2_rep = multi_dimensional_attention(
                s2_seq_rep, self.sent2_token_mask, 'multi_dimensional_attention',
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
                                wd=cfg.wd, input_keep_prob=cfg.dropout,is_train=self.is_train)
        return logits, (s1_act, s1_logpa), (s2_act, s2_logpa), (s1_percentage, s2_percentage)  # logits


