from configs import cfg
from src.utils.record_log import _logger
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod


class ModelTemplate(metaclass=ABCMeta):
    def __init__(self, token_emb_mat, glove_emb_mat, tds, cds, tl, scope):
        self.scope = scope
        self.disable_rl = False  # todo (underline): switch for reinforcement learning
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        self.update_global_step = self.global_step.assign(self.global_step + 1)

        self.if_train_rl = False
        self.count_to_alternate = 0

        self.token_emb_mat, self.glove_emb_mat = token_emb_mat, glove_emb_mat

        # ---- place holder -----
        self.sent1_token = tf.placeholder(tf.int32, [None, None], name='sent1_token')
        self.sent1_char = tf.placeholder(tf.int32, [None, None, tl], name='sent1_char')

        self.sent2_token = tf.placeholder(tf.int32, [None, None], name='sent2_token')
        self.sent2_char = tf.placeholder(tf.int32, [None, None, tl], name='sent2_char')

        self.target_distribution = tf.placeholder(tf.float32, [None, 5], name='target_distribution')
        self.gold_label = tf.placeholder(tf.float32, [None], name='gold_label')
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')

        # ----------- parameters -------------
        self.tds, self.cds = tds, cds
        self.tl = tl
        self.tel = cfg.word_embedding_length
        self.cel = cfg.char_embedding_length
        self.cos = cfg.char_out_size
        self.ocd = list(map(int, cfg.out_channel_dims.split(',')))
        self.fh = list(map(int, cfg.filter_heights.split(',')))
        self.hn = cfg.hidden_units_num
        self.finetune_emb = cfg.fine_tune

        self.output_class = 5
        self.bs = tf.shape(self.sent1_token)[0]
        self.sl1 = tf.shape(self.sent1_token)[1]
        self.sl2 = tf.shape(self.sent2_token)[1]

        # ------------ other ---------
        self.sent1_token_mask = tf.cast(self.sent1_token, tf.bool)
        self.sent1_char_mask = tf.cast(self.sent1_char, tf.bool)
        self.sent1_token_len = tf.reduce_sum(tf.cast(self.sent1_token_mask, tf.int32), -1)
        self.sent1_char_len = tf.reduce_sum(tf.cast(self.sent1_char_mask, tf.int32), -1)

        self.sent2_token_mask = tf.cast(self.sent2_token, tf.bool)
        self.sent2_char_mask = tf.cast(self.sent2_char, tf.bool)
        self.sent2_token_len = tf.reduce_sum(tf.cast(self.sent2_token_mask, tf.int32), -1)
        self.sent2_char_len = tf.reduce_sum(tf.cast(self.sent2_char_mask, tf.int32), -1)

        self.tensor_dict = {}

        # ------ start ------
        self.logits = None
        self.s1_logpa = None
        self.s1_act = None
        self.s2_logpa = None
        self.s2_act = None

        self.s1_percentage = None
        self.s2_percentage = None

        self.loss_sl = None
        self.loss_rl = None
        self.reward_mean = None
        self.choose_percentage = None

        self.accuracy = None
        self.var_ema = None
        self.ema = None
        self.summary = None
        self.opt_sl = None
        self.opt_rl = None
        self.train_op_sl = None
        self.train_op_rl = None

    @abstractmethod
    def build_network(self):
        return None, None, None, None

    def build_loss(self):
        _logger.add('regularization var num: %d' % len(set(tf.get_collection('reg_vars', self.scope))))
        _logger.add('trainable var num: %d' % len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)))
        # weight_decay
        with tf.name_scope("weight_decay"):
            for var in set(tf.get_collection('reg_vars', self.scope)):
                tensor_name = var.op.name
                weight_decay = tf.multiply(tf.nn.l2_loss(var), cfg.wd,
                                           name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
                if not tensor_name.startswith(self.scope+'/hard_network'):
                    tf.add_to_collection('losses_sl', weight_decay)
                if tensor_name.startswith(self.scope+'/hard_network'):
                    tf.add_to_collection('losses_rl', weight_decay)

        if cfg.mse_logits:
            cost_batch = 0.5 * (self.logits - self.gold_label) ** 2
            # tf.add_to_collection('losses', tf.reduce_mean(mse, name='mse_mean'))
        else:
            if cfg.use_mse:
                predicted_dist = tf.nn.softmax(self.logits)
                mask = tf.tile(tf.expand_dims(tf.range(1, self.output_class + 1), 0), [self.bs, 1])
                predicted_score = tf.reduce_sum(predicted_dist * tf.cast(mask, tf.float32), -1)  # bs
                cost_batch = 0.5 * (predicted_score - self.gold_label) ** 2
                # tf.add_to_collection('losses', tf.reduce_mean(mse, name='mse_mean'))

            else:
                target_dist = tf.clip_by_value(self.target_distribution, 1e-10, 1.)
                predicted_dist = tf.clip_by_value(tf.nn.softmax(self.logits), 1e-10, 1.)

                cost_batch = tf.reduce_sum(target_dist * tf.log(target_dist / predicted_dist), -1)
                # kl_batch = tf.reduce_sum((target_dist - predicted_dist) ** 2, -1)
                # tf.add_to_collection('losses', tf.reduce_mean(kl_batch, name='kl_divergence_mean'))

        # @ 1. for normal
        cost_sl = tf.reduce_mean(cost_batch, name='cost_sl')
        tf.add_to_collection('losses_sl', cost_sl)
        loss_sl = tf.add_n(tf.get_collection('losses_sl', self.scope), name='loss_sl')
        tf.summary.scalar(loss_sl.op.name, loss_sl)
        tf.add_to_collection('ema/scalar', loss_sl)

        # @ 2. for rl
        self.choose_percentage = tf.reduce_mean(tf.stack([self.s1_percentage, self.s2_percentage]),
                                                name='choose_percentage')
        tf.summary.scalar(self.choose_percentage.op.name, self.choose_percentage)
        tf.add_to_collection('ema/scalar', self.choose_percentage)

        # # loss_rl
        s1_rewards_raw = - (cost_batch + cfg.rl_sparsity * self.s1_percentage)
        s2_rewards_raw = - (cost_batch + cfg.rl_sparsity * self.s2_percentage)

        self.reward_mean = tf.reduce_mean(tf.stack([s1_rewards_raw, s2_rewards_raw]), name='reward_mean')
        tf.summary.scalar(self.reward_mean.op.name, self.reward_mean)
        tf.add_to_collection('ema/scalar', self.reward_mean)

        cost_rl = - tf.reduce_mean(
            s1_rewards_raw * tf.reduce_sum(self.s1_logpa, 1) +
            s2_rewards_raw * tf.reduce_sum(self.s2_logpa, 1), name='cost_rl')
        tf.add_to_collection('losses_rl', cost_rl)
        loss_rl = tf.add_n(tf.get_collection('losses_rl', self.scope), name='loss_rl')
        tf.summary.scalar(loss_rl.op.name, loss_rl)
        tf.add_to_collection('ema/scalar', loss_rl)

        return loss_sl, loss_rl

    def build_mse(self):
        if cfg.mse_logits:
            mse = 0.5 * (self.logits - self.gold_label) ** 2
            predicted_score = self.logits
        else:
            predicted_dist = tf.nn.softmax(self.logits)
            mask = tf.tile(tf.expand_dims(tf.range(1, self.output_class+1), 0), [self.bs, 1])
            predicted_score = tf.reduce_sum(predicted_dist * tf.cast(mask, tf.float32), -1)  # bs

            mse = (predicted_score - self.gold_label) ** 2
        return mse, predicted_score

    def update_tensor_add_ema_and_opt(self):
        self.logits, (self.s1_act, self.s1_logpa), (self.s2_act, self.s2_logpa), \
        (self.s1_percentage, self.s2_percentage) = self.build_network()
        self.loss_sl, self.loss_rl = self.build_loss()
        self.mse, self.predicted_score = self.build_mse()

        # ------------ema-------------
        if True:
            self.var_ema = tf.train.ExponentialMovingAverage(cfg.var_decay)
            self.build_var_ema()

        if cfg.mode == 'train':
            self.ema = tf.train.ExponentialMovingAverage(cfg.decay)
            self.build_ema()
        self.summary = tf.summary.merge_all()

        # ---------- optimization ---------
        if cfg.optimizer.lower() == 'adadelta':
            assert cfg.learning_rate > 0.1 and cfg.learning_rate < 1.
            self.opt_sl = tf.train.AdadeltaOptimizer(cfg.learning_rate)
            self.opt_rl = tf.train.AdadeltaOptimizer(cfg.learning_rate)
        elif cfg.optimizer.lower() == 'adam':
            assert cfg.learning_rate < 0.1
            self.opt_sl = tf.train.AdamOptimizer(cfg.learning_rate)
            self.opt_rl = tf.train.AdamOptimizer(cfg.learning_rate)
        elif cfg.optimizer.lower() == 'rmsprop':
            assert cfg.learning_rate < 0.1
            self.opt_sl = tf.train.RMSPropOptimizer(cfg.learning_rate)
            self.opt_rl = tf.train.RMSPropOptimizer(cfg.learning_rate)
        else:
            raise AttributeError('no optimizer named as \'%s\'' % cfg.optimizer)

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        # trainable param num:
        # print params num
        all_params_num = 0
        for elem in trainable_vars:
            # elem.name
            var_name = elem.name.split(':')[0]
            if var_name.endswith('emb_mat'):
                continue
            params_num = 1
            for l in elem.get_shape().as_list(): params_num *= l
            all_params_num += params_num
        _logger.add('Trainable Parameters Number: %d' % all_params_num)

        sl_vars = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
                   if not var.op.name.startswith(self.scope + '/hard_network')]
        self.train_op_sl = self.opt_sl.minimize(
            self.loss_sl, self.global_step,
            var_list=sl_vars)

        rl_vars = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
                   if var.op.name.startswith(self.scope + '/hard_network')]
        if len(rl_vars) > 0:
            self.train_op_rl = self.opt_rl.minimize(
                self.loss_rl,
                var_list=rl_vars)
        else:
            self.train_op_rl = None

    def build_var_ema(self):
        ema_op = self.var_ema.apply(tf.trainable_variables())
        with tf.control_dependencies([ema_op]):
            self.loss_sl = tf.identity(self.loss_sl)
            self.loss_rl = tf.identity(self.loss_rl)

    def build_ema(self):
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + \
                  tf.get_collection("ema/vector", scope=self.scope)
        ema_op = self.ema.apply(tensors)
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = self.ema.average(var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector", scope=self.scope):
            ema_var = self.ema.average(var)
            tf.summary.histogram(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss_sl = tf.identity(self.loss_sl)
            self.loss_rl = tf.identity(self.loss_rl)

    def step(self, sess, batch_samples, get_summary=False, global_step_value=None):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples, 'train')

        # summary
        summary_tf = self.summary if get_summary else self.is_train

        # cfg.
        cfg.time_counter.add_start()

        if self.disable_rl or global_step_value <= cfg.start_only_rl:
            loss_sl, summary, _ = sess.run(
                [self.loss_sl, summary_tf, self.train_op_sl], feed_dict=feed_dict)
            loss_rl = 0.
        else:
            if cfg.start_only_rl < global_step_value <= cfg.end_only_rl:
                # rl
                loss_sl, loss_rl, summary, _ = sess.run(
                    [self.loss_sl, self.loss_rl, summary_tf, self.train_op_rl], feed_dict=feed_dict)
                sess.run(self.update_global_step)
            else:
                if cfg.rl_strategy == 'sep':
                    if self.if_train_rl:
                        # rl
                        loss_sl, loss_rl, summary, _ = sess.run(
                            [self.loss_sl, self.loss_rl, summary_tf, self.train_op_rl], feed_dict=feed_dict)
                        sess.run(self.update_global_step)
                    else:
                        # sl
                        loss_sl, summary, _ = sess.run(
                            [self.loss_sl, summary_tf, self.train_op_sl], feed_dict=feed_dict)
                        loss_rl = 0.
                    # update count
                    self.count_to_alternate += 1
                    if (self.if_train_rl and self.count_to_alternate >= cfg.step_for_rl) or \
                            ((not self.if_train_rl) and self.count_to_alternate >= cfg.step_for_sl):
                        self.count_to_alternate = 0
                        self.if_train_rl = not self.if_train_rl
                else:
                    loss_sl, loss_rl, summary, _, _ = sess.run(
                        [self.loss_sl, self.loss_rl, summary_tf, self.train_op_sl, self.train_op_rl],
                        feed_dict=feed_dict)

        cfg.time_counter.add_stop()

        # summary
        summary = summary if get_summary else None

        return (loss_sl, loss_rl), summary

    def get_feed_dict(self, sample_batch, data_type='train'):
        # max lens
        sl1, sl2 = 0, 0

        for sample in sample_batch:
            sl1 = max(sl1, len(sample['sentence1_token_digital']))
            sl2 = max(sl2, len(sample['sentence2_token_digital']))


        # token and char
        sent1_token_b = []
        sent1_char_b = []
        sent2_token_b = []
        sent2_char_b = []
        for sample in sample_batch:
            sent1_token = np.zeros([sl1], cfg.intX)
            sent1_char = np.zeros([sl1, self.tl], cfg.intX)
            for idx_t, (token, char_seq_v) in enumerate(zip(sample['sentence1_token_digital'],
                                                            sample['sentence1_char_digital'])):
                sent1_token[idx_t] = token
                for idx_c, char in enumerate(char_seq_v):
                    if idx_c < self.tl:
                        sent1_char[idx_t, idx_c] = char

            sent2_token = np.zeros([sl2], cfg.intX)
            sent2_char = np.zeros([sl2, self.tl], cfg.intX)

            for idx_t, (token, char_seq_v) in enumerate(zip(sample['sentence2_token_digital'],
                                                            sample['sentence2_char_digital'])):
                sent2_token[idx_t] = token
                for idx_c, char in enumerate(char_seq_v):
                    if idx_c < self.tl:
                        sent2_char[idx_t, idx_c] = char
            sent1_token_b.append(sent1_token)
            sent1_char_b.append(sent1_char)
            sent2_token_b.append(sent2_token)
            sent2_char_b.append(sent2_char)
        sent1_token_b = np.stack(sent1_token_b)
        sent1_char_b = np.stack(sent1_char_b)
        sent2_token_b = np.stack(sent2_token_b)
        sent2_char_b = np.stack(sent2_char_b)

        # label
        target_distribution_b = []
        gold_label_b = []
        for sample in sample_batch:
            target_distribution_b.append(sample['distribution'])
            gold_label_b.append(sample['relatedness_score'])
        target_distribution_b = np.array(target_distribution_b, cfg.floatX)
        gold_label_b = np.stack(gold_label_b).astype(cfg.floatX)

        feed_dict = {
            self.sent1_token: sent1_token_b, self.sent1_char: sent1_char_b,
            self.sent2_token: sent2_token_b, self.sent2_char: sent2_char_b,
            self.target_distribution: target_distribution_b,
            self.gold_label: gold_label_b,
            self.is_train: True if data_type == 'train' else False
        }

        return feed_dict



