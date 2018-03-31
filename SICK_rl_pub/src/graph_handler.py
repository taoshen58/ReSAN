from configs import cfg
from src.utils.record_log import _logger
import tensorflow as tf
import os


class GraphHandler(object):
    def __init__(self, model):
        self.model = model
        # choose variable to build saver
        var_list = list(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model.scope))
        var_list += [self.model.global_step]
        self.saver = tf.train.Saver(var_list, max_to_keep=10)

        self.writer = None

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())

        if cfg.load_model:
            if cfg.mode != 'train':
                self.restore(sess)
            else:
                self.restore_part(sess)

        if cfg.mode == 'train':
            self.writer = tf.summary.FileWriter(logdir=cfg.summary_dir, graph=tf.get_default_graph())

    def add_summary(self, summary, global_step):
        _logger.add()
        _logger.add('saving summary...')
        self.writer.add_summary(summary, global_step)
        _logger.done()

    def add_summaries(self, summaries, global_step):
        for summary in summaries:
            self.add_summary(summary, global_step)

    def save(self, sess, global_step = None):
        _logger.add()
        _logger.add('saving model to %s'% cfg.ckpt_path)
        self.saver.save(sess, cfg.ckpt_path, global_step)
        _logger.done()

    def restore(self,sess):
        _logger.add()

        if cfg.load_step is None:
            if cfg.load_path is None:
                _logger.add('trying to restore from dir %s' % cfg.ckpt_dir)
                latest_checkpoint_path = tf.train.latest_checkpoint(cfg.ckpt_dir)
            else:
                latest_checkpoint_path = cfg.load_path
        else:
            latest_checkpoint_path = cfg.ckpt_path+'-'+str(cfg.load_step)

        if latest_checkpoint_path is not None:
            _logger.add('trying to restore from ckpt file %s' % latest_checkpoint_path)
            try:
                self.saver.restore(sess, latest_checkpoint_path)
                _logger.add('success to restore')
            except tf.errors.NotFoundError:
                _logger.add('failure to restore')
                if cfg.mode != 'train': raise FileNotFoundError('canot find model file')
        else:
            _logger.add('No check point file in dir %s '% cfg.ckpt_dir)
            if cfg.mode != 'train': raise FileNotFoundError('canot find model file')

        _logger.done()

    def restore_part(self, sess):

        _logger.add()
        # print(cfg.ckpt_dir)
        all_variable = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.model.scope)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model.scope)
        load_var_list = trainable_vars

        variables_to_restore = [var for var in load_var_list
                                if not var.op.name.startswith(self.model.scope + '/hard_network')]
        private_saver = tf.train.Saver(variables_to_restore)

        load_path = cfg.load_path or cfg.default_pretrained_model_path

        if cfg.load_step is None:
            if load_path is None:
                _logger.add('trying to restore from dir %s' % cfg.ckpt_dir)
                latest_checkpoint_path = tf.train.latest_checkpoint(cfg.ckpt_dir)
            else:
                latest_checkpoint_path = load_path
        else:
            latest_checkpoint_path = cfg.ckpt_path+'-'+str(cfg.load_step)

        if latest_checkpoint_path is not None:
            _logger.add('trying to restore from ckpt file %s' % latest_checkpoint_path)
            try:
                private_saver.restore(sess, latest_checkpoint_path)
                _logger.add('success to restore')
            except tf.errors.NotFoundError:
                _logger.add('failure to restore')
                if cfg.mode != 'train': raise FileNotFoundError('canot find model file')
        else:
            _logger.add('No check point file in dir %s '% cfg.ckpt_dir)
            if cfg.mode != 'train': raise FileNotFoundError('canot find model file')

        _logger.done()

