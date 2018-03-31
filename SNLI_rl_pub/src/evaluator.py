from configs import cfg
from src.utils.record_log import _logger
import numpy as np
import tensorflow as tf


class Evaluator(object):
    def __init__(self, model):
        self.model = model
        self.global_step = model.global_step

        ## ---- summary----
        self.build_summary()
        self.writer = tf.summary.FileWriter(cfg.summary_dir)

    def get_evaluation(self, sess, dataset_obj, global_step=None, time_counter=None):
        _logger.add()
        _logger.add('getting evaluation result for %s' % dataset_obj.data_type)

        logits_list, loss_sl_list, loss_rl_list, accu_list = [], [], [], []
        percentage_list = []
        for sample_batch, _, _, _ in dataset_obj.generate_batch_sample_iter():
            feed_dict = self.model.get_feed_dict(sample_batch, 'dev')
            if time_counter is not None:
                time_counter.add_start()
            logits, loss_sl, loss_rl, accu, percentage = \
                sess.run([self.model.logits,
                          self.model.loss_sl, self.model.loss_rl,
                          self.model.accuracy, self.model.choose_percentage], feed_dict)
            if time_counter is not None:
                time_counter.add_stop()
            logits_list.append(np.argmax(logits, -1))
            loss_sl_list.append(loss_sl)
            loss_rl_list.append(loss_rl)
            accu_list.append(accu)
            percentage_list.append(percentage)


        logits_array = np.concatenate(logits_list, 0)
        loss_sl_value = np.mean(loss_sl_list)
        loss_rl_value = np.mean(loss_rl_list)
        accu_array = np.concatenate(accu_list, 0)
        accu_value = np.mean(accu_array)
        percentage_value = np.mean(percentage_list)

        # todo: analysis
        # analysis_save_dir = cfg.mkdir(cfg.answer_dir, 'gs_%d' % global_step or 0)
        # OutputAnalysis.do_analysis(dataset_obj, logits_array, accu_array, analysis_save_dir,
        #                            cfg.fine_grained)

        if global_step is not None:
            if dataset_obj.data_type == 'train':
                summary_feed_dict = {
                    self.train_loss_sl: loss_sl_value,
                    self.train_loss_rl: loss_rl_value,
                    self.train_accuracy: accu_value,
                    self.train_percentage: percentage_value,
                }
                summary = sess.run(self.train_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)
            elif dataset_obj.data_type == 'dev':
                summary_feed_dict = {
                    self.dev_loss_sl: loss_sl_value,
                    self.dev_loss_rl: loss_rl_value,
                    self.dev_accuracy: accu_value,
                    self.dev_percentage: percentage_value,
                }
                summary = sess.run(self.dev_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)
            else:
                summary_feed_dict = {
                    self.test_loss_sl: loss_sl_value,
                    self.test_loss_rl: loss_rl_value,
                    self.test_accuracy: accu_value,
                    self.test_percentage: percentage_value,
                }
                summary = sess.run(self.test_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)

        return (loss_sl_value, loss_rl_value), accu_value, percentage_value


    # --- internal use ------
    def build_summary(self):
        with tf.name_scope('train_summaries'):
            self.train_loss_sl = tf.placeholder(tf.float32, [], 'train_loss_sl')
            self.train_loss_rl = tf.placeholder(tf.float32, [], 'train_loss_rl')
            self.train_accuracy = tf.placeholder(tf.float32, [], 'train_accuracy')
            self.train_percentage = tf.placeholder(tf.float32, [], 'train_percentage')

            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_loss_sl', self.train_loss_sl))
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_loss_rl', self.train_loss_rl))
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_accuracy', self.train_accuracy))
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_percentage',
                                                                                 self.train_percentage))
            self.train_summaries = tf.summary.merge_all('train_summaries_collection')

        with tf.name_scope('dev_summaries'):
            self.dev_loss_sl = tf.placeholder(tf.float32, [], 'dev_loss_sl')
            self.dev_loss_rl = tf.placeholder(tf.float32, [], 'dev_loss_rl')
            self.dev_accuracy = tf.placeholder(tf.float32, [], 'dev_accuracy')
            self.dev_percentage = tf.placeholder(tf.float32, [], 'dev_percentage')

            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_loss_sl',self.dev_loss_sl))
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_loss_rl',self.dev_loss_rl))
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_accuracy',self.dev_accuracy))
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_percentage',self.dev_percentage))


            self.dev_summaries = tf.summary.merge_all('dev_summaries_collection')

        with tf.name_scope('test_summaries'):
            self.test_loss_sl = tf.placeholder(tf.float32, [], 'test_loss_sl')
            self.test_loss_rl = tf.placeholder(tf.float32, [], 'test_loss_rl')
            self.test_accuracy = tf.placeholder(tf.float32, [], 'test_accuracy')
            self.test_percentage = tf.placeholder(tf.float32, [], 'test_percentage')

            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_loss_sl',self.test_loss_sl))
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_loss_rl',self.test_loss_rl))
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_accuracy',self.test_accuracy))
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_percentage',self.test_percentage))
            self.test_summaries = tf.summary.merge_all('test_summaries_collection')




