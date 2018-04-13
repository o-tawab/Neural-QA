import tensorflow as tf
from utils.logger import Logger


class BaselineModel:
    def __init__(self, config):
        self.config = config

        self.init_placeholders()

        self.init_cur_epoch()

        self.build_model()

        self.saver = tf.train.Saver(max_to_keep=5)

    def save(self, sess):
        self.saver.save(sess, self.config.checkpoint_dir, self.cur_epoch_tensor)
        Logger.info("Model saved")

    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            Logger.info("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            Logger.info("Model loaded")

    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.cur_epoch_input = tf.placeholder('int32', None, name='cur_epoch_input')
            self.cur_epoch_assign_op = self.cur_epoch_tensor.assign(self.cur_epoch_input)

    def init_placeholders(self):
        with tf.name_scope('inputs'):
            pass

    def setup_embeddings(self):
        pass

    def add_loss_op(self):
        pass

    def add_optimization_op(self):
        pass

    def encode(self):
        pass

    def decode(self):
        pass

    def build_model(self):
        pass
