import tensorflow as tf
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils.logger import Logger
import os


class LSTMTrainer:
    def __init__(self, sess, model, train_data_generator, val_data_generator, test_data_generator, config):
        self.sess = sess
        self.model = model
        self.train_data_generator = train_data_generator.yield_batch()
        self.num_train_batchs = len(train_data_generator)
        self.val_data_generator = val_data_generator.yield_batch()
        self.num_val_batchs = len(val_data_generator)
        self.test_data_generator = test_data_generator.yield_batch()
        self.num_test_batchs = len(test_data_generator)
        self.config = config

        # To initialize all variables
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        self.logger = Logger(self.sess, self.config.summary_dir)

        if self.config.training.continue_training:
            self.model.load(sess)

    def create_feed_dict(self):
        pass

    def train(self):
        Logger.info("Starting training...")

        Logger.info("Training finished")

    def eval(self):
        Logger.info("Starting evaluating...")

        Logger.info("Finished evaluation")

    def infer(self):
        Logger.info("Starting inference...")

        Logger.info("Finished inference")
