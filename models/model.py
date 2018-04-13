import logging
from utils.general import batches, Progbar, get_random_samples, find_best_span
from utils.eval import evaluate
import numpy as np
import tensorflow as tf
from os.path import join as pjoin
from abc import ABCMeta, abstractmethod

logging.basicConfig(level=logging.INFO)


class Model(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config

        self.init_cur_epoch()

        self.preds = None
        self.loss = None
        self.train_op = None

        self.saver = None

    @abstractmethod
    def add_placeholders(self):
        pass

    @abstractmethod
    def add_preds_op(self):
        pass

    @abstractmethod
    def add_loss_op(self):
        pass

    @abstractmethod
    def add_training_op(self):
        pass

    @abstractmethod
    def create_feed_dict(self, context, question, answer_span_start_batch=None, answer_span_end_batch=None,
                         is_train=True):
        pass

    @abstractmethod
    def setup_embeddings(self):
        pass

    def build(self):
        self.setup_embeddings()
        self.add_preds_op()
        self.add_loss_op()
        self.add_training_op()

    def train(self, session, train, val, logger):
        variables = tf.trainable_variables()
        num_vars = np.sum([np.prod(v.get_shape().as_list()) for v in variables])
        logging.info("Number of variables in models: {}".format(num_vars))
        for i in range(self.config.training.num_epochs):
            self.run_epoch(session, train, val, logger=logger)

    def run_epoch(self, session, train, val, logger):
        num_samples = len(train["context"])
        num_batches = int(np.ceil(num_samples) * 1.0 / self.config.training.batch_size)

        progress = Progbar(target=num_batches)
        best_f1 = 0
        losses = []
        for i, train_batch in enumerate(
                batches(train, is_train=True, batch_size=self.config.training.batch_size, window_size=self.config.training.window_size)):
            _, loss = self.optimize(session, train_batch)
            losses.append(loss)
            progress.update(i, [("training loss", np.mean(losses))])

            if i % self.config.training.eval_num == 0 or i == num_batches:

                # Randomly get some samples from the dataset
                train_samples = get_random_samples(train, self.config.training.samples_used_for_evaluation)
                val_samples = get_random_samples(val, self.config.training.samples_used_for_evaluation)

                # First evaluate on the training set for not using best span
                f1_train, EM_train = self.evaluate_answer(session, train_samples, use_best_span=False)

                # Then evaluate on the val set
                f1_val, EM_val = self.evaluate_answer(session, val_samples, use_best_span=False)

                logging.info("Not using best span")
                logging.info("F1: {}, EM: {}, for {} training samples".format(f1_train, EM_train,
                                                                              self.config.training.samples_used_for_evaluation))
                logging.info("F1: {}, EM: {}, for {} validation samples".format(f1_val, EM_val,
                                                                                self.config.training.samples_used_for_evaluation))

                # First evaluate on the training set
                f1_train, EM_train = self.evaluate_answer(session, train_samples, use_best_span=True)

                # Then evaluate on the val set
                f1_val, EM_val = self.evaluate_answer(session, val_samples, use_best_span=True)

                logging.info("Using best span")
                logging.info("F1: {}, EM: {}, for {} training samples".format(f1_train, EM_train,
                                                                              self.config.training.samples_used_for_evaluation))
                logging.info("F1: {}, EM: {}, for {} validation samples".format(f1_val, EM_val,
                                                                                self.config.training.samples_used_for_evaluation))

                summaries_dict = {
                    "f1_train": f1_train,
                    "EM_train": EM_train,
                    "f1_val": f1_val,
                    "EM_val": EM_val,
                    "training_loss": np.mean(losses)
                }

                logger.add_scalar_summary(self.cur_epoch_tensor.eval(session), summaries_dict)

                if f1_val > best_f1:
                    self.save(session)
                    best_f1 = f1_val

    def optimize(self, session, batch):
        context = batch["context"]
        question = batch["question"]
        answer_span_start = batch["answer_span_start"]
        answer_span_end = batch["answer_span_end"]

        input_feed = self.create_feed_dict(context, question, answer_span_start, answer_span_end)
        output_feed = [self.train_op, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def evaluate_answer(self, session, data, use_best_span):

        # Now we whether finding the best span improves the score
        start_indicies, end_indicies = self.predict_for_batch(session, data, use_best_span)
        pred_answer, truth_answer = self.get_sentences_from_indices(data, start_indicies, end_indicies)
        result = evaluate(pred_answer, truth_answer)

        f1 = result["f1"]
        EM = result["EM"]

        return f1, EM

    def predict_for_batch(self, session, data, use_best_span):
        start_indices = []
        end_indices = []
        for batch in batches(data, is_train=False, shuffle=False):
            start_index, end_index = self.answer(session, batch, use_best_span)
            start_indices.extend(start_index)
            end_indices.extend(end_index)
        return start_indices, end_indices

    def get_sentences_from_indices(self, data, start_index, end_index):
        answer_word_pred = []
        answer_word_truth = []
        word_context = data["word_context"]
        answer_span_start = data["answer_span_start"]
        answer_span_end = data["answer_span_end"]

        for span, context in zip(zip(start_index, end_index), word_context):
            prediction = " ".join(context.split()[span[0]:span[1] + 1])
            answer_word_pred.append(prediction)

        for span, context in zip(zip(answer_span_start, answer_span_end), word_context):
            truth = " ".join(context.split()[span[0]:span[1] + 1])
            answer_word_truth.append(truth)

        return answer_word_pred, answer_word_truth

    def test(self, session, val):
        context = val["context"]
        question = val["question"]
        answer_span_start = val["answer_span_start"]
        answer_span_end = val["answer_span_end"]

        input_feed = self.create_feed_dict(context, question, answer_span_start, answer_span_end, is_train=False)
        output_feed = self.loss

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, batch):
        context = batch["context"]
        question = batch["question"]
        answer_span_start = batch["answer_span_start"]
        answer_span_end = batch["answer_span_end"]

        input_feed = self.create_feed_dict(context, question, answer_span_start, answer_span_end, is_train=False)
        output_feed = self.preds

        start, end = session.run(output_feed, input_feed)

        return start, end

    def answer(self, session, data, use_best_span):

        start, end = self.decode(session, data)

        if use_best_span:
            start_index, end_index = find_best_span(start, end)
        else:
            start_index = np.argmax(start, axis=1)
            end_index = np.argmax(end, axis=1)

        return start_index, end_index

    def validate(self, sess, val):
        valid_cost = self.test(sess, val)

        return valid_cost

    def save(self, sess):
        self.saver.save(sess, self.config.checkpoint_dir, self.cur_epoch_tensor)
        logging.info("Model saved")

    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            logging.info("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            logging.info("Model loaded")

    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.cur_epoch_input = tf.placeholder('int32', None, name='cur_epoch_input')
            self.cur_epoch_assign_op = self.cur_epoch_tensor.assign(self.cur_epoch_input)
