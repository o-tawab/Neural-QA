import tensorflow as tf
from utils.logger import Logger
from utils.model import *
from models.model import Model
from utils.general import *


class Encoder(object):
    def __init__(self, size):
        self.size = size

    def encode(self, inputs, masks, initial_state_fw=None, initial_state_bw=None, dropout=1.0, reuse=False):
        # The contextual level embeddings
        output_concat, (final_state_fw, final_state_bw) = BiLSTM(inputs, masks, self.size, initial_state_fw,
                                                                 initial_state_bw, dropout, reuse)
        Logger.debug("output shape: {}".format(output_concat.get_shape()))

        return output_concat, (final_state_fw, final_state_bw)


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, inputs, mask, max_input_length):
        with tf.variable_scope("start"):
            start = logits_helper(inputs, max_input_length)
            start = prepro_for_softmax(start, mask)

        with tf.variable_scope("end"):
            end = logits_helper(inputs, max_input_length)
            end = prepro_for_softmax(end, mask)

        return start, end


class BaselineModel(Model):
    def __init__(self, embeddings, config):
        super().__init__(config)
        self.config = config
        self.embeddings = embeddings

        self.encoder = Encoder(config.model.hidden_size)
        self.decoder = Decoder(config.model.hidden_size)

        self.add_placeholders()

        self.build()

        self.saver = tf.train.Saver(max_to_keep=5)

    def add_placeholders(self):
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.context_mask_placeholder = tf.placeholder(tf.bool, shape=(None, None))
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, None))
        self.question_mask_placeholder = tf.placeholder(tf.bool, shape=(None, None))

        self.answer_span_start_placeholder = tf.placeholder(tf.int32)
        self.answer_span_end_placeholder = tf.placeholder(tf.int32)

        self.max_context_length_placeholder = tf.placeholder(tf.int32)
        self.max_question_length_placeholder = tf.placeholder(tf.int32)
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def setup_embeddings(self):
        with tf.variable_scope("embeddings"):
            if self.config.training.retrain_embeddings:
                embeddings = tf.get_variable("embeddings", initializer=self.embeddings)
            else:
                embeddings = tf.cast(self.embeddings, dtype=tf.float32)

            self.question_embeddings = self._embedding_lookup(embeddings, self.question_placeholder,
                                                              self.max_question_length_placeholder)
            self.context_embeddings = self._embedding_lookup(embeddings, self.context_placeholder,
                                                             self.max_context_length_placeholder)

    def _embedding_lookup(self, embeddings, indicies, max_length):
        embeddings = tf.nn.embedding_lookup(embeddings, indicies)
        embeddings = tf.reshape(embeddings, shape=[-1, max_length, self.config.model.embedding_size])
        return embeddings

    def add_preds_op(self):
        with tf.variable_scope("q"):
            Hq, (q_final_state_fw, q_final_state_bw) = self.encoder.encode(self.question_embeddings,
                                                                           self.question_mask_placeholder,
                                                                           dropout=self.dropout_placeholder)

        if self.config.model.share_encoder_weights:
            with tf.variable_scope("q"):
                Hc, (_, _) = self.encoder.encode(self.context_embeddings,
                                                 self.context_mask_placeholder,
                                                 initial_state_fw=q_final_state_fw,
                                                 initial_state_bw=q_final_state_bw,
                                                 dropout=self.dropout_placeholder,
                                                 reuse=True)
        else:
            with tf.variable_scope("c"):
                Hc, (_, _) = self.encoder.encode(self.context_embeddings,
                                                 self.context_mask_placeholder,
                                                 initial_state_fw=q_final_state_fw,
                                                 initial_state_bw=q_final_state_bw,
                                                 dropout=self.dropout_placeholder)

        with tf.variable_scope("decoding"):
            start, end = self.decoder.decode(Hc, self.context_mask_placeholder,
                                             self.max_context_length_placeholder)
        self.preds = start, end

    def add_loss_op(self):
        with tf.variable_scope("loss"):
            answer_span_start_one_hot = tf.one_hot(self.answer_span_start_placeholder, self.max_context_length_placeholder)
            answer_span_end_one_hot = tf.one_hot(self.answer_span_end_placeholder, self.max_context_length_placeholder)

            start, end = self.preds
            loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=start, labels=answer_span_start_one_hot))
            loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=end, labels=answer_span_end_one_hot))
            self.loss = loss1 + loss2

    def add_training_op(self):
        variables = tf.trainable_variables()
        gradients = tf.gradients(self.loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config.training.max_grad_norm)

        if self.config.training.learning_rate_annealing:
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.config.training.learning_rate, global_step, 1250, 0.96,
                                                       staircase=False)
            global_step = tf.add(1, global_step)
        else:
            learning_rate = self.config.training.learning_rate

        optimizer = get_optimizer(self.config.model.optimizer, learning_rate)
        train_op = optimizer.apply_gradients(zip(gradients, variables))

        # For applying EMA for trained parameters
        if self.config.training.ema_for_weights:
            ema = tf.train.ExponentialMovingAverage(0.999)
            ema_op = ema.apply(variables)

            with tf.control_dependencies([train_op]):
                train_op = tf.group(ema_op)

        self.train_op = train_op

    def create_feed_dict(self, context, question, answer_span_start_batch=None, answer_span_end_batch=None,
                         is_train=True):

        context_batch, context_mask, max_context_length = pad_sequences(context,
                                                                        max_sequence_length=self.config.training.max_context_length)
        question_batch, question_mask, max_question_length = pad_sequences(question,
                                                                           max_sequence_length=self.config.training.max_question_length)

        feed_dict = {self.context_placeholder: context_batch,
                     self.context_mask_placeholder: context_mask,
                     self.question_placeholder: question_batch,
                     self.question_mask_placeholder: question_mask,
                     self.max_context_length_placeholder: max_context_length,
                     self.max_question_length_placeholder: max_question_length}

        if is_train:
            feed_dict[self.dropout_placeholder] = self.config.model.keep_prob
        else:
            feed_dict[self.dropout_placeholder] = 1.0

        if answer_span_start_batch is not None and answer_span_end_batch is not None:
            feed_dict[self.answer_span_start_placeholder] = answer_span_start_batch
            feed_dict[self.answer_span_end_placeholder] = answer_span_end_batch

        return feed_dict
