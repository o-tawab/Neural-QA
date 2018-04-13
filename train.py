import argparse
import tensorflow as tf
from utils.exp_utils import process_config, create_dirs
from utils.logger import Logger
from utils.data_reader import *
from models.baseline import BaselineModel
from models.attention import LuongAttention


def main():
    parser = argparse.ArgumentParser(description="Neural-QA")
    parser.add_argument('--config', default='./configs/attention.json', type=str, help='Configuration file')

    # Parse the arguments
    args = parser.parse_args()

    config = process_config(args.config)
    create_dirs([config.summary_dir, config.checkpoint_dir])

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # load the data
    train, val = load_and_preprocess_data(config.data.dir)

    # load the word matrix
    embeddings = load_word_embeddings(config.data.dir)

    # tensorflow session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    if config.model.name == 'baseline':
        model = BaselineModel(embeddings, config)
    elif config.model.name == 'attention':
        model = LuongAttention(embeddings, config)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    logger = Logger(sess, config.summary_dir)

    model.train(sess, train, val, logger)


if __name__ == '__main__':
    main()
