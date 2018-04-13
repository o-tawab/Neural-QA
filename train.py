import argparse
import tensorflow as tf
from utils.exp_utils import process_config, create_dirs
from utils.logger import Logger
from utils.data_reader import *
from models.baseline import BaselineModel


def main():
    parser = argparse.ArgumentParser(description="Weather prediction")
    parser.add_argument('--config', default=None, type=str, help='Configuration file')

    # Parse the arguments
    args = parser.parse_args()

    config = process_config(args.config)
    create_dirs([config.summary_dir, config.checkpoint_dir])

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # load the data
    train, val = load_and_preprocess_data(config.data.data_dir)

    # load the word matrix
    embeddings = load_word_embeddings(config.data.data_dir)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    Logger.info('Starting building the model...')
    baseline = BaselineModel(embeddings, config)
    Logger.info('Finished building the model')

    logger = Logger(sess, config.summary_dir)

    baseline.train(sess, train, val, logger)

if __name__ == '__main__':
    main()