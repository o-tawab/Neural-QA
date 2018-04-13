from pprint import pprint
import json
import os
import sys

from easydict import EasyDict as edict


def get_config_from_json(json_file, verbose=False):
    """
    Parse the arguments of the program
    :return: (config_args)
    :rtype: tuple
    """
    try:
        if json_file is not None:
            with open(json_file, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(json_file), file=sys.stderr)
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
        exit(1)

    config_args = edict(config_args_dict)

    if verbose:
        pprint(config_args)
        print("\n")

    return config_args, config_args_dict


def process_config(jsonfile):
    config, _ = get_config_from_json(jsonfile)
    config.summary_dir = os.path.join("../experiments", config.exp_name, "summaries/")
    config.checkpoint_dir = os.path.join("../experiments", config.exp_name, "checkpoints/")

    return config


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
