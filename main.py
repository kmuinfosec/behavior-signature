import gc
import os

from utils import parse_config, rename_folder


def experiments(config):
    rename_folder('test', config)


if __name__ == '__main__':
    config_path = './CIC.ini'
    config = parse_config(config_path)
    experiments(config)
