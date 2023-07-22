import gc
import os

from utils import parse_config, rename_folder
from preprocess import preprocessing
from signature import generate_signature
from detect import detect_signature
from result import save_result


def experiments(config):
    train_pattern_dict, test_pattern_dict = preprocessing(config)
    signature = generate_signature(train_pattern_dict, config)
    detect_ip_dict = detect_signature(test_pattern_dict, signature, config)
    save_result(detect_ip_dict, train_pattern_dict, test_pattern_dict, config)
    rename_folder('test', config)


if __name__ == '__main__':
    config_path = './CIC.ini'
    config = parse_config(config_path)
    experiments(config)
