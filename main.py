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
    rename_folder(f'{config["data_name"]}_confidence_{config["confidence"]}_min_{config["min_len"]}',
                  config)


if __name__ == '__main__':
    config_path = './CIC.ini'
    config = parse_config(config_path)
    for confi in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 2.5, 6]:
        for min_len in [2,4,6]:
            config['min_len'] = min_len
            config['confidence'] = confi
            print(f"Min Length : {min_len} Confidence : {confi}")
            experiments(config)
