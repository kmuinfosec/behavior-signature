import os
import json

from configparser import ConfigParser
from datetime import datetime


def format_date(precision: int) -> str:
    format_order = ["%Y", "%m", "%d", "%H", "%M", "%S"]
    format_str = ""
    for i, fmt in enumerate(format_order):
        if i == precision + 1:
            break
        else:
            format_str += fmt
            if i == 2:
                format_str += '_'

    return datetime.today().strftime(format_str)


def init_save_folder():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(os.path.dirname(base_dir), 'RESULT')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print(save_dir)
    save_path = os.path.join(save_dir, format_date(6))
    os.mkdir(save_path)
    return save_dir, save_path


def parse_config(config_path):
    configparser = ConfigParser()
    if os.path.exists(config_path):
        configparser.read(config_path)
    else:
        raise FileNotFoundError("There is no config File.")
    config = dict()
    config['save_dir'], config['save_path'] = init_save_folder()
    config['data_name'] = configparser['DATA']['DATA_NAME']
    config['train_data_list'] = json.loads(configparser['DATA']['TRAIN_DATA'])
    config['test_data_list'] = json.loads(configparser['DATA']['TEST_DATA'])
    config['train_preprocess'] = configparser['DATA']['TRAIN_PREPROCESSED'].strip('"')
    config['test_preprocess'] = configparser['DATA']['TEST_PREPROCESSED'].strip('"')
    config['dup_ip'] = json.loads(configparser['DATA']['DUPLICATED_IP'])
    config['target_ip'] = json.loads(configparser['DATA']['TRAIN_MALICIOUS_IP'])
    config['label_ip'] = json.loads(configparser['DATA']['TEST_MALICIOUS_IP'])
    config['n_components'] = int(configparser['GMM']['N_COMPONENTS'])
    config['max_iter'] = int(configparser['GMM']['MAX_ITERATION'])
    config['only_benign'] = configparser.getboolean('GMM', 'ONLY_BENIGN')
    config['gmm_preprocess'] = configparser['GMM']['PREPROCESSED_PATH'].strip('"')
    config['confidence'] = float(configparser['GMM']['CONFIDENCE'])
    config['theta1'] = float(configparser['SIGNATURE']['USING_WORD_THETA'])
    config['theta2'] = float(float(configparser['SIGNATURE']['SIGNATURE_CANDIDATE_THETA']))
    config['min_len'] = int(configparser['SIGNATURE']['MIN_LENGTH'])
    config['using_word'] = configparser.getboolean('SIGNATURE', 'USING_WORD')

    return config


def rename_folder(save_name, config):
    if os.path.exists(rf"{config['save_dir']}\{save_name}"):
        save_name = save_name + str(format_date(6))
    os.rename(config['save_path'], rf"{config['save_dir']}\{save_name}")


def make_sig(list_str):
    tmp = list_str.split(' ')
    return [i for i in tmp]


def process_port(port):
    port = int(port)
    if port < 1024:
        return 0
    elif port < 49151:
        return 1
    else:
        return 2