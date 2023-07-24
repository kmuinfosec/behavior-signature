import csv

import numpy as np
from suffix_tree import Tree


def save_config(config):
    with open(rf"{config['save_path']}\config.txt", 'w', newline='', encoding='utf-8') as f:
        f.write(f'Dataset : {config["data_name"]}\n')
        f.write(f'GMM components : {config["n_components"]}\n')
        f.write(f'GMM max_iter : {config["max_iter"]}\n')
        f.write(f'GMM only_benign : {config["only_benign"]}\n')
        f.write(f'GMM confidence : {config["confidence"]}\n')
        f.write(f'Theta 1 : {config["theta1"]}\n')
        f.write(f'Theta 2 : {config["theta2"]}\n')
        f.write(f'Min Length : {config["min_len"]}')
        f.write(f'\nDATA PATH\n')
        if config['train_preprocess']:
            f.write(f'Train Preprocessed by {config["train_preprocess"]}')
        else:
            f.write(f'Train Path : {config["train_data_list"]}')
        if config['test_preprocess']:
            f.write(f'Test Preprocessed by {config["test_preprocess"]}')
        else:
            f.write(f'Test Path : {config["test_data_list"]}')
        if config['gmm_preprocess']:
            f.write(f'GMM Preprocessed by {config["gmm_preprocess"]}')



def save_confusion(detect_ip_dict, test_pattern_dict, config):
    with open(rf"{config['save_path']}\confusion.txt", 'w', newline='', encoding='utf-8') as f:
        f.write(f"True Positive({len(set(detect_ip_dict.keys()) & set(config['label_ip']))}) : {list(set(detect_ip_dict.keys()) & set(config['label_ip']))}\n")
        f.write(f"True Negative({len(set(test_pattern_dict.keys()) - set(detect_ip_dict.keys())- set(config['label_ip']))}\n")
        f.write(f"False Positive({len(set(detect_ip_dict.keys()) - set(config['label_ip']))}) : {list(set(detect_ip_dict.keys()) - set(config['label_ip']))}\n")
        f.write(f"False Negative({len(set(config['label_ip']) - set(detect_ip_dict.keys()))}) : {list(set(config['label_ip']) - set(detect_ip_dict.keys()))}\n")


def save_fp(detect_ip_dict, config):
    fp_signature = []
    with open(rf"{config['save_path']}\false_positive.csv", 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['IP', 'SIGNATURE'])
        for ip in list(set(detect_ip_dict.keys()) - set(config['label_ip'])):
            wr.writerow([ip, detect_ip_dict[ip]])
            fp_signature += detect_ip_dict[ip]
    return fp_signature


def debug_fp(fp_signature, config):
    fp_sig, fp_count = np.unique(list(map(lambda x: ' '.join(x), fp_signature)), return_counts=True)
    with open(rf"{config['save_path']}\fp_count.csv", 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['COUNT', 'SIGNATURE'])
        for i in range(len(fp_sig)):
            wr.writerow([fp_count[i], fp_sig[i]])


def save_analysis(train_pattern_dict, test_pattern_dict, config):
    tmp_tree = Tree()
    for ip in config['target_ip']:
        tmp_tree.add(ip, train_pattern_dict[ip])
    for idx, num in enumerate(range(100, 100 + len(config['label_ip']))):
        tmp_tree.add(num, test_pattern_dict[config['label_ip'][idx]])
    with open(rf"{config['save_path']}\analysis.csv", 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['COUNT', 'SIGNATURE'])
        for C, path in sorted(tmp_tree.maximal_repeats()):
            if C >= (len(config['target_ip']) + len(config['label_ip'])) * 1.0:
                wr.writerow([C, path])


def save_result(detect_ip_dict, train_pattern_dict, test_pattern_dict, config):
    save_config(config)
    save_confusion(detect_ip_dict, test_pattern_dict, config)
    fp_signature = save_fp(detect_ip_dict, config)
    debug_fp(fp_signature, config)
    save_analysis(train_pattern_dict, test_pattern_dict, config)
