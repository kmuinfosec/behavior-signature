import os
import pickle

import numpy as np

from utils import process_port
from tqdm.auto import tqdm
from collections import Counter
from module.patterning import GMM_Pattering


def b_flow(config, phase='train'):
    ret_data = []
    ret_key = []
    target_ip = set(config['dup_ip'])
    if phase == 'train':
        mal_ip = set(config['target_ip'])
    else:
        mal_ip = set()
    data_path = config['train_data_list'] if phase == 'train' else config['test_data_list']
    ben_idx = []
    index = 0
    for idx, file in enumerate(data_path):
        change_flag = {}
        with open(file, 'r', encoding='utf-8') as f:
            col = f.readline().strip().split(',')
            column_index = {i: idx for idx, i in enumerate(col)}
            for tmp_flow in tqdm(f.readlines()):
                flow = tmp_flow.strip().split(',')
                ts = flow[column_index['first']]
                if flow[column_index['prot']] != '6' and flow[column_index['prot']] != '17':
                    continue
                sa, da = flow[column_index['source']], flow[column_index['destination']]
                if sa not in mal_ip and da not in mal_ip:
                    ben_idx.append(index)
                    ben_idx.append(index + 1)
                index += 2
                if sa in target_ip:
                    change_flag[sa] = idx
                    sa = idx
                if da in target_ip:
                    change_flag[da] = idx
                    da = idx
                ret_key.append(f"{sa}_{ts}")
                ret_data.append(
                    [process_port(flow[column_index['src_port']]), process_port(flow[column_index['dst_port']]),
                     int(float(flow[column_index['prot']])), int(float(flow[column_index['duration']])),
                     int(float(flow[column_index['in_pkts']])), int(float(flow[column_index['out_pkts']])),
                     int(float(flow[column_index['in_bytes']])), int(float(flow[column_index['out_bytes']]))])

                ret_key.append(f"{da}_{ts}")
                ret_data.append(
                    [process_port(flow[column_index['dst_port']]), process_port(flow[column_index['src_port']]),
                     int(float(flow[column_index['prot']])), int(float(flow[column_index['duration']])),
                     int(float(flow[column_index['out_pkts']])), int(float(flow[column_index['in_pkts']])),
                     int(float(flow[column_index['out_bytes']])), int(float(flow[column_index['in_bytes']]))])

        if change_flag:
            print(change_flag, 'Changed')

    if mal_ip and phase == 'train':
        with open(rf'{config["save_path"]}\ben_feature.pkl', 'wb') as f:
            pickle.dump(np.array(ret_data)[ben_idx].tolist(), f)

    with open(rf'{config["save_path"]}\{phase}_feature.pkl', 'wb') as f:
        pickle.dump(ret_data, f)

    with open(rf'{config["save_path"]}\{phase}_key.pkl', 'wb') as f:
        pickle.dump(ret_key, f)


def make_pattern_dict(key_list, data_list, using_port=False):
    ret_pattern_dict = {}
    for idx, key in enumerate(key_list):
        ip, ts = key.split('_')
        if ip not in ret_pattern_dict:
            ret_pattern_dict[ip] = []
        if using_port:
            ret_pattern_dict[ip].append(data_list[idx])
        else:
            ret_pattern_dict[ip].append(data_list[idx][4:])
    return ret_pattern_dict


def load_data(config, phase):
    if config[f'{phase}_preprocess'] == '':
        b_flow(config, phase)
        with open(rf'{config["save_path"]}\{phase}_feature.pkl', 'rb') as f:
            data_raw = pickle.load(f)

        with open(rf'{config["save_path"]}\{phase}_key.pkl', 'rb') as f:
            data_key = pickle.load(f)
    else:
        with open(rf'{config[f"{phase}_preprocess"]}\{phase}_feature.pkl', 'rb') as f:
            data_raw = pickle.load(f)

        with open(rf'{config[f"{phase}_preprocess"]}\{phase}_key.pkl', 'rb') as f:
            data_key = pickle.load(f)
    return data_raw, data_key


def make_using_word(train_pattern_dict, config):
    if config['theta1'] < 1:
        th = len(config['target_ip']) * config['theta']
    else:
        th = config['theta1']
    using_word = set()
    target_words = []
    for idx, ip in enumerate(tqdm(config['target_ip'])):
        target_words += list(set(train_pattern_dict[ip]))

    word_count_dict = dict(Counter(target_words))
    for word in word_count_dict:
        if word_count_dict[word] >= th:
            using_word.add(word)
    return using_word


def remake_pattern_dict(pattern_dict:dict, using_word:set):
    ret_dict = {}
    for ip in tqdm(pattern_dict):
        payloads = pattern_dict[ip]
        ret_dict[ip] = []
        for word in payloads:
            if word in using_word:
                ret_dict[ip].append(word)
    return ret_dict


def preprocessing(config):
    train_raw, train_key = load_data(config, 'train')
    test_raw, test_key = load_data(config, 'test')
    if config['only_benign']:
        with open(rf'{config["save_path"]}\ben_feature.pkl', 'rb') as f:
            gmm_data = pickle.load(f)
    else:
        gmm_data = train_raw

    if config['gmm_preprocess']:
        print("Load Model")
        with open(rf"{config['gmm_preprocess']}\gmm.pkl", 'rb') as f:
            pattern_gmm = pickle.load(f)
    else:
        pattern_gmm = GMM_Pattering(ignore_idx=[0,1,2], random_seed=43, covariance_type='spherical',
                                    n_components=config['n_components'], max_iter=config['max_iter'])
        pattern_gmm.fit(gmm_data)
        with open(rf"{config['save_path']}\gmm.pkl", 'wb') as f:
            pickle.dump(pattern_gmm, f)

    train_data = pattern_gmm.transform_tokenize(train_raw, confidence=config['confidence'])
    test_data = pattern_gmm.transform_tokenize(test_raw, confidence=config['confidence'])

    train_pattern_dict = make_pattern_dict(train_key, train_data, using_port=False)
    test_pattern_dict = make_pattern_dict(test_key, test_data, using_port=False)

    if config['using_word']:
        using_word = make_using_word(train_pattern_dict, config)
        train_pattern_dict = remake_pattern_dict(train_pattern_dict, using_word)
        test_pattern_dict = remake_pattern_dict(test_pattern_dict, using_word)

    return train_pattern_dict, test_pattern_dict

