import ahocorasick

from tqdm.auto import tqdm


def detect_signature(test_pattern_dict, signature, config):
    detect_ip_dict = {}

    print("Grow Test Tree..")
    aho_tree = ahocorasick.Automaton()
    for idx, key in enumerate(tqdm(signature)):
        key = '_'.join(key)
        aho_tree.add_word(key, (idx, key))
    aho_tree.make_automaton()

    print("Test")
    for ip in tqdm(test_pattern_dict):
        seq = '_'.join(test_pattern_dict[ip])
        result = list(aho_tree.iter(seq))
        if result:
            result = [i[1][1] for i in result]
            if ip not in detect_ip_dict:
                detect_ip_dict[ip] = []
            detect_ip_dict[ip].append(result)

    print("\nDetect IP :", list(detect_ip_dict.keys()))
    print('\nFalse Positive :', list(set(detect_ip_dict.keys()) - set(config['label_ip'])))
    print('\nFalse negative :', list(set(config['label_ip']) - set(detect_ip_dict.keys())))
    return detect_ip_dict
