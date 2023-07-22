import ahocorasick

from suffix_tree import Tree
from tqdm.auto import tqdm
from module.suffixTree import SuffixTree


def erase_signature(cand_sig, ip_pattern_dict, target_ip):
    ret_sig = []
    target_ip = set(target_ip)
    tmp_tree = Tree()
    print("Grow Tree...")
    for idx, ip in enumerate(tqdm(ip_pattern_dict)):
        if ip not in target_ip:
            tmp_tree.add(idx, ip_pattern_dict[ip])

    print("Erase WhiteList...")
    for sig in tqdm(cand_sig):
        if not tmp_tree.find(sig):
            ret_sig.append(sig)
    return ret_sig


def select_signature(sig_candi: list):
    sorted_list = sig_candi[::-1]
    sig_list = []
    tmp_tree = Tree()
    for idx, sig in enumerate(tqdm(sorted_list)):
        if not tmp_tree.find(sig):
            sig_list.append(sig)
            tmp_tree.add(idx, sig)
    return sig_list


def erase_signature(cand_sig, ip_pattern_dict, target_ip):
    target_ip = set(ip_pattern_dict.keys()) - set(target_ip)
    str_benign_flows = ["_".join(ip_pattern_dict[ip]) for ip in target_ip]

    candidate_set = list(["_".join(c) for c in cand_sig])
    false_sig = set()
    ret_sig = []

    aho_tree = ahocorasick.Automaton()
    for idx, sig in enumerate(tqdm(candidate_set)):
        aho_tree.add_word(sig, (idx, sig))
    aho_tree.make_automaton()
    for i, flow in enumerate(tqdm(str_benign_flows)):
        result = list(aho_tree.iter(flow))
        for _, data in result:
            false_sig.add(data[0])

    for i in range(len(cand_sig)):
        if i not in false_sig:
            ret_sig.append(cand_sig[i])

    return ret_sig


def generate_signature(train_pattern_dict, config):
    signature_candidate = set(
        SuffixTree([train_pattern_dict[ip] for ip in config['target_ip']]).get_frequency(k=config['min_len'],
                                                                                         th=len(
                                                                                             config['target_ip']) *
                                                                                            config['theta2']).keys())
    print("\nSignature Candidate Count :", len(signature_candidate))
    print("============ Delete WhiteList Signature ============")
    confirm_sig = erase_signature(list(signature_candidate), train_pattern_dict, config['target_ip'])
    print("Confirm Signature Count :", len(confirm_sig))
    return confirm_sig