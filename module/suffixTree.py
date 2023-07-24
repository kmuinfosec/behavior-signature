from tqdm.auto import tqdm


class Node:
    def __init__(self, sub="", children=None, i=-1):
        self.sub = sub
        self.ch = children or []
        self.count_value = 1
        self.recent_index = i

    def count(self, idx):
        if self.recent_index != idx:
            self.count_value += 1
            self.recent_index = idx


class SuffixTree:
    def __init__(self, l):
        self.current_len = 0
        self.nodes = [Node()]
        for input_data in tqdm(l, total=len(l)):
            self.add(input_data)

    def add(self, input_data):
        for i in range(len(input_data) - 1, -1, -1):
            target = input_data[i:]
            self._addSuffix(target)
        self.current_len += 1

    def _addSuffix(self, suf):
        idx = self.current_len
        n = 0
        i = 0
        while i < len(suf):
            b = suf[i]
            x2 = 0
            while True:
                children = self.nodes[n].ch
                if x2 == len(children):
                    # no matching child, remainder of suf becomes new node
                    n2 = len(self.nodes)
                    self.nodes.append(Node(suf[i:], [], idx))
                    self.nodes[n].count(idx)
                    self.nodes[n].ch.append(n2)
                    return
                n2 = children[x2]
                if self.nodes[n2].sub[0] == b:
                    break
                x2 = x2 + 1

            # find prefix of remaining suffix in common with child
            sub2 = self.nodes[n2].sub
            # print(suf, suf[i:], sub2)
            if suf[i:] == sub2:
                self.nodes[n].count(idx)
                self.nodes[n2].count(idx)
                return
            j = 0
            flag = False
            if len(sub2) == 1:
                j = j + 1
                self.nodes[n2].count(idx)
            else:
                while j < len(sub2):
                    if i + j > len(suf) - 1 or suf[i + j] != sub2[j]:
                        # split n2
                        n3 = n2
                        # new node for the part in common
                        n2 = len(self.nodes)
                        self.nodes.append(Node(sub2[:j], [n3], idx))
                        self.nodes[n2].count_value = self.nodes[n3].count_value
                        self.nodes[n3].sub = sub2[j:]  # old node loses the part in common
                        self.nodes[n2].recent_index = self.nodes[n3].recent_index
                        self.nodes[n2].count(idx)
                        self.nodes[n].ch[x2] = n2
                        flag = True
                        break  # continue down the tree
                    j = j + 1
            # if flag == False:
            #     self.nodes[n2].count += 1
            i = i + j  # advance past part in common
            self.nodes[n].count(idx)
            n = n2  # continue down the tree

    def get_frequency(self, th=0, k=4) -> list:
        count_dict = dict()

        def _find(n=0, pre=()) -> list:
            if n in first_children:
                bar.update(1)
            children = self.nodes[n].ch

            if self.nodes[n].count_value >= th:
                for i in range(1, len(self.nodes[n].sub) + 1):
                    substring = pre + tuple(self.nodes[n].sub[:i])
                    if substring not in count_dict:
                        if len(substring) >= k:
                            count_dict[substring] = self.nodes[n].count_value
                        continue

                    if self.nodes[n].count_value > count_dict[substring]:
                        count_dict[substring] = self.nodes[n].count_value

                for c in children:
                    _find(c, pre + tuple(self.nodes[n].sub))

        first_children = set(self.nodes[0].ch)
        bar = tqdm(total=len(first_children))
        _find()
        return count_dict