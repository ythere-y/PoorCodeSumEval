import spacy

from tree_sitter import Language, Parser

import os
import json
import numpy as np
from copy import deepcopy
import nltk

lang = "java"
tot = 0


def in_oprand(op):
    if (
        op == "+"
        or op == "*"
        or op == "<"
        or op == "<="
        or op == ">"
        or op == ">="
        or op == "=="
        or op == "!="
        or op == "<>"
    ):
        return True
    else:
        return False


def anti_oprand(op):
    tmop = op
    if op == ">":
        tmop = "<"
    elif op == "<":
        tmop = ">"
    elif op == "<=":
        tmop = ">="
    elif op == ">=":
        tmop = "<="
    return tmop


def switch_operand(code_string):
    if lang == "java":
        code_string = "class CLS { \n" + code_string + " \n }"
    LANGUAGE = Language("build/my-languages.so", lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes(code_string, "utf-8"))
    root_node = tree.root_node
    nodes = [root_node]

    ilist = []
    new_code = code_string.split("\n")
    # print(new_code)

    while len(nodes) > 0:
        cur_node = nodes[0]
        nodes = nodes[1:]
        if (
            lang == "python"
            and (
                cur_node.type == "comparison_operator"
                or cur_node.type == "binary_operator"
            )
        ) or (
            (lang == "java" or lang == "go") and (cur_node.type == "binary_expression")
        ):
            s = cur_node.start_point
            e = cur_node.end_point

            tmp = new_code[s[0]]
            stmp = ""
            flag = True
            for child in cur_node.children:
                # if child.type == "identifier" or child.type == "parenthesized_expression" or child.type == "decimal_integer_literal":
                if len(child.type) > 4:
                    sc = child.start_point
                    ec = child.end_point
                    bc = new_code[sc[0]][sc[-1] : ec[-1]]
                    stmp = bc + " " + stmp
                    # print(child.type)
                    # print(bc)
                else:
                    # print(child.type)
                    if not (in_oprand(child.type)):
                        flag = False
                        break
                    tmop = anti_oprand(child.type)
                    stmp = tmop + " " + stmp
            # print(cur_node.type)
            # print(flag)

            if flag:
                tmp = new_code[s[0]]
                tmp = tmp[: s[-1]] + stmp + tmp[e[-1] :]
                new_code[s[0]] = tmp
                if s[0] != e[0]:
                    for i in range(s[0] + 1, e[0]):
                        new_code[i] = ""
                    new_code[e[0]] = new_code[e[0]][e[-1] :]

            # block_swap(new_code, c1, c2)

        else:
            for child in cur_node.children:
                nodes.append(child)

    if lang == "java":
        new_code = new_code[1 : len(new_code) - 1]
    # print("\n".join(new_code))

    return "\n".join(new_code)


def block_swap(new_code, b1, b2):
    s1 = ""
    for i in range(b1.start_point[0], b1.end_point[0]):
        s1 = s1 + new_code[i] + "\n"

    if b1.start_point[0] != b1.end_point[0]:
        s1 = s1 + new_code[b1.end_point[0]][: b1.end_point[1]]
    else:
        s1 = s1 + new_code[b1.end_point[0]][b1.start_point[1] : b1.end_point[1]]

    s2 = ""
    for i in range(b2.start_point[0], b2.end_point[0]):
        s2 = s2 + new_code[i] + "\n"

    if b2.start_point[0] != b2.end_point[0]:
        s2 = s2 + new_code[b2.end_point[0]][: b2.end_point[1]]
    else:
        s2 = s2 + new_code[b2.end_point[0]][b2.start_point[1] : b2.end_point[1]]

    print("s1: {}".format(s1))
    print("s2: {}".format(s2))


def code_tokenizer(code_string):
    code_tokens = []

    def calc_ident(node):
        node_childs = node.children
        if lang == "python" and node.type == "expression_statement":
            if node.children[0].type == "string":
                return
        if lang == "python" and node.type == "comment":
            return
        if lang == "go" and node.type == "comment":
            return

        if (
            not node.children
            or node.type == "interpreted_string_literal"
            or node.type == "string"
        ):
            s = node.start_point
            e = node.end_point
            tmp = new_code[s[0]][s[-1] : e[-1]]
            if (tmp != "") and (tmp != " "):
                code_tokens.append(tmp)
            return
        for i in range(len(node_childs)):
            calc_ident(node_childs[i])

    LANGUAGE = Language("build/my-languages.so", lang)
    if lang == "java":
        code_string = "class CLS { \n" + code_string + " \n }"
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes(code_string, "utf8"))

    new_code = code_string.split("\n")

    calc_ident(tree.root_node)
    # print(len(identi_list))
    if lang == "java":
        code_tokens = code_tokens[3 : len(code_tokens) - 1]

    return code_tokens


def find_ttt(code_string):
    LANGUAGE = Language("build/my-languages.so", lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes(code_string, "utf-8"))
    root_node = tree.root_node
    nodes = [root_node]
    ilist = []
    new_code = code_string.split("\n")

    while len(nodes) > 0:
        cur_node = nodes[0]
        nodes = nodes[1:]

        print(cur_node.type, end=" ")
        for child in cur_node.children:
            print(child.type, end=" ")
            if child.type == "identifier":
                s = child.start_point
                e = child.end_point
                ide = new_code[s[0]][s[-1] : e[-1]]
                print("###{}###".format(ide), end=" ")

            nodes.append(child)

        print("\n")

    return ilist


def check_tokenize():
    path = "D:\\Mywork\\data\\CodeT5\\summarize\\" + lang + "\\train.jsonl"

    with open(path, "r") as pf:
        data = pf.readlines()

    tot = 0
    for d in data:
        line_a = json.loads(d)
        tmp = code_tokenizer(line_a["original_string"])
        ct = line_a["code_tokens"]

        new_ct = list(filter(lambda x: x != "\n", ct))
        ct = new_ct
        new_ct = list(filter(lambda x: x != "\\n", ct))

        if tmp != new_ct:
            print(line_a["original_string"])
            print(tmp)
            print(new_ct)
            print(ct)
            # find_ttt(line_a["original_string"])
            # if tmp[1] == 'load_file': continue
            tot += 1
            return
    print(tot)
    return


def handle():
    # path = "D:\\Mywork\\data\\CodeT5\\summarize\\" + lang + "\\test.jsonl"
    path = os.path.join(
        "D:\\Mywork\\data\\bench\\original\\CSN\\summarize\\java", "train.jsonl"
    )

    with open(path, "r") as pf:
        data = pf.readlines()

    examples = []

    for d in data:
        line_a = json.loads(d)

        tmp = code_tokenizer(switch_operand(line_a["original_string"]))

        line_a["code_tokens"] = tmp

        examples.append(json.dumps(line_a))

    # file_path = "D:\\Mywork\\data\\bench\\structral\\Operand Swap\\" + lang + "\\" + "CSN" + "\\test.jsonl"
    file_path = os.path.join(
        "D:\\Mywork\\data\\curriculum\\CSN\\IOE\\java", "train.jsonl"
    )
    print(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines("\n".join(examples))


def handleDC():
    lang = "java"
    path = "D:\\Mywork\\data\\bench\\original\\DC\\data\\test.json"
    with open(path, "r") as pf:
        data = pf.readlines()

    examples = []

    for d in data:
        line_a = json.loads(d)

        # line_a["code_tokens"] = code_tokenizer(line_a["code"])
        line_a["docstring_tokens"] = nltk.word_tokenize(line_a["nl"])
        # print(line_a["code"])
        tmp = code_tokenizer(switch_operand(line_a["code"]))

        line_a["code_tokens"] = tmp

        examples.append(json.dumps(line_a))

    file_path = (
        "D:\\Mywork\\data\\bench\\structral\\Operand Swap\\"
        + lang
        + "\\"
        + "DC"
        + "\\test.jsonl"
    )
    print(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines("\n".join(examples))


def handleTL():
    lang = "java"
    path = "D:\\Mywork\\data\\bench\\original\\TL\\data\\test.json"
    with open(path, "r") as pf:
        data = pf.readlines()

    examples = []

    for d in data:
        line_a = json.loads(d)

        # line_a["code_tokens"] = code_tokenizer(line_a["code"])
        line_a["docstring_tokens"] = nltk.word_tokenize(line_a["comment"])
        # print(line_a["code"])
        tmp = code_tokenizer(switch_operand(line_a["code"]))

        line_a["code_tokens"] = tmp

        examples.append(json.dumps(line_a))

    file_path = (
        "D:\\Mywork\\data\\bench\\structral\\Operand Swap\\"
        + lang
        + "\\"
        + "TL"
        + "\\test.jsonl"
    )
    print(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines("\n".join(examples))


if __name__ == "__main__":
    # nlp = spacy.load('en_core_web_lg')
    # doc1 = nlp("Downloads Dailymotion videos by URL .")
    # doc2 = nlp("Download a single video .")
    # doc3 = nlp("Main entry point .")

    # print(doc1.similarity(doc2))
    # print(doc2.similarity(doc1))
    # print(nlp.tokenizer("Downloads"))

    handleDC()
    handleTL()

    tmp = """
    def max(a, b): 
        if a > 1 > "a" :
            c = a - (b + b)
        elif a < (a + b):
            c = a * (a + b)
        return c
    """

    # tmp = """
    #     public static int max(int a, int b) {
    #         int r;
    #         if (a > b) {
    #             r = a
    #         } else if (a > (b + b) ) {
    #             r = b
    #         } else {
    #             r = a + b
    #         }
    #         return r
    #     }
    # """
    # tmp = """public static long addCap(long a, long b) {
    #     long u = a + b;
    #     if (u < 0L) {
    #         return Long.MAX_VALUE;
    #     }
    #     return u;
    # }
    # """
    # find_ttt(tmp)

    # code = switch_operand(tmp)
    # print(code)
    # handle()

    # for i in range(0)


# print("""
# def _best_subset(self, n_qubits):\n        \"\"\"Computes the qubit mapping with the best connectivity.\n\n        Args:\n            n_qubits (int): Number of subset qubits to consider.\n\n        Returns:\n            ndarray: Array of qubits to use for best connectivity mapping.\n        \"\"\"\n        if n_qubits == 1:\n            return np.array([0])\n\n        device_qubits = self.coupling_map.size()\n\n        cmap = np.asarray(self.coupling_map.get_edges())\n        data = np.ones_like(cmap[:, 0])\n        sp_cmap = sp.coo_matrix((data, (cmap[:, 0], cmap[:, 1])),\n                                shape=(device_qubits, device_qubits)).tocsr()\n        best = 0\n        best_map = None\n        # do bfs with each node as starting point\n        for k in range(sp_cmap.shape[0]):\n            bfs = cs.breadth_first_order(sp_cmap, i_start=k, directed=False,\n                                         return_predecessors=False)\n\n            connection_count = 0\n            sub_graph = []\n            for i in range(n_qubits):\n                node_idx = bfs[i]\n                for j in range(sp_cmap.indptr[node_idx],\n                               sp_cmap.indptr[node_idx + 1]):\n                    node = sp_cmap.indices[j]\n                    for counter in range(n_qubits):\n                        if node == bfs[counter]:\n                            connection_count += 1\n                            sub_graph.append([node_idx, node])\n                            break\n\n            if connection_count > best:\n                best = connection_count\n                best_map = bfs[0:n_qubits]\n                # Return a best mapping that has reduced bandwidth\n                mapping = {}\n                for edge in range(best_map.shape[0]):\n                    mapping[best_map[edge]] = edge\n                new_cmap = [[mapping[c[0]], mapping[c[1]]] for c in sub_graph]\n                rows = [edge[0] for edge in new_cmap]\n                cols = [edge[1] for edge in new_cmap]\n                data = [1]*len(rows)\n                sp_sub_graph = sp.coo_matrix((data, (rows, cols)),\n                                             shape=(n_qubits, n_qubits)).tocsr()\n                perm = cs.reverse_cuthill_mckee(sp_sub_graph)\n                best_map = best_map[perm]\n        return best_map
# """
# )
