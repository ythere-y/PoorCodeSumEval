import spacy

import random

from tree_sitter import Language, Parser

import os
import json
import numpy as np
from copy import deepcopy
import nltk

# from ttss import find_java_ide
from only_func import find_python_ide, find_java_ide, find_go_ide


lang = "go"
tot = 0


def get_str(new_code, node):
    s = node.start_point
    e = node.end_point
    ide = new_code[s[0]][s[-1] : e[-1]]
    # print(ide)
    return ide


def get_tab(new_code, line):
    tmp = ""
    k = 0
    while new_code[line][k] == "\t" or new_code[line][k] == " ":
        tmp += new_code[line][k]
        k += 1
    # print(new_code[line][k])
    # print("tab_{}:{}".format(k, tmp))
    return tmp


def code_update(code, s, e, stmp):
    new_code = code
    l = s[0]
    if s[0] == e[0]:
        new_code[l] = code[l][: s[-1]] + stmp + code[l][e[-1] :]
    else:
        new_code[l] = new_code[l][: s[-1]] + stmp
        for i in range(s[0] + 1, e[0]):
            new_code[i] = ""
        # print(code)
        tmp = get_tab(code, e[0])
        new_code[e[0]] = tmp + new_code[e[0]][e[-1] :]
    return new_code


def branch_inject(code_string):
    if lang == "java":
        code_string = "class CLS { \n" + code_string + " \n }"

    LANGUAGE = Language("build/my-languages.so", lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes(code_string, "utf-8"))
    root_node = tree.root_node
    nodes = [root_node]

    ilist = []
    if lang != "java" and lang == "go":
        code_string = code_string + " \n }"
    new_code = code_string.split("\n")
    # print(new_code)

    body_list = []

    while len(nodes) > 0:
        cur_node = nodes[0]
        nodes = nodes[1:]
        if cur_node.type == "ERROR":
            print(code_string)

        if lang == "java":
            s = cur_node.start_point
            e = cur_node.end_point
            if (
                cur_node.type == "method_declaration"
                or cur_node.type == "constructor_declaration"
            ):
                for child in cur_node.children:
                    if child.type == "block":
                        for gchild in cur_node.children:
                            if gchild.type == "{":
                                s = gchild.start_point

                sl = s[0]

                ttt = random.randrange(0, 2, 1)
                if ttt == 0:
                    new_code[sl] = new_code[sl] + "\n" + " if (true) { \n"
                else:
                    new_code[sl] = (
                        new_code[sl] + "\n" + " if (false) { int i = 1; } else { \n"
                    )

        if lang == "go":
            s = cur_node.start_point
            e = cur_node.end_point
            if (
                cur_node.type == "function_declaration"
                or cur_node.type == "method_declaration"
            ):
                for child in cur_node.children:
                    if child.type == "block":
                        for gchild in cur_node.children:
                            if gchild.type == "{":
                                s = gchild.start_point

                sl = s[0]

                ttt = random.randrange(0, 2, 1)
                if ttt == 0:
                    new_code[sl] = new_code[sl] + "\n" + " if (true) { \n"
                else:
                    new_code[sl] = (
                        new_code[sl] + "\n" + " if (false) { i := 1 } else { \n"
                    )

        if lang == "python":
            if cur_node.type == "function_definition":
                for child in cur_node.children:
                    if child.type == "block":
                        s = child.start_point

                sl = s[0]

                for i in range(sl, len(new_code)):
                    new_code[i] = "\t" + new_code[i]

                ttt = random.randrange(0, 2, 1)
                if ttt == 0:
                    new_code[sl] = "\tif (True): \n" + new_code[sl]
                else:
                    new_code[sl] = (
                        "\tif (False): \n\t\t i = 1 \n\telse: \n" + new_code[sl]
                    )

        for child in cur_node.children:
            nodes.append(child)

    # print(body_list)
    # if body_list != []: print(1)

    if lang == "java":
        new_code = new_code[1 : len(new_code)]
    # print("\n".join(new_code))

    return "\n".join(new_code)


def change_body():
    ttt = random.randrange(0, 2, 1)
    ret = ""

    if lang == "java":
        if ttt == 0:
            # ret = "if (pingInterval == newPingInterval)\n            return;\n\n        if (newPingInterval > 0)\n            enableExecutorService();\n\n        pingInterval = newPingInterval;\n\n        if (pingInterval < 0) {\n            stopPinging();\n        } else {\n            schedulePingServerTask();\n        }"
            ret = "return Math.sin(Math.PI * x) / (Math.PI * x);"
        else:
            ret = "return !Strings.isNullOrEmpty(objectName)\n        && objectName.endsWith(GoogleCloudStorage.PATH_DELIMITER);"
            # ret = "int[] rowcol = CellUtility.getRowColFromComponentAttributes(target); \nint row = rowcol[0]; \nint col = rowcol[1]; \nreturn validateWithRowColInCurrentPage(row, col, true);"

    if lang == "python":
        if ttt == 0:
            ret = "h = line(y, self._effective_thickness(p), 0.0)\n\t\treturn h.sum()"
        else:
            ret = "super(Composite,self).state_pop()\n\t\tfor gen in self.generators:\n\t\t\tgen.state_pop()"

    if lang == "go":
        if ttt == 0:
            ret = "return storeTrustedKey(ks.LocalRootPath, r)"
        else:
            ret = "return filepath.Join(exitedGarbageDir(p.dataDir), p.UUID.String())"
    return ret


def branch_body_inject(code_string):
    if lang == "java":
        code_string = "class CLS { \n" + code_string + " \n }"

    LANGUAGE = Language("build/my-languages.so", lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes(code_string, "utf-8"))
    root_node = tree.root_node
    nodes = [root_node]

    ilist = []
    if lang != "java" and lang == "go":
        code_string = code_string + " \n }"
    new_code = code_string.split("\n")
    # print(new_code)

    nbody = change_body()

    body_list = []

    while len(nodes) > 0:
        cur_node = nodes[0]
        nodes = nodes[1:]
        if cur_node.type == "ERROR":
            print(code_string)

        if lang == "java":
            if (
                cur_node.type == "method_declaration"
                or cur_node.type == "constructor_declaration"
            ):
                s = cur_node.start_point
                e = cur_node.end_point
                for child in cur_node.children:
                    if child.type == "block":
                        for gchild in child.children:
                            if gchild.type == "{":
                                s = gchild.start_point
                            elif gchild.type == "}":
                                e = gchild.end_point

                sl = s[0]
                ttt = random.randrange(0, 2, 1)
                if ttt == 0:
                    new_code[sl] = new_code[sl] + "\n" + " if (true) { \n"
                    new_code[e[0]] = new_code[e[0]] + "\n else { \n" + nbody + "\n }"
                else:
                    new_code[sl] = (
                        new_code[sl]
                        + "\n"
                        + " if (false) { \n"
                        + nbody
                        + " \n } else { \n"
                    )

        if lang == "go":
            if (
                cur_node.type == "function_declaration"
                or cur_node.type == "method_declaration"
            ):
                s = cur_node.start_point
                e = cur_node.end_point
                for child in cur_node.children:
                    if child.type == "block":
                        for gchild in child.children:
                            if gchild.type == "{":
                                s = gchild.start_point
                            elif gchild.type == "}":
                                e = gchild.end_point

                sl = s[0]

                ttt = random.randrange(0, 2, 1)
                if ttt == 0:
                    new_code[sl] = new_code[sl] + "\n" + " if (true) { \n"
                    new_code[e[0]] = new_code[e[0]] + "\n else { \n" + nbody + "\n }"
                else:
                    new_code[sl] = (
                        new_code[sl]
                        + "\n"
                        + " if (false) { \n"
                        + nbody
                        + " \n } else { \n"
                    )

        if lang == "python":
            if cur_node.type == "function_definition":
                for child in cur_node.children:
                    if child.type == "block":
                        s = child.start_point

                sl = s[0]

                for i in range(sl, len(new_code)):
                    new_code[i] = "\t" + new_code[i]

                ttt = random.randrange(0, 2, 1)
                if ttt == 0:
                    new_code[sl] = "\tif (True): \n" + new_code[sl]
                    new_code[-1] = new_code[-1] + "\n\telse: \n\t\t" + nbody
                else:
                    new_code[sl] = (
                        "\tif (False): \n\t\t" + nbody + " \n\telse: \n" + new_code[sl]
                    )

        for child in cur_node.children:
            nodes.append(child)

    # print(body_list)
    # if body_list != []: print(1)

    if lang == "java":
        new_code = new_code[1 : len(new_code)]
    # print("\n".join(new_code))

    return "\n".join(new_code)


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
        # if cur_node.type == "method_declaration" or cur_node.type == "constructor_declaration":
        #     for child in cur_node.children:
        #         if child.type == "block":

        #             for gchild in cur_node.children:
        #                 if gchild.type == "{":
        #                    print("!!!!!!!!")

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


def handle():
    # lang = "python"
    path = "D:\\Mywork\\data\\CodeT5\\summarize\\" + lang + "\\test.jsonl"

    with open(path, "r") as pf:
        data = pf.readlines()

    examples = []
    tot = 0

    for d in data:
        line_a = json.loads(d)

        tmp = code_tokenizer(branch_body_inject(line_a["original_string"]))

        line_a["code_tokens"] = tmp

        examples.append(json.dumps(line_a))
        if len(line_a["code_tokens"]) > 120:
            tot += 1
    print(tot)

    file_path = (
        "D:\\Mywork\\data\\bench\\structral\\BranchBody\\"
        + lang
        + "\\"
        + "CSN"
        + "\\test.jsonl"
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

    tot = 0
    for d in data:
        line_a = json.loads(d)

        # line_a["code_tokens"] = code_tokenizer(line_a["code"])
        line_a["docstring_tokens"] = nltk.word_tokenize(line_a["nl"])
        # print(line_a["code"])
        tmp = code_tokenizer(branch_body_inject(line_a["code"]))

        line_a["code_tokens"] = tmp

        examples.append(json.dumps(line_a))
        if len(line_a["code_tokens"]) > 120:
            tot += 1
    print(tot)

    file_path = (
        "D:\\Mywork\\data\\bench\\structral\\BranchBody\\"
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
        tmp = code_tokenizer(branch_body_inject(line_a["code"]))

        line_a["code_tokens"] = tmp

        examples.append(json.dumps(line_a))

    file_path = (
        "D:\\Mywork\\data\\bench\\structral\\BranchBody\\"
        + lang
        + "\\"
        + "TL"
        + "\\test.jsonl"
    )
    print(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines("\n".join(examples))


def switch_ide(original_ide):
    new_dict = dict()
    cc = 0
    for ide in original_ide:
        tmp = "v" + str(cc)
        cc += 1
        new_dict.update({ide: tmp})
    return new_dict


def handlefunc():
    func_dict = {"python": find_python_ide, "java": find_java_ide, "go": find_go_ide}
    path = "D:\\Mywork\\data\\CodeT5\\summarize\\" + lang + "\\test.jsonl"

    with open(path, "r") as pf:
        data = pf.readlines()

    examples = []
    tot = 0

    for d in data:
        line_a = json.loads(d)

        idict = switch_ide(func_dict[lang](line_a["original_string"]))

        tmp = code_tokenizer(branch_body_inject(line_a["original_string"]))
        line_a["code_tokens"] = tmp

        if idict != None:
            p = False
            for k, d in enumerate(line_a["code_tokens"]):
                if d in idict:
                    line_a["code_tokens"][k] = idict[d]
                    p = True
            if not p:
                print(line_a)

                return

            sentences = nltk.sent_tokenize(" ".join(line_a["docstring_tokens"]))
            sdict = dict()
            for sent in sentences:
                tmp = nltk.pos_tag(nltk.word_tokenize(sent))
                for t in tmp:
                    if t[1][0] == "N":
                        sdict.update({t[0]: 1})

            p = False
            for k, d in enumerate(line_a["docstring_tokens"]):
                if (d in idict) and (d in sdict):
                    p = True
                    line_a["docstring_tokens"][k] = idict[d]

        examples.append(json.dumps(line_a))
        if len(line_a["code_tokens"]) > 120:
            tot += 1
    print(tot)

    file_path = (
        "D:\\Mywork\\data\\bench\\structral\\branchfuncbody\\"
        + lang
        + "\\"
        + "CSN"
        + "\\test.jsonl"
    )
    print(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines("\n".join(examples))


if __name__ == "__main__":
    handlefunc()

    # tmp = """
    # protected final void fastPathOrderedEmit(U value, boolean delayError, Disposable disposable) {\n        final Observer<? super V> observer = downstream;\n        final SimplePlainQueue<U> q = queue;\n\n        if (wip.get() == 0 && wip.compareAndSet(0, 1)) {\n            if (q.isEmpty()) {\n                accept(observer, value);\n                if (leave(-1) == 0) {\n                    return;\n                }\n            } else {\n                q.offer(value);\n            }\n        } else {\n            q.offer(value);\n            if (!enter()) {\n                return;\n            }\n        }\n        QueueDrainHelper.drainLoop(q, observer, delayError, disposable, this);\n    }
    # """
    # print(branch_body_inject(tmp))
