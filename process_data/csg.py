from tree_sitter import Language, Parser

import random

import os
import json
import numpy as np
from copy import deepcopy
import nltk
import difflib
from datasets import load_dataset

# import bleu


python_dict = [
    ("data", 14119),
    ("name", 13844),
    ("value", 9501),
    ("result", 8697),
    ("cls", 8038),
    ("path", 7955),
    ("key", 6608),
    ("url", 6548),
    ("response", 6343),
    ("args", 6062),
    ("msg", 4970),
    ("filename", 4933),
    ("params", 4810),
    ("x", 4638),
    ("request", 4637),
    ("ret", 4197),
    ("obj", 3872),
    ("res", 3413),
    ("message", 3359),
    ("f", 3320),
    ("s", 3285),
    ("start", 3218),
    ("r", 3181),
    ("text", 3137),
    ("cmd", 2977),
    ("config", 2940),
    ("func", 2922),
    ("n", 2863),
    ("y", 2797),
    ("index", 2790),
    ("context", 2752),
    ("d", 2697),
    ("node", 2628),
    ("p", 2569),
    ("query", 2436),
    ("line", 2432),
    ("timeout", 2385),
    ("out", 2332),
    ("output", 2231),
    ("user", 2230),
    ("headers", 2209),
    ("size", 2115),
    ("method", 2101),
    ("i", 2098),
    ("kwargs", 2069),
    ("m", 2056),
    ("a", 2052),
    ("values", 1994),
    ("b", 1951),
    ("results", 1909),
    ("val", 1900),
    ("c", 1874),
    ("event", 1862),
    ("model", 1853),
    ("content", 1815),
    ("t", 1808),
    ("target", 1767),
    ("lines", 1766),
    ("options", 1762),
    ("status", 1739),
    ("count", 1707),
    ("parser", 1697),
    ("source", 1694),
    ("end", 1665),
    ("body", 1660),
    ("prefix", 1657),
    ("version", 1647),
    ("offset", 1630),
    ("v", 1612),
    ("match", 1606),
    ("command", 1602),
    ("payload", 1597),
    ("item", 1574),
    ("state", 1559),
    ("ctx", 1552),
    ("get", 1543),
    ("session", 1531),
    ("client", 1513),
]


java_dict = [
    ("i", 14990),
    ("value", 7816),
    ("result", 6985),
    ("name", 6818),
    ("key", 4851),
    ("type", 3949),
    ("response", 3691),
    ("index", 3359),
    ("request", 2997),
    ("context", 2831),
    ("c", 2798),
    ("path", 2752),
    ("sb", 2518),
    ("obj", 2419),
    ("id", 2413),
    ("data", 2339),
    ("get", 2237),
    ("s", 2154),
    ("x", 1976),
    ("start", 1942),
    ("node", 1907),
    ("file", 1906),
    ("message", 1904),
    ("url", 1877),
    ("args", 1871),
    ("j", 1846),
    ("e", 1819),
    ("length", 1792),
    ("b", 1772),
    ("n", 1755),
    ("list", 1727),
    ("service", 1727),
    ("builder", 1724),
    ("client", 1679),
    ("size", 1635),
    ("count", 1563),
    ("clazz", 1553),
    ("p", 1537),
    ("offset", 1490),
    ("out", 1488),
    ("t", 1481),
    ("ret", 1476),
    ("v", 1450),
    ("buffer", 1417),
    ("y", 1380),
    ("run", 1371),
    ("input", 1362),
    ("msg", 1360),
    ("resource", 1315),
    ("len", 1314),
    ("m", 1281),
    ("source", 1270),
    ("values", 1253),
    ("text", 1248),
    ("map", 1244),
    ("params", 1234),
    ("in", 1231),
    ("method", 1227),
    ("config", 1223),
    ("add", 1195),
    ("event", 1190),
    ("o", 1179),
    ("target", 1170),
    ("parent", 1166),
    ("field", 1152),
    ("buf", 1150),
    ("entry", 1143),
    ("a", 1132),
    ("r", 1121),
    ("pos", 1071),
    ("object", 1065),
    ("state", 1013),
    ("query", 1005),
    ("end", 998),
    ("uri", 972),
    ("f", 966),
    ("element", 952),
    ("resp", 950),
    ("it", 943),
    ("session", 920),
    ("str", 914),
    ("properties", 912),
    ("line", 907),
    ("reader", 897),
    ("val", 892),
    ("filter", 875),
    ("qPath", 874),
    ("bytes", 860),
    ("root", 846),
    ("listener", 838),
    ("k", 827),
    ("writer", 810),
    ("next", 805),
    ("d", 801),
    ("output", 797),
    ("create", 787),
    ("ctx", 783),
    ("item", 783),
    ("set", 774),
    ("className", 772),
    ("options", 757),
    ("row", 743),
    ("call", 736),
    ("res", 735),
    ("methodName", 720),
    ("is", 719),
    ("update", 713),
    ("token", 712),
    ("array", 708),
    ("parameters", 705),
    ("prefix", 702),
    ("width", 694),
    ("status", 692),
    ("iterator", 691),
    ("serviceName", 682),
    ("position", 676),
    ("defaultValue", 675),
    ("connection", 672),
    ("stream", 670),
    ("info", 662),
    ("idx", 655),
    ("iter", 654),
    ("model", 648),
    ("props", 645),
    ("l", 632),
    ("version", 629),
    ("table", 628),
    ("first", 623),
    ("content", 618),
    ("close", 616),
    ("entity", 616),
    ("init", 612),
    ("req", 612),
    ("from", 610),
    ("format", 609),
    ("remove", 604),
    ("max", 601),
    ("execute", 599),
    ("task", 596),
    ("property", 579),
    ("factory", 576),
    ("results", 575),
    ("handler", 570),
    ("error", 568),
    ("fileName", 566),
    ("user", 566),
    ("parse", 564),
    ("instance", 553),
    ("tmp", 547),
    ("pattern", 541),
    ("child", 541),
    ("matcher", 537),
    ("height", 535),
    ("date", 535),
    ("delete", 535),
    ("action", 532),
    ("attributes", 528),
    ("src", 528),
    ("write", 527),
    ("read", 524),
    ("current", 518),
    ("password", 517),
    ("conf", 513),
    ("port", 513),
    ("resources", 512),
    ("timeout", 501),
    ("json", 493),
    ("ch", 493),
    ("other", 490),
    ("fieldName", 489),
    ("tag", 484),
    ("w", 482),
    ("dir", 482),
    ("string", 478),
    ("keys", 475),
    ("propertyName", 474),
    ("g", 472),
    ("scope", 472),
    ("doc", 470),
    ("conn", 470),
    ("process", 469),
    ("fields", 467),
    ("to", 466),
    ("address", 466),
    ("description", 463),
    ("attribute", 462),
    ("filename", 457),
    ("code", 456),
    ("record", 453),
    ("column", 453),
    ("body", 453),
    ("apply", 452),
    ("parser", 447),
    ("formatter", 446),
    ("headers", 445),
    ("put", 445),
    ("component", 443),
    ("callback", 436),
    ("h", 435),
    ("ref", 434),
]

go_dict = [
    ("err", 63652),
    ("s", 26468),
    ("c", 21460),
    ("ctx", 14832),
    ("ok", 13520),
    ("v", 13152),
    ("r", 12447),
    ("p", 9861),
    ("m", 9121),
    ("name", 7999),
    ("b", 7746),
    ("i", 7560),
    ("t", 7202),
    ("f", 6121),
    ("w", 5897),
    ("key", 5756),
    ("out", 5263),
    ("n", 5217),
    ("d", 5026),
    ("a", 4895),
    ("e", 4588),
    ("in", 4553),
    ("data", 4274),
    ("id", 4269),
    ("args", 4249),
    ("l", 4130),
    ("resp", 4041),
    ("o", 3986),
    ("req", 3886),
    ("h", 3838),
    ("path", 3834),
    ("value", 3402),
    ("client", 3046),
    ("result", 3008),
    ("opts", 2817),
    ("g", 2462),
    ("u", 2325),
    ("config", 2310),
    ("ret", 2287),
    ("res", 2262),
    ("val", 2131),
    ("buf", 2118),
    ("db", 2104),
    ("k", 1969),
    ("msg", 1925),
    ("cmd", 1858),
    ("params", 1777),
    ("conn", 1719),
    ("options", 1706),
    ("q", 1696),
    ("x", 1629),
    ("info", 1570),
    ("obj", 1499),
    ("tx", 1492),
    ("cfg", 1430),
    ("addr", 1424),
    ("New", 1415),
    ("j", 1404),
    ("body", 1372),
    ("url", 1353),
    ("index", 1283),
    ("fn", 1217),
    ("logger", 1179),
    ("node", 1120),
    ("size", 1111),
    ("file", 1106),
    ("format", 1060),
    ("src", 1052),
    ("api", 1046),
    ("response", 1041),
    ("found", 1038),
    ("token", 1026),
    ("timeout", 1010),
    ("request", 1002),
    ("ret0", 994),
    ("cancel", 990),
    ("results", 970),
    ("query", 966),
    ("state", 953),
]


# from spacy_t import find_sim_list

lang = "java"
# dataset = 'TL'
tot = 0


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


# METHOD:替换变量名
def switch_ide(original_ide):
    # if len(original_ide) < 1: print(1)
    new_dict = dict()
    cc = 0
    for ide in original_ide:
        tmp = "v" + str(cc)
        cc += 1
        new_dict.update({ide: tmp})
    # print(new_dict)
    return new_dict


# METHOD:随机打乱
def shuffle_ide(original_ide):
    # print(original_ide)
    if original_ide == []:
        return None
    new_dict = dict(zip(original_ide[:-1], original_ide[1:]))
    new_dict.update({original_ide[-1]: original_ide[0]})
    # print(new_dict)
    return new_dict


# METHOD:筛选出高频词
def high_ide(original_ide):
    tdict = dict()
    if lang == "python":
        tdict = dict(python_dict)
    elif lang == "java":
        tdict = dict(java_dict)
    elif lang == "go":
        tdict = dict(go_dict)
    # print(tdict)

    ilist = list(tdict.keys())
    ilist = ilist[: len(original_ide)]
    random.shuffle(ilist)

    # print(ilist)

    # print(len(original_ide))

    new_dict = dict()
    cc = 0
    for ide in original_ide:
        tmp = ilist[cc]
        cc += 1
        new_dict.update({ide: tmp})
    # print(new_dict)
    return new_dict


def find_python_ide(code_string):
    LANGUAGE = Language("D:\\Mywork\\test_results\\build\\my-languages.so", "python")
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
        # for child in cur_node.children:
        #     nodes.append(child)
        if cur_node.type == "function_definition":
            for child in cur_node.children:
                if child.type == "identifier":
                    s = child.start_point
                    e = child.end_point
                    ide = new_code[s[0]][s[-1] : e[-1]]
                    if not ide in ilist:
                        ilist.append(ide)

                elif child.type == "parameters":
                    for gchild in child.children:
                        # print(gchild.type)
                        if gchild.type == "identifier":
                            s = gchild.start_point
                            e = gchild.end_point

                            ide = new_code[s[0]][s[-1] : e[-1]]
                            if not ide in ilist:
                                ilist.append(ide)
                        elif gchild.type == "default_parameter":
                            for ggchild in gchild.children:
                                if ggchild.type == "identifier":
                                    s = ggchild.start_point
                                    e = ggchild.end_point

                                    ide = new_code[s[0]][s[-1] : e[-1]]
                                    if not ide in ilist:
                                        ilist.append(ide)

                else:
                    nodes.append(child)

        elif cur_node.type == "assignment":
            for child in cur_node.children:
                if child.type == "identifier":
                    s = child.start_point
                    e = child.end_point

                    ide = new_code[s[0]][s[-1] : e[-1]]
                    if not ide in ilist:
                        ilist.append(ide)

                elif child.type == "pattern_list":
                    for gchild in child.children:
                        if gchild.type == "identifier":
                            s = gchild.start_point
                            e = gchild.end_point

                            ide = new_code[s[0]][s[-1] : e[-1]]
                            if not ide in ilist:
                                ilist.append(ide)

                elif child.type == "=":
                    break

        else:
            for child in cur_node.children:
                nodes.append(child)
    # print(ilist)
    return ilist


def find_java_ide(code_string):
    code_string = "class CLS { \n" + code_string + " \n }"
    # LANGUAGE = Language('build/my-languages.so', "java")D:\Mywork\test_results\build
    LANGUAGE = Language("D:\\Mywork\\test_results\\build\\my-languages.so", "java")
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
        if cur_node.type == "ERROR":
            print(code_string)

        if (
            cur_node.type == "variable_declarator"
            or cur_node.type == "formal_parameter"
            or cur_node.type == "method_declaration"
            or cur_node.type == "constructor_declaration"
            or cur_node.type == "assignment_expression"
        ):
            for child in cur_node.children:
                if child.type == "identifier":
                    s = child.start_point
                    e = child.end_point
                    ide = new_code[s[0]][s[-1] : e[-1]]
                    if not ide in ilist:
                        ilist.append(ide)

        for child in cur_node.children:
            nodes.append(child)

    return ilist


def find_go_ide(code_string):
    LANGUAGE = Language("build/my-languages.so", "go")
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

        if cur_node.type == "short_var_declaration":
            child = cur_node.children[0]
            if child.type == "expression_list":
                for gchild in child.children:
                    if gchild.type == "identifier":
                        s = gchild.start_point
                        e = gchild.end_point
                        ide = new_code[s[0]][s[-1] : e[-1]]
                        if not ide in ilist:
                            ilist.append(ide)

        elif "declaration" in cur_node.type:
            for child in cur_node.children:
                if child.type == "identifier":
                    s = child.start_point
                    e = child.end_point
                    ide = new_code[s[0]][s[-1] : e[-1]]
                    if not ide in ilist:
                        ilist.append(ide)

        for child in cur_node.children:
            nodes.append(child)

    return ilist


# METHOD:替换函数名
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


def switch(lang):
    func_dict = {"python": find_python_ide, "java": find_java_ide, "go": find_go_ide}
    # path = "D:\\Mywork\\data\\CodeT5\\summarize\\" + lang + "\\test.jsonl"
    # path = "D:\\Mywork\\data\\bench\\original\\DC\\data\\test.json"
    # path = "D:\\Mywork\\data\\bench\\structral\\Operand Swap\\" + lang + "\\"  + "CSN" + "\\test.jsonl"
    path = os.path.join(
        "D:\\Mywork\\data\\bench\\original\\CSN\\summarize\\python", "train.jsonl"
    )
    with open(path, "r") as pf:
        data = pf.readlines()

    examples = []

    for d in data:
        line_a = json.loads(d)

        ide_list = func_dict[lang](line_a["original_string"])
        # print(ide_list)

        # random
        rate = 0.8
        num = round(rate * len(ide_list))
        ide_list = random.sample(ide_list, num)
        # print(ide_list)

        idict = switch_ide(ide_list)

        if idict != None:
            for k, d in enumerate(line_a["code_tokens"]):
                if d in idict:
                    line_a["code_tokens"][k] = idict[d]

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

    # file_path = "D:\\Mywork\\data\\bench\\identity\\sub\\highfreq\\" + lang + "\\" + "CSN" + "\\test.jsonl"
    # file_path = "D:\\Mywork\\data\\bench\\cross\\ophfreq\\" + lang + "\\"  + "CSN" + "\\test.jsonl"
    # file_path = os.path.join("D:\\Mywork\\data\\curriculum\\CSN\\IOE\\python", "train.jsonl")

    file_path = os.path.join("D:\\python\\train{}".format(rate), "train.jsonl")
    print(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines("\n".join(examples))


def checkast():
    func_dict = {"python": find_python_ide, "java": find_java_ide, "go": find_go_ide}
    path = "D:\\Mywork\\data\\CodeT5\\summarize\\" + lang + "\\test.jsonl"

    with open(path, "r") as pf:
        data = pf.readlines()

    examples = []

    for d in data:
        line_a = json.loads(d)
        ide_list = func_dict[lang](line_a["original_string"])


def get_sorted_list(d, reverse=False):
    return sorted(d.items(), key=lambda x: x[1], reverse=reverse)


def switch_high_ide(original_ide, ilist):
    new_dict = dict()
    cc = 0
    for ide in original_ide:
        if ilist[cc][0] == "_":
            cc += 1
        tmp = ilist[cc][0]
        cc += 1
        new_dict.update({ide: tmp})
    # print(new_dict)
    return new_dict


def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


def save_totide():
    tmp = 0
    totlist = []
    opath = "D:\\Mywork\\data\\CodeT5\\summarize\\" + lang
    path0 = opath + "\\train.jsonl"
    path1 = opath + "\\valid.jsonl"
    path2 = opath + "\\test.jsonl"
    save_path = opath + "\\ide.txt"

    for path in [path0, path1, path2]:
        print(path)
        with open(path, "r") as pf:
            data = pf.readlines()

        for d in data:
            line_a = json.loads(d)

            ilist = find_python_ide(line_a["original_string"])

            for ide in ilist:
                if not ide in totlist:
                    totlist.append(ide)

            stt = line_a["func_name"].split(".")
            for ide in stt:
                if not ide in totlist:
                    totlist.append(ide)
    f = open(save_path, "w")
    f.write("\n".join(totlist))
    f.close()


def switchDC(lang):
    lang = "java"
    func_dict = {"python": find_python_ide, "java": find_java_ide, "go": find_go_ide}
    # path = "D:\\Mywork\\data\\bench\\original\\DC\\data\\test.json"
    path = (
        "D:\\Mywork\\data\\bench\\structral\\Operand Swap\\"
        + lang
        + "\\"
        + "DC"
        + "\\test.jsonl"
    )
    with open(path, "r") as pf:
        data = pf.readlines()

    examples = []

    for d in data:
        line_a = json.loads(d)

        ide_list = func_dict[lang](line_a["code"])

        idict = high_ide(ide_list)

        if len(ide_list) < 1:
            print(line_a)
            print(ide_list)
            print(idict)
            return

        # line_a["code_tokens"] = code_tokenizer(line_a["code"])
        # line_a["docstring_tokens"] = nltk.word_tokenize(line_a["nl"])

        if idict != None:
            for k, d in enumerate(line_a["code_tokens"]):
                if d in idict:
                    line_a["code_tokens"][k] = idict[d]

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
        # return

    # file_path = "D:\\Mywork\\data\\bench\\identity\\sub\\highfreq\\" + lang + "\\" + "DC" + "\\test.jsonl"
    file_path = (
        "D:\\Mywork\\data\\bench\\cross\\ophfreq\\"
        + lang
        + "\\"
        + "DC"
        + "\\test.jsonl"
    )
    print(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines("\n".join(examples))


# METHOD:替换操作符
def switchTL(lang):
    lang = "java"
    func_dict = {"python": find_python_ide, "java": find_java_ide, "go": find_go_ide}
    # path = "D:\\Mywork\\data\\bench\\original\\TL\\data\\test.json"
    path = (
        "D:\\Mywork\\data\\bench\\structral\\Operand Swap\\"
        + lang
        + "\\"
        + "TL"
        + "\\test.jsonl"
    )
    with open(path, "r") as pf:
        data = pf.readlines()

    examples = []

    for d in data:
        line_a = json.loads(d)

        ide_list = func_dict[lang](line_a["code"])

        idict = high_ide(ide_list)

        if len(ide_list) < 1:
            print(line_a)
            print(ide_list)
            print(idict)
            return

        # line_a["code_tokens"] = code_tokenizer(line_a["code"])
        # line_a["docstring_tokens"] = nltk.word_tokenize(line_a["comment"])

        if idict != None:
            for k, d in enumerate(line_a["code_tokens"]):
                if d in idict:
                    line_a["code_tokens"][k] = idict[d]

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
        # return

    # file_path = "D:\\Mywork\\data\\bench\\identity\\sub\\highfreq\\" + lang + "\\" + "TL" + "\\test.jsonl"
    file_path = (
        "D:\\Mywork\\data\\bench\\cross\\ophfreq\\"
        + lang
        + "\\"
        + "TL"
        + "\\test.jsonl"
    )
    print(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines("\n".join(examples))


# METHOD:找到高频词
def find_high_freq():
    lang = "java"
    tmp = 0
    idict = dict()
    opath = "D:\\Mywork\\data\\CodeT5\\summarize\\" + lang
    path0 = opath + "\\train.jsonl"
    path1 = opath + "\\valid.jsonl"
    path2 = opath + "\\test.jsonl"

    for path in [path0, path1, path2]:
        with open(path, "r") as pf:
            data = pf.readlines()

        for d in data:
            line_a = json.loads(d)

            ilist = find_java_ide(line_a["original_string"])
            if len(ilist) > tmp:
                tmp = len(ilist)
            for ide in ilist:
                if not ide in idict:
                    idict.update({ide: 1})
                else:
                    idict[ide] += 1

    ilist = get_sorted_list(idict, True)

    print(ilist[:200])


def calc_bleu():
    predPath = "D:\\Mywork\\test_results\\CodeBERT\\code2nl\\java\\test_1.output"
    goldPath = "D:\\Mywork\\test_results\\CodeBERT\\code2nl\\java\\test_1.gold"
    with open(predPath, "r") as pf:
        data = pf.readlines()
    predictions = []
    for d in data:
        predictions.append(d)

    (goldMap, predictionMap) = bleu.computeMaps(predictions, goldPath)
    dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    print("  %s = %s " % ("bleu-4", str(dev_bleu)))
    print("  " + "*" * 20)


def find_case():
    goldPath = "D:\\Mywork\\test_results\\CodeT5\\summarize\\go\\prediction\\test_best-bleu.gold"
    predPath = "D:\\Mywork\\test_results\\CodeT5\\summarize\\go\\prediction\\test_best-bleu.output"
    srcPath = "D:\\Mywork\\test_results\\CodeT5\\summarize\\go\\prediction\\test_best-bleu.src"

    goldPath1 = "D:\\Mywork\\data\\results\\identity\\func\\go\\CSN\\CodeT5\\prediction\\test_best-bleu.gold"
    predPath1 = "D:\\Mywork\\data\\results\\identity\\func\\go\\CSN\\CodeT5\\prediction\\test_best-bleu.output"

    with open(goldPath, "r") as pfs:
        data = pfs.readlines()
    srcs = []
    for d in data:
        srcs.append(d)

    with open(goldPath, "r") as pf:
        data = pf.readlines()
    golds = []
    for d in data:
        golds.append(d)

    with open(predPath, "r") as pf1:
        data = pf1.readlines()
    preds = []
    for d in data:
        preds.append(d)

    with open(goldPath1, "r") as pf2:
        data = pf2.readlines()
    golds1 = []
    for d in data:
        golds1.append(d)

    with open(predPath1, "r", encoding="utf-8") as pf3:
        data = pf3.readlines()
    preds1 = []
    for d in data:
        preds1.append(d)

    print(len(golds))
    print(len(golds1))
    print(len(preds))
    print(len(preds1))

    for i in range(len(golds)):
        bleu0 = bleu.bleu([golds[i]], preds[i])[0]
        bleu1 = bleu.bleu([golds1[i]], preds1[i])[0]
        tmp = srcs[i]

        if (bleu0 - bleu1 > 0.3) and (len(tmp.split(" ")) < 15):
            print("{}\t{}\t{}\t{}".format(i, golds[i], preds[i], preds1[i]))

    # (goldMap, predictionMap) = bleu.computeMaps(predPath, goldPath)
    # dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
    # print("  %s = %s "%("bleu-4",str(dev_bleu)))
    # print("  "+"*"*20)


class RobustCodeSum:
    def __init__(self) -> None:
        self.dataset = load_dataset("code_search_net", "python")["test"]
        self.langurage = "python"
        pass

    def gen_FNE(self) -> None:
        pass


if __name__ == "__main__":
    dataset = load_dataset("code_search_net", "python")["test"]
    print(dataset.shape)

    switch("python")

    # t = tuple((1, 2, (3, 3)))

    # print(len(t))

    # goldPath = "D:\\Mywork\\data\\results\\identity\\func\\java\\CSN\\CodeT5\\prediction\\test_best-bleu.gold"
    # predPath = "D:\\Mywork\\data\\results\\identity\\func\\java\\CSN\\CodeT5\\prediction\\test_best-bleu.output"
    # (goldMap, predictionMap) = bleu.computeMaps(predPath, goldPath)
    # dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
    # print("  %s = %s "%("bleu-4",str(dev_bleu)))
    # print("  "+"*"*20)

    # for l in ['python', 'java', 'go']:
    #     switch(l)
    # switchDC('java')
    # switchTL('java')
    # find_case()
    # print("def stop_step(self, step_name):\n        \"\"\" Stop a step. \"\"\"\n        if self.finished is not None:\n            raise AlreadyFinished()\n\n        steps = copy.deepcopy(self.steps)\n\n        step_data = self._get_step(step_name, steps=steps)\n        if step_data is None:\n            raise StepNotStarted()\n        elif 'stop' in step_data:\n            raise StepAlreadyFinished()\n\n        step_data['stop'] = datetime.utcnow()\n\n        step_data['duration'] = util.timedelta_total_seconds(step_data['stop'] - step_data['start'])\n        self._save(steps=steps)")
    # print(bleu.bleu(["Downloads Dailymotion videos by URL ."], "Download a single video ."))
    # print("""
    # def get_pandas_df(self, hql, parameters=None):\n        \"\"\"\n        Get a pandas dataframe from a sql query.\n        \"\"\"\n        import pandas\n        cursor = self.get_cursor()\n        try:\n            cursor.execute(self._strip_sql(hql), parameters)\n            data = cursor.fetchall()\n        except DatabaseError as e:\n            raise PrestoException(self._get_pretty_exception_message(e))\n        column_descriptions = cursor.description\n        if data:\n            df = pandas.DataFrame(data)\n            df.columns = [c[0] for c in column_descriptions]\n        else:\n            df = pandas.DataFrame()\n        return df
    # """)
    # print("""
    # static public byte[] intToBytes(int v) {\r\n        byte[] b       = new byte[4];\r\n        int    allbits = 255;\r\n        for (int i = 0; i < 4; i++) {\r\n            b[3 - i] = (byte) ((v & (allbits << i * 8)) >> i * 8);\r\n        }\r\n        return b;\r\n    }
    # """)
    # print("""
    # func NewResource(name, rtype, state, owner string, t time.Time) Resource {\n\treturn Resource{\n\t\tName:       name,\n\t\tType:       rtype,\n\t\tState:      state,\n\t\tOwner:      owner,\n\t\tLastUpdate: t,\n\t\tUserData:   &UserData{},\n\t}\n}
    # """)
