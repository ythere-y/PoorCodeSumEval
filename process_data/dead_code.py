import spacy

import random

from tree_sitter import Language, Parser

from ttss import find_python_ide, find_java_ide, find_go_ide

import os
import json
import numpy as np
from copy import deepcopy
import nltk

python_dict = [('data', 14119), ('name', 13844), ('value', 9501), ('result', 8697), ('cls', 8038), ('path', 7955), ('key', 6608), \
        ('url', 6548), ('response', 6343), ('args', 6062), ('msg', 4970), ('filename', 4933), ('params', 4810), ('x', 4638), ('request', 4637), ('ret', 4197), \
            ('obj', 3872), ('res', 3413), ('message', 3359), ('f', 3320), ('s', 3285), ('start', 3218), ('r', 3181), ('text', 3137), ('cmd', 2977), \
                ('config', 2940), ('func', 2922), ('n', 2863), ('y', 2797), ('index', 2790), ('context', 2752), ('d', 2697), ('node', 2628), ('p', 2569), ('query', 2436), \
                    ('line', 2432), ('timeout', 2385), ('out', 2332), ('output', 2231), ('user', 2230), ('headers', 2209), ('size', 2115), ('method', 2101), ('i', 2098), \
                        ('kwargs', 2069), ('m', 2056), ('a', 2052), ('values', 1994), ('b', 1951), ('results', 1909), ('val', 1900), ('c', 1874), ('event', 1862), ('model', 1853), \
                            ('content', 1815), ('t', 1808), ('target', 1767), ('lines', 1766), ('options', 1762), ('status', 1739), ('count', 1707), ('parser', 1697), ('source', 1694), \
                                ('end', 1665), ('body', 1660), ('prefix', 1657), ('version', 1647), ('offset', 1630), ('v', 1612), ('match', 1606), ('command', 1602), ('payload', 1597), \
                                    ('item', 1574), ('state', 1559), ('ctx', 1552), ('get', 1543), ('session', 1531), ('client', 1513)]


java_dict = [('i', 14990), ('value', 7816), ('result', 6985), ('name', 6818), ('key', 4851), ('type', 3949), ('response', 3691), ('index', 3359), ('request', 2997), ('context', 2831), ('c', 2798), ('path', 2752), ('sb', 2518), ('obj', 2419), ('id', 2413), ('data', 2339), ('get', 2237), ('s', 2154), ('x', 1976), ('start', 1942), ('node', 1907), ('file', 1906), ('message', 1904), ('url', 1877), ('args', 1871), ('j', 1846), ('e', 1819), ('length', 1792), ('b', 1772), ('n', 1755), ('list', 1727), ('service', 1727), ('builder', 1724), ('client', 1679), ('size', 1635), ('count', 1563), ('clazz', 1553), ('p', 1537), ('offset', 1490), ('out', 1488), ('t', 1481), ('ret', 1476), ('v', 1450), ('buffer', 1417), ('y', 1380), ('run', 1371), ('input', 1362), ('msg', 1360), ('resource', 1315), ('len', 1314), ('m', 1281), ('source', 1270), ('values', 1253), ('text', 1248), ('map', 1244), ('params', 1234), ('in', 1231), ('method', 1227), ('config', 1223), ('add', 1195), ('event', 1190), ('o', 1179), ('target', 1170), ('parent', 1166), ('field', 1152), ('buf', 1150), ('entry', 1143), ('a', 1132), ('r', 1121), ('pos', 1071), ('object', 1065), ('state', 1013), ('query', 1005), ('end', 998), ('uri', 972), ('f', 966), ('element', 952), ('resp', 950), ('it', 943), ('session', 920), ('str', 914), ('properties', 912), ('line', 907), ('reader', 897), ('val', 892), ('filter', 875), ('qPath', 874), ('bytes', 860), ('root', 846), ('listener', 838), ('k', 827), ('writer', 810), ('next', 805), ('d', 801), ('output', 797), ('create', 787), ('ctx', 783), ('item', 783), ('set', 774), ('className', 772), ('options', 757), ('row', 743), ('call', 736), ('res', 735), ('methodName', 720), ('is', 719), ('update', 713), ('token', 712), ('array', 708), ('parameters', 705), ('prefix', 702), ('width', 694), ('status', 692), ('iterator', 691), ('serviceName', 682), ('position', 676), ('defaultValue', 675), ('connection', 672), ('stream', 670), ('info', 662), ('idx', 655), ('iter', 654), ('model', 648), ('props', 645), ('l', 632), ('version', 629), ('table', 628), ('first', 623), ('content', 618), ('close', 616), ('entity', 616), ('init', 612), ('req', 612), ('from', 610), ('format', 609), ('remove', 604), ('max', 601), ('execute', 599), ('task', 596), ('property', 579), ('factory', 576), ('results', 575), ('handler', 570), ('error', 568), ('fileName', 566), ('user', 566), ('parse', 564), ('instance', 553), ('tmp', 547), ('pattern', 541), ('child', 541), ('matcher', 537), ('height', 535), ('date', 535), ('delete', 535), ('action', 532), ('attributes', 528), ('src', 528), ('write', 527), ('read', 524), ('current', 518), ('password', 517), ('conf', 513), ('port', 513), ('resources', 512), ('timeout', 501), ('json', 493), ('ch', 493), ('other', 490), ('fieldName', 489), ('tag', 484), ('w', 482), ('dir', 482), ('string', 478), ('keys', 475), ('propertyName', 474), ('g', 472), ('scope', 472), ('doc', 470), ('conn', 470), ('process', 469), ('fields', 467), ('to', 466), ('address', 466), 
('description', 463), ('attribute', 462), ('filename', 457), ('code', 456), ('record', 453), ('column', 453), ('body', 453), ('apply', 452), ('parser', 447), ('formatter', 446), ('headers', 445), ('put', 445), ('component', 443), ('callback', 436), ('h', 435), ('ref', 434)]

go_dict = [('err', 63652), ('s', 26468), ('c', 21460), ('ctx', 14832), ('ok', 13520), ('v', 13152), ('r', 12447), ('p', 9861), \
    ('m', 9121), ('name', 7999), ('b', 7746), ('i', 7560), ('t', 7202), ('f', 6121), ('w', 5897), ('key', 5756), ('out', 5263), ('n', 5217), \
        ('d', 5026), ('a', 4895), ('e', 4588), ('in', 4553), ('data', 4274), ('id', 4269), ('args', 4249), ('l', 4130), ('resp', 4041), ('o', 3986), \
            ('req', 3886), ('h', 3838), ('path', 3834), ('value', 3402), ('client', 3046), ('result', 3008), ('opts', 2817), ('g', 2462), ('u', 2325), \
                ('config', 2310), ('ret', 2287), ('res', 2262), ('val', 2131), ('buf', 2118), ('db', 2104), ('k', 1969), ('msg', 1925), ('cmd', 1858), \
                    ('params', 1777), ('conn', 1719), ('options', 1706), ('q', 1696), ('x', 1629), ('info', 1570), ('obj', 1499), ('tx', 1492), ('cfg', 1430), \
                        ('addr', 1424), ('New', 1415), ('j', 1404), ('body', 1372), ('url', 1353), ('index', 1283), ('fn', 1217), ('logger', 1179), ('node', 1120), \
                            ('size', 1111), ('file', 1106), ('format', 1060), ('src', 1052), ('api', 1046), ('response', 1041), ('found', 1038), ('token', 1026), \
                                ('timeout', 1010), ('request', 1002), ('ret0', 994), ('cancel', 990), ('results', 970), ('query', 966), ('state', 953)]

# lang = 'java'
tot = 0

def get_str(new_code, node):
    s = node.start_point
    e = node.end_point
    ide = new_code[s[0]][s[-1]:e[-1]]
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
        new_code[l] = code[l][:s[-1]] + stmp + code[l][e[-1]:]
    else:
        new_code[l] = new_code[l][:s[-1]] + stmp
        for i in range(s[0] + 1, e[0]): new_code[i] = ""
        # print(code)
        tmp = get_tab(code, e[0])
        new_code[e[0]] = tmp + new_code[e[0]][e[-1]:]
    return new_code


def code_inject(code_string, lang): 

    ide_list = []
    ii = []
    tt = 3

    if lang == "python":
        ide_list = find_python_ide(code_string)
        for i in range(len(python_dict)):
            if not python_dict[i][0] in ide_list:
                ii.append(python_dict[i][0])
                tt -= 1
            if tt == 0: break

    elif lang == "java":
        ide_list = find_java_ide(code_string)
        for i in range(len(java_dict)):
            if not java_dict[i][0] in ide_list:
                ii.append(java_dict[i][0])
                tt -= 1
            if tt == 0: break

    elif lang == "go":
        ide_list = find_go_ide(code_string)
        for i in range(len(go_dict)):
            if not go_dict[i][0] in ide_list:
                ii.append(go_dict[i][0])
                tt -= 1
            if tt == 0: break

    # print(ii)
    if len(ii) == 0:
        print("?????????????????")
        # print(code_string)


    if lang == 'java':        
        code_string = "class CLS { \n" + code_string +  " \n }"

    LANGUAGE = Language('build/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes(code_string, 'utf-8'))
    root_node = tree.root_node
    nodes = [root_node]

    ilist = []
    new_code = code_string.split('\n')
    # print(new_code)

    body_list = []

    while len(nodes) > 0:
        cur_node = nodes[0]
        nodes = nodes[1:]
        # if cur_node.type == "ERROR":
        #     print(code_string)

        s = []
        e = []
        

        if lang == "java":
            if cur_node.type == "method_declaration" or cur_node.type == "constructor_declaration":
                for child in cur_node.children:
                    if child.type == "block":
                        for gchild in child.children:
                            s.append(gchild.start_point)
                            e.append(gchild.end_point)


                if len(s) > 2:

                    pos = random.randrange(1, len(s) - 1, 1)
                    sl = e[pos][0]

                    stmp = ""
                    for i in ii:
                        mtmp = random.randrange(0, 100, 1)
                        stmp = stmp + "\n" + "int " + i + "=" + str(mtmp) + ";"

                    new_code[sl] = new_code[sl] + "\n" + stmp


        if lang == "go":
            if cur_node.type == "function_declaration" or cur_node.type == "method_declaration":
                for child in cur_node.children:
                    if child.type == "block":

                        for gchild in child.children:
                            s.append(gchild.start_point)
                            e.append(gchild.end_point)

                if len(s) > 2:
                    pos = random.randrange(1, len(s) - 1, 1)
                    sl = e[pos][0]

                    stmp = ""
                    for i in ii:
                        mtmp = random.randrange(0, 100, 1)
                        stmp = stmp + "\n" + i + ":=" + str(mtmp)

                    new_code[sl] = new_code[sl] + "\n" + stmp
        
        if lang == "python":
            
            if cur_node.type == "function_definition":

                for child in cur_node.children:
                    if child.type == "block":
                        for gchild in child.children:
                            s.append(gchild.start_point)
                            e.append(gchild.end_point)
                
                pos = random.randrange(0, len(s), 1)
                sl = e[pos][0]

                ntab = get_tab(new_code, s[pos][0])

                stmp = ""
                for i in ii:
                    mtmp = random.randrange(0, 100, 1)
                    stmp = stmp + "\n" + ntab + i + "=" + str(mtmp)

                new_code[sl] = new_code[sl] + "\n" + ntab + stmp


        
        for child in cur_node.children: nodes.append(child)

    
    # print(body_list)
    # if body_list != []: print(1)



    if lang == "java": new_code = new_code[1:len(new_code)-1]
    # print("\n".join(new_code))

    return "\n".join(new_code)


def code_tokenizer(code_string, lang):
    code_tokens = []


    def calc_ident(node):
        node_childs = node.children
        if (lang == "python" and node.type == "expression_statement"):
            if node.children[0].type == "string":
                return
        if (lang == "python" and node.type == "comment"): return
        if (lang == "go" and node.type == "comment"): return

        if not node.children or node.type == "interpreted_string_literal" or node.type == "string": 
            s = node.start_point
            e = node.end_point
            tmp = new_code[s[0]][s[-1]:e[-1]]
            if (tmp != '') and (tmp != ' '): code_tokens.append(tmp)
            return
        for i in range(len(node_childs)):
            calc_ident(node_childs[i])

    LANGUAGE = Language('build/my-languages.so', lang)
    if lang == 'java':        
        code_string = "class CLS { \n" + code_string +  " \n }"
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes(code_string, "utf8"))
    
    new_code = code_string.split('\n')
    
    calc_ident(tree.root_node)
    # print(len(identi_list))
    if lang == "java":
        code_tokens = code_tokens[3:len(code_tokens)-1]

    return code_tokens

def find_ttt(code_string, lang): 
    LANGUAGE = Language('build/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes(code_string, 'utf-8'))
    root_node = tree.root_node
    nodes = [root_node]
    ilist = []
    new_code = code_string.split('\n')

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
                ide = new_code[s[0]][s[-1]:e[-1]]
                print("###{}###".format(ide), end= " ")

            nodes.append(child)

        print("\n")
        
    
    return ilist
            
def handle():

    for lang in ["go"]:
        path = "D:\\Mywork\\data\\CodeT5\\summarize\\" + lang + "\\test.jsonl"

        with open(path, 'r') as pf:
            data = pf.readlines() 

        examples = [] 
        

        for d in data:
            
            line_a = json.loads(d)
            
            tmp = code_tokenizer(code_inject( line_a["original_string"], lang ), lang )
            
            line_a["code_tokens"] = tmp
        
            examples.append(json.dumps(line_a))
        
        file_path = "D:\\Mywork\\data\\bench\\structral\\Deadcode\\" + lang + "\\"  + "CSN" + "\\test.jsonl"
        print(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(examples))


def handleDC():
    lang = "java"
    path = "D:\\Mywork\\data\\bench\\original\\DC\\data\\test.json"
    with open(path, 'r') as pf:
        data = pf.readlines() 

    examples = [] 
    

    for d in data:
        
        line_a = json.loads(d)
        
        # line_a["code_tokens"] = code_tokenizer(line_a["code"])
        line_a["docstring_tokens"] = nltk.word_tokenize(line_a["nl"])
        # print(line_a["code"])
        tmp = code_tokenizer(code_inject( line_a["code"], lang ), lang )
        
        line_a["code_tokens"] = tmp
     
        examples.append(json.dumps(line_a))
    
    file_path = "D:\\Mywork\\data\\bench\\structral\\Deadcode\\" + lang + "\\"  + "DC" + "\\test.jsonl"
    print(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(examples))


def handleTL():
    lang = "java"
    path = "D:\\Mywork\\data\\bench\\original\\TL\\data\\test.json"
    with open(path, 'r') as pf:
        data = pf.readlines() 

    examples = [] 
    

    for d in data:
        
        line_a = json.loads(d)
        
        # line_a["code_tokens"] = code_tokenizer(line_a["code"])
        line_a["docstring_tokens"] = nltk.word_tokenize(line_a["comment"])
        # print(line_a["code"])
        tmp = code_tokenizer(code_inject( line_a["code"], lang ), lang )
        
        line_a["code_tokens"] = tmp
     
        examples.append(json.dumps(line_a))
    
    file_path = "D:\\Mywork\\data\\bench\\structral\\Deadcode\\" + lang + "\\"  + "TL" + "\\test.jsonl"
    print(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(examples))

                
if __name__ == "__main__":

    # handle()
    handleTL()
    handleDC()

    # tmp = """

    # """

    # tmp = "class CLS { \n" + tmp + "\n }" 

    # find_ttt(tmp, "java")
    # print(code_inject(tmp, "java"))
    # print(find_java_ide(tmp))