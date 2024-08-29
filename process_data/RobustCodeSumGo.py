import random
import os
from h11 import Data
import pandas as pd
import parso
import time
from typing import List, Dict, Any
from tree_sitter import Language, Parser, Node, Tree
from datasets import load_dataset, Dataset, load_from_disk
from colorama import Fore, Back, Style
from tqdm import tqdm
from statistics import go_dict

from process_data.utils import ParserUtils, DataUtils

# 设置随机种子
random.seed(0)


def match_from_span(node, blob: str) -> str:
    """
    Get real code string of node
    """
    lines = blob.split("\n")
    line_start = node.start_point[0]
    line_end = node.end_point[0]
    char_start = node.start_point[1]
    char_end = node.end_point[1]
    if line_start != line_end:
        return "\n".join(
            [lines[line_start][char_start:]]
            + lines[line_start + 1 : line_end]
            + [lines[line_end][:char_end]]
        )
    else:
        return lines[line_start][char_start:char_end]


def dfs_method(node, docstring_all: List[str], blob: str):
    if node is None:
        return
    if node.type == "comment":
        docstring_all.append(match_from_span(node), blob).strip()
        return
    elif node.children is not None:
        for n in node.children:
            dfs_method(n, docstring_all, blob)


class GoParser:
    @staticmethod
    def wrap_java_code(java_code):
        return f"class CLS {{\n{java_code}\n}}"

    @staticmethod
    def unwrap_java_code(java_code):
        # 去掉上面这个函数加入的前缀和后缀,包括\n也要去掉
        return java_code[len("class CLS {\n") : -len("\n}")]

    @staticmethod
    def extract_method_body(tree: Tree, blob: str) -> str:
        method_head = tree.root_node.children[0].child_by_field_name("body").children[1]
        body_node = method_head.child_by_field_name("body")
        # body_block = body_block[1:-1].lstrip("{\n\t").rstrip("}\n\t")
        return match_from_span(body_node, blob).lstrip("{\n\t").rstrip("}\n\t")

    @staticmethod
    def get_method_body_node(root_node: Node) -> Node:
        try:
            method_body_node = root_node.children[0].child_by_field_name("body")
            if method_body_node == None:
                print("error")
                raise Exception("get_method_body_node error")
        except Exception as e:
            raise e
        return method_body_node

    @staticmethod
    def get_method_identifier_node(root_node: Node) -> Node:
        identifier_node = root_node.children[0].child_by_field_name("name")
        return identifier_node

    @staticmethod
    def get_method_def_node(tree: Tree) -> Node:
        return tree.root_node.children[0].child_by_field_name("body").children[1]

    @staticmethod
    def get_all_identifiers(tree: Tree) -> List[Node]:
        pass

    @staticmethod
    def get_all_target_ide_name(root_node: Node) -> List[str]:
        nodes = [root_node]

        target_ide_name_set = set()
        while len(nodes) > 0:
            cur_node = nodes.pop(0)

            if cur_node.type == "short_var_declaration":
                child = cur_node.child_by_field_name("left")
                if child.type == "expression_list":
                    for gchild in child.children:
                        if gchild.type == "identifier":
                            identifier_string = gchild.text.decode("utf8")
                            target_ide_name_set.add(identifier_string)

            elif "declaration" in cur_node.type:
                name_node = cur_node.child_by_field_name("name")
                if name_node:
                    identifier_string = name_node.text.decode("utf8")
                    target_ide_name_set.add(identifier_string)

            for child in cur_node.children:
                nodes.append(child)

        return list(target_ide_name_set)

    TARGET_OPERATIONS = {
        "<": ">",
        "<=": ">=",
        ">": "<",
        ">=": "<=",
        "==": "==",
        "*": "*",
        "+": "+",
        "!=": "!=",
        "&&": "&&",
        "||": "||",
        "&": "&",
        "|": "|",
        "^": "^",
    }

    @staticmethod
    def oper_swap(root_node: Node, blob: str):
        binary_expressions_list = []
        ParserUtils.shallow_traverse_type(
            root_node, binary_expressions_list, "binary_expression"
        )
        nodes_info = []
        # print(len(binary_expressions_list))
        for _binary in binary_expressions_list:
            # 检查是否可交换
            work_flag = True
            for child in _binary.children:
                # 操作符的type就是自身字符串，所以可以用长度来筛查
                if len(child.type) <= 4:
                    if child.type not in GoParser.TARGET_OPERATIONS:
                        work_flag = False
                        break
            if work_flag:
                # 可以进行swap工作
                new_binary_exp = ""
                for child in _binary.children:
                    if len(child.type) > 4:
                        # 是操作数，不是操作符
                        new_binary_exp = f'{child.text.decode("utf8")} {new_binary_exp}'
                    else:
                        # 是操作符，需要考虑翻转
                        new_op = GoParser.TARGET_OPERATIONS[child.type]
                        new_binary_exp = f"{new_op} {new_binary_exp}"
                # 记录到替换队列中
                nodes_info.append((_binary, new_binary_exp))
        if len(nodes_info) > 0:
            swap_happened = True
        else:
            swap_happened = False
        new_blob = ParserUtils.replace_nodes_str(nodes_info, blob)
        return new_blob, swap_happened


class GoRobustCodeSum:
    def __init__(self) -> None:
        part_name = "CSN"

        if part_name == "CSN":
            DATASET_PATH = (
                "/data2/huchao/11.novels/download/code_x_glue_ct_code_to_text"
            )
            self.dataset = load_dataset(DATASET_PATH, "go")["test"]
            print(f"load data from codex blue, size: {self.dataset.shape}")

        self.lang_str = "go"
        self.langurage = Language("process_data/build/my-languages.so", self.lang_str)
        self.parser = Parser()
        self.parser.set_language(self.langurage)

    def gen_FNE(self, dataset_part_name="CSN", dataset=None, save_mode=True):
        """
        找到函数identifier
        替换为v0
        """
        if dataset == None:
            dataset = self.dataset
        new_data = []
        mode_name = "FNE"
        type_name = "semantic"
        for i in tqdm(range(0, len(dataset)), desc=f"gen {mode_name}"):
            row = dataset[i]
            code_string = row["code"]
            tree = self.parser.parse(bytes(code_string, "utf8"))
            root_node = tree.root_node
            identifier_node = GoParser.get_method_identifier_node(root_node)

            new_code_string = ParserUtils.replace_node_str(
                identifier_node, code_string, "v0"
            )

            # 展示结果
            # DataUtils.display_before_and_after(code_string, new_code_string)
            # 将处理后的数据添加到新的列表中
            new_row = row.copy()  # 复制当前行的数据
            new_row["code"] = new_code_string  # 更新"code"字段
            new_data.append(new_row)  # 添加到新的数据列表中
        if save_mode:
            DataUtils.save_new_data(
                new_data,
                self.lang_str,
                type_name,
                mode_name,
                dataset_part=dataset_part_name,
            )
        return new_data

    def gen_IOE(self, dataset_part_name="CSN", dataset=None, save_mode=True):
        """
        IOE：标识符变成v0，v1…，函数名也会变
        # 找到所有的identifier_list
        # 将其中的80%作为替换目标
        # 替换identifier_list，每个ide给予一个对象(v_n)
        # 在code中替换对象，作为混淆结果
        """
        if dataset == None:
            dataset = self.dataset
        new_data = []
        mode_name = "IOE"
        for i in tqdm(range(0, len(dataset)), desc=f"gen {mode_name}"):
            row = dataset[i]
            code_string = row["code"]

            tree = self.parser.parse(bytes(code_string, "utf8"))
            root_node = tree.root_node

            # 使用traverse方法来获取所有的identifier
            all_ide_list = []
            ParserUtils.traverse_type(root_node, all_ide_list, "identifier")
            ParserUtils.traverse_type(root_node, all_ide_list, "field_identifier")

            # 继承原来的方法来获取所有对象identifier（主要是要筛去api调用之类的）
            target_ide_name_list = GoParser.get_all_target_ide_name(root_node)

            # 两者结合得到所有的name和node对应关系
            identifier_dict = {}
            for _ide in all_ide_list:
                _name = _ide.text.decode("utf8")
                if _name in target_ide_name_list:
                    if not _name in identifier_dict:
                        identifier_dict[_name] = [_ide]
                    else:
                        identifier_dict[_name].append(_ide)

            # 生成替换字典
            replace_dict = {}
            for i, ide_name in enumerate(target_ide_name_list):
                replace_dict[ide_name] = f"v{i}"

            # 替换identifier
            nodes_info = []
            for identifier, replace_str in replace_dict.items():
                nodes = identifier_dict[identifier]
                for node in nodes:
                    nodes_info.append((node, replace_str))
            new_code_string = ParserUtils.replace_nodes_str(nodes_info, code_string)

            # 展示数据
            # DataUtils.display_before_and_after(code_string, new_code_string)
            # 将处理后的数据添加到新的列表中
            new_row = row.copy()  # 复制当前行的数据
            new_row["code"] = new_code_string  # 更新"code"字段
            new_data.append(new_row)  # 添加到新的数据列表中

            # break
        # return
        if save_mode:
            DataUtils.save_new_data(
                new_data,
                self.lang_str,
                "semantic",
                mode_name,
                dataset_part=dataset_part_name,
            )
        return new_data

    def gen_IHR(self, dataset_part_name="CSN", dataset=None, save_mode=True):
        """
        IHR：使用代码高频词替换标识符
        # 找到所有的identifier_list
        # 对于每一个目标ide,去高频词库中随机找一个词出来替换(ide,random_word)
        同时不同ide的替换词不能相同
        # 在code中替换对象，作为混淆结果
        """
        if dataset == None:
            dataset = self.dataset
        new_data = []
        mode_name = "IHR"
        for i in tqdm(range(0, len(dataset)), desc=f"gen {mode_name}"):
            row = dataset[i]
            code_string = row["code"]

            tree = self.parser.parse(bytes(code_string, "utf8"))
            root_node = tree.root_node

            # 使用traverse方法来获取所有的identifier
            all_ide_list = []
            ParserUtils.traverse_type(root_node, all_ide_list, "identifier")
            ParserUtils.traverse_type(root_node, all_ide_list, "field_identifier")

            # 继承原来的方法来获取所有对象identifier（主要是要筛去api调用之类的）
            target_ide_name_list = GoParser.get_all_target_ide_name(root_node)

            # 两者结合得到所有的name和node对应关系
            identifier_dict = {}
            for _ide in all_ide_list:
                _name = _ide.text.decode("utf8")
                if _name in target_ide_name_list:
                    if not _name in identifier_dict:
                        identifier_dict[_name] = [_ide]
                    else:
                        identifier_dict[_name].append(_ide)

            # 获取高频词数组
            high_freq_list = [word[0] for word in go_dict]

            # 对所有identifier随机匹配一个高频词，且不能有重复的
            # 做法是，先shuffle high_freq_dict，然后按顺序匹配

            # shuffle
            random.shuffle(high_freq_list)

            # 使用匹配的方式，生成替换字典
            replace_dict = {}
            for identifier in target_ide_name_list:
                replace_dict[identifier] = high_freq_list.pop()

            # 替换identifier
            nodes_info = []
            for identifier, replace_str in replace_dict.items():
                nodes = identifier_dict[identifier]
                for node in nodes:
                    nodes_info.append((node, replace_str))

            new_code_string = ParserUtils.replace_nodes_str(nodes_info, code_string)
            # 展示数据
            # DataUtils.display_before_and_after(code_string, new_code_string)
            # 将处理后的数据添加到新的列表中
            new_row = row.copy()  # 复制当前行的数据
            new_row["code"] = new_code_string  # 更新"code"字段
            new_data.append(new_row)  # 添加到新的数据列表中

            # break
        if save_mode:
            DataUtils.save_new_data(
                new_data,
                self.lang_str,
                "semantic",
                mode_name,
                dataset_part=dataset_part_name,
            )
        return new_data

    def gen_IRS(self, dataset_part_name="CSN", dataset=None, save_mode=True):
        """
        IRS: 随机调换标识符名字（使用函数片段中已有的标识符名字）
        # 找到所有的identifier_list
        # 打乱ide_list的顺序，然后形成对应，确定替换关系(ide -> ide')
        # 在code中替换对象，作为混淆结果
        """
        if dataset == None:
            dataset = self.dataset
        new_data = []
        type_name = "semantic"
        mode_name = "IRS"
        for i in tqdm(range(0, len(dataset)), desc=f"gen {mode_name}"):
            row = dataset[i]
            code_string = row["code"]

            tree = self.parser.parse(bytes(code_string, "utf8"))
            root_node = tree.root_node

            # 使用traverse方法来获取所有的identifier
            all_ide_list = []
            ParserUtils.traverse_type(root_node, all_ide_list, "identifier")
            ParserUtils.traverse_type(root_node, all_ide_list, "field_identifier")

            # 继承原来的方法来获取所有对象identifier（主要是要筛去api调用之类的）
            target_ide_name_list = GoParser.get_all_target_ide_name(root_node)

            # 两者结合得到所有的name和node对应关系
            identifier_dict = {}
            for _ide in all_ide_list:
                _name = _ide.text.decode("utf8")
                if _name in target_ide_name_list:
                    if not _name in identifier_dict:
                        identifier_dict[_name] = [_ide]
                    else:
                        identifier_dict[_name].append(_ide)

            # 生成替换字典
            # 匹配,使用zip，使用循环替换的方法来做到打乱效果
            replace_dict = dict(
                zip(target_ide_name_list[:-1], target_ide_name_list[1:])
            )
            replace_dict.update({target_ide_name_list[-1]: target_ide_name_list[0]})

            # 替换identifier
            nodes_info = []
            for identifier, replace_str in replace_dict.items():
                nodes = identifier_dict[identifier]
                for node in nodes:
                    nodes_info.append((node, replace_str))
            new_code_string = ParserUtils.replace_nodes_str(nodes_info, code_string)

            # 展示数据
            # DataUtils.display_before_and_after(code_string, new_code_string)
            # 将处理后的数据添加到新的列表中
            new_row = row.copy()  # 复制当前行的数据
            new_row["code"] = new_code_string  # 更新"code"字段
            new_data.append(new_row)  # 添加到新的数据列表中

            # break
        if save_mode:
            DataUtils.save_new_data(
                new_data,
                self.lang_str,
                type_name,
                mode_name,
                dataset_part=dataset_part_name,
            )
        return new_data

    def gen_OOS(self, dataset_part_name="CSN", dataset=None, save_mode=True):
        """
        OOS：计算交换
        """
        if dataset == None:
            dataset = self.dataset
        new_data = []
        type_name = "synatic"
        mode_name = "OOS"
        for i in tqdm(range(0, len(dataset)), desc=f"gen {mode_name}"):
            row = dataset[i]
            code_string = row["code"]
            tree = self.parser.parse(bytes(code_string, "utf8"))
            root_node = tree.root_node

            new_blob, cur_swap_happened = GoParser.oper_swap(root_node, code_string)

            new_code_string = new_blob

            # 展示数据
            # DataUtils.display_before_and_after(code_string, new_code_string)
            # 将处理后的数据添加到新的列表中
            new_row = row.copy()  # 复制当前行的数据
            new_row["code"] = new_code_string  # 更新"code"字段
            new_row["swap_happened"] = cur_swap_happened
            new_data.append(new_row)  # 添加到新的数据列表中
            # break
        # return None
        if save_mode:
            DataUtils.save_new_data(
                new_data,
                self.lang_str,
                type_name,
                mode_name,
                dataset_part=dataset_part_name,
            )

        return new_data

    def gen_DBI(self, dataset_part_name="CSN", dataset=None, save_mode=True):
        """
        DBI：插入死分支
        """
        if dataset == None:
            dataset = self.dataset
        new_data = []
        # 遍历dataset中的每一行
        type_name = "synatic"
        mode_name = "DBI"
        for i in tqdm(range(0, len(dataset)), desc=f"gen {mode_name}"):
            row = dataset[i]
            code_string = row["code"]
            tree = self.parser.parse(bytes(code_string, "utf8"))
            root_node = tree.root_node

            func_body_node = GoParser.get_method_body_node(root_node)

            # 获取正文退格
            insert_indent_count = 4
            # 获取插入体内容
            insert_body = DataUtils.get_dead_if_body(
                self.lang_str, " " * insert_indent_count
            )

            # 开始插入，先给原body做缩进
            code_lines = code_string.split("\n")
            ## 确定起始位置
            move_start_line = func_body_node.start_point[0] + 1
            move_end_line = func_body_node.end_point[0] + 1
            for i in range(move_start_line, move_end_line):
                code_lines[i] = "    " + code_lines[i]

            random_case = random.randint(0, 1)
            if random_case == 0:
                # 插入 if true & else
                if_stmt = f"{' '*insert_indent_count}if true {{"
                code_lines.insert(move_start_line, if_stmt)
                else_stmt = f'{" " *insert_indent_count}else {{{insert_body}\n{" "*insert_indent_count}}}\n}}'
                code_lines.insert(move_end_line + 1, else_stmt)
            else:
                ## 插入 if false & else
                front_stmt = f"{' '*insert_indent_count}if (false) {{{insert_body}\n{' '*insert_indent_count}}} else {{"
                code_lines.insert(move_start_line, front_stmt)
                code_lines.insert(move_end_line + 2, "}")

            new_code_string = "\n".join(code_lines)
            # 展示数据
            # DataUtils.display_before_and_after(code_string, new_code_string)
            # 将处理后的数据添加到新的列表中
            new_row = row.copy()  # 复制当前行的数据
            new_row["code"] = new_code_string  # 更新"code"字段
            new_data.append(new_row)  # 添加到新的数据列表中

            # break
        if save_mode:
            DataUtils.save_new_data(
                new_data,
                self.lang_str,
                type_name,
                mode_name,
                dataset_part=dataset_part_name,
            )
        return new_data

    def gen_HVI(self, dataset_part_name="CSN", dataset=None, save_mode=True):
        """
        HVI：加入无意义的高频次的变量声明
        """
        if dataset == None:
            dataset = self.dataset
        new_data = []
        type_name = "synatic"
        mode_name = "HVI"
        for i in tqdm(range(0, len(dataset)), desc=f"gen {mode_name}"):
            row = dataset[i]
            code_string = row["code"]
            tree = self.parser.parse(bytes(code_string, "utf8"))
            root_node = tree.root_node

            # 继承原来的方法来获取所有对象identifier（主要是要筛去api调用之类的）
            target_ide_name_list = GoParser.get_all_target_ide_name(root_node)

            # 确定即将插入的标识符名字
            insert_list = []
            insert_num = 3
            for i in range(len(go_dict)):
                if not go_dict[i][0] in target_ide_name_list:
                    insert_list.append(go_dict[i][0])
                    if len(insert_list) == insert_num:
                        break

            # 获取函数体的node
            func_body_node = GoParser.get_method_body_node(root_node)
            # 确定插入位置
            candidates_list = []
            for node in func_body_node.children:
                if len(node.type) > 2:
                    candidates_list.append(node)
            insert_node = random.choice(candidates_list)
            # 获取这个位置的缩进
            insert_indent_count = ParserUtils.get_node_indent_count(
                insert_node, code_string
            )
            # print(f"insert indent = {insert_indent_count}")

            # 制作插入体
            insert_str = ""
            for var in insert_list:
                insert_str += f"{' '*insert_indent_count}var {var} int = {random.randint(0,100)}\n"
            # 最后的位置去掉换行
            insert_str = insert_str[:-1]

            # 获取这个位置的行号
            insert_position = insert_node.start_point[0]
            # 进行插入
            code_lines = code_string.split("\n")
            code_lines.insert(insert_position, insert_str)
            # 插入结果重建
            new_code_string = "\n".join(code_lines)

            # 展示数据
            # DataUtils.display_before_and_after(code_string, new_code_string)
            # 将处理后的数据添加到新的列表中
            new_row = row.copy()  # 复制当前行的数据
            new_row["code"] = new_code_string  # 更新"code"字段
            new_data.append(new_row)  # 添加到新的数据列表中

            # break
        if save_mode:
            DataUtils.save_new_data(
                new_data,
                self.lang_str,
                type_name,
                mode_name,
                dataset_part=dataset_part_name,
            )
        return new_data

    def gen_origin(self, dataset_part_name="CSN", dataset=None, save_mode=True):
        if dataset == None:
            dataset = self.dataset
        new_data = []
        mode_name = "origin"
        for i in tqdm(range(0, len(dataset)), desc=f"gen {mode_name}"):
            row = dataset[i]
            code_string = row["code"]
            try:
                tree = self.parser.parse(bytes(code_string, "utf8"))
            except:
                pass
            new_row = row.copy()  # 复制当前行的数据
            new_data.append(new_row)  # 添加到新的数据列表中

            # break
        if save_mode:
            DataUtils.save_new_data(
                new_data,
                self.lang_str,
                "origin",
                mode_name,
                dataset_part=dataset_part_name,
            )
        return new_data

    def batch_gen(self):
        start_time = time.time()
        self.gen_FNE()
        self.gen_DBI()
        self.gen_HVI()
        self.gen_IOE()
        self.gen_IHR()
        self.gen_IRS()
        self.gen_OOS()
        time_usage = time.time() - start_time
        print(f"all work finished~!, time usage = [{time_usage}]s")

    def set_dataset(self, dataset):
        self.dataset = dataset

    def cross_base_FNE(self):
        FNE_dataset = self.gen_FNE(save_mode=False)

        FNE_DBI_dataset = self.gen_DBI(dataset=FNE_dataset, save_mode=False)
        DataUtils.save_new_data(
            FNE_DBI_dataset, self.lang_str, "FNE", "DBI", partition="cross"
        )

        FNE_OOS_dataset = self.gen_OOS(dataset=FNE_dataset, save_mode=False)
        DataUtils.save_new_data(
            FNE_OOS_dataset, self.lang_str, "FNE", "OOS", partition="cross"
        )

        FNE_HVI_dataset = self.gen_HVI(dataset=FNE_dataset, save_mode=False)
        DataUtils.save_new_data(
            FNE_HVI_dataset, self.lang_str, "FNE", "HVI", partition="cross"
        )

        print(f"********************************")
        print(f"FNE_based_dataset gen finished~!")
        print(f"********************************")

    def cross_base_IOE(self):
        IOE_dataset = self.gen_IOE(save_mode=False)

        IOE_DBI_dataset = self.gen_DBI(dataset=IOE_dataset, save_mode=False)
        DataUtils.save_new_data(
            IOE_DBI_dataset, self.lang_str, "IOE", "DBI", partition="cross"
        )

        IOE_OOS_dataset = self.gen_OOS(dataset=IOE_dataset, save_mode=False)
        DataUtils.save_new_data(
            IOE_OOS_dataset, self.lang_str, "IOE", "OOS", partition="cross"
        )

        IOE_HVI_dataset = self.gen_HVI(dataset=IOE_dataset, save_mode=False)
        DataUtils.save_new_data(
            IOE_HVI_dataset, self.lang_str, "IOE", "HVI", partition="cross"
        )

        print(f"********************************")
        print(f"IOE_based_dataset gen finished~!")
        print(f"********************************")

    def cross_base_IHR(self):
        IHR_dataset = self.gen_IHR(save_mode=False)

        IHR_DBI_dataset = self.gen_DBI(dataset=IHR_dataset, save_mode=False)
        DataUtils.save_new_data(
            IHR_DBI_dataset, self.lang_str, "IHR", "DBI", partition="cross"
        )

        IHR_OOS_dataset = self.gen_OOS(dataset=IHR_dataset, save_mode=False)
        DataUtils.save_new_data(
            IHR_OOS_dataset, self.lang_str, "IHR", "OOS", partition="cross"
        )

        IHR_HVI_dataset = self.gen_HVI(dataset=IHR_dataset, save_mode=False)
        DataUtils.save_new_data(
            IHR_HVI_dataset, self.lang_str, "IHR", "HVI", partition="cross"
        )

        print(f"********************************")
        print(f"IHR_based_dataset gen finished~!")
        print(f"********************************")

    def cross_base_IRS(self):
        IRS_dataset = self.gen_IRS(save_mode=False)

        IRS_DBI_dataset = self.gen_DBI(dataset=IRS_dataset, save_mode=False)
        DataUtils.save_new_data(
            IRS_DBI_dataset, self.lang_str, "IRS", "DBI", partition="cross"
        )

        IRS_OOS_dataset = self.gen_OOS(dataset=IRS_dataset, save_mode=False)
        DataUtils.save_new_data(
            IRS_OOS_dataset, self.lang_str, "IRS", "OOS", partition="cross"
        )

        IRS_HVI_dataset = self.gen_HVI(dataset=IRS_dataset, save_mode=False)
        DataUtils.save_new_data(
            IRS_HVI_dataset, self.lang_str, "IRS", "HVI", partition="cross"
        )

        print(f"********************************")
        print(f"IRS_based_dataset gen finished~!")
        print(f"********************************")

    def cross_gen(self):
        """
        semantic x synatic
        """

        # self.cross_base_FNE()
        self.cross_base_IOE()
        # self.cross_base_IHR()
        # self.cross_base_IRS()

        print(f"********************************")
        print(f"***********[all finished]**********")
        print(f"********************************")


if __name__ == "__main__":
    go_rob = GoRobustCodeSum()
    # go_rob.gen_FNE()
    # go_rob.gen_IOE()
    # go_rob.gen_IHR()
    # go_rob.gen_IRS()
    # go_rob.gen_HVI()
    # go_rob.gen_DBI()
    # go_rob.gen_OOS()
    # go_rob.gen_origin()
    # go_rob.batch_gen()
    go_rob.cross_gen()
