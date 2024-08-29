import random
import os
import pandas as pd
import parso
import time

from tree_sitter import Language, Parser, Node
from datasets import load_dataset, Dataset, load_from_disk
from colorama import Fore, Back, Style
from tqdm import tqdm

from statistics import python_dict


class PythonParsoParser:
    def find_function_body_block(func_node):
        return func_node.children[-1]

    def get_body_block_string(body_node):
        res = ""
        try:
            for child in body_node.children[2:]:
                res += child.get_code()
        except:
            raise Exception("get_body_block_string error")
        return res

    def get_ast_prefix_string(func_node):
        """
        获取body_string前面的所有部分
        """
        res = ""
        try:
            if len(func_node.parent.children) == 3:
                # it is async
                res += func_node.parent.children[0].get_code()
            elif len(func_node.parent.children) > 3:
                print(f"error ~!")

            for child in func_node.children[:-1]:
                res += child.get_code()
            for child in func_node.children[-1].children[:2]:
                res += child.get_code()
        except:
            raise Exception("get_prefix_string error")
        return res

    def get_all_identifers(func_node):
        nodes = [func_node]
        identifier_name_to_nodes = {}
        while len(nodes) > 0:
            cur_node = nodes.pop(0)
            # print(cur_node.type)
            if cur_node.type == "name" and cur_node.parent.type != "trailer":
                name = cur_node.value
                which_son = cur_node.parent.children.index(cur_node)
                if cur_node.parent.type == "atom_expr":
                    # 如果是atom expression，那么要考虑外部函数引用情况，要排除外部函数引用
                    # 排除方法就是检查这个名字是否是前文定义过的变量
                    if name in identifier_name_to_nodes:
                        identifier_name_to_nodes[name].append(cur_node)
                        # print(
                        #     f"name:{Fore.YELLOW}{name}{Fore.RESET},parent type = ["
                        #     + f"{Fore.BLUE}{cur_node.parent.type}{Fore.RESET}] "
                        #     + f"which son = {Fore.RED}{which_son}{Fore.RESET}"
                        #     + f"parent code =\n{cur_node.parent.get_code()}"
                        # )
                else:
                    if name in identifier_name_to_nodes:
                        identifier_name_to_nodes[name].append(cur_node)
                    else:
                        identifier_name_to_nodes[name] = [cur_node]
            else:
                try:
                    for child in cur_node.children:
                        nodes.append(child)
                except:
                    pass
        # for key in identifier_name_to_nodes:
        #     print(f"{key}:{len(identifier_name_to_nodes[key])}")
        return identifier_name_to_nodes

    TARGET_COMPARISON_OP = {
        "<": ">",
        "<=": ">=",
        ">": "<",
        ">=": "<=",
        "==": "==",
        "!=": "!=",
    }

    def oper_swap(node):
        # if hasattr(node, "value") and node.value == "<":
        # print(node)
        swap_happened = False
        if hasattr(node, "type"):
            if node.type == "comparison":
                # parson的定义：comparison: expr (comp_op expr)*
                # 为了处理连续的比较 a<b<c -> c>b>a
                # 校验是否都是目标操作符
                work_flag = True
                for child in node.children:
                    if (
                        child.type == "operator"
                        and child.value not in PythonParsoParser.TARGET_COMPARISON_OP
                    ):
                        work_flag = False
                        break
                if work_flag:
                    node.children = node.children[::-1]
                    for child in node.children:
                        if child.type == "operator":
                            child.value = PythonParsoParser.TARGET_COMPARISON_OP[
                                child.value
                            ]
                            swap_happened = True
            elif node.type == "arith_expr":
                # parso的定义： arith_expr: term (('+'|'-') term)*
                # 校验是否都是目标操作符
                work_flag = True
                for child in node.children:
                    if child.type == "operator" and child.value != "+":
                        work_flag = False
                        break
                if work_flag:
                    node.children = node.children[::-1]
                    swap_happened = True

            elif node.type == "term":
                # parson的定义：term: factor (('*'|'@'|'/'|'%'|'//') factor)*
                # 校验是否都是目标操作符
                work_flag = True
                for child in node.children:
                    if child.type == "operator" and child.value != "*":
                        work_flag = False
                        break
                if work_flag:
                    node.children = node.children[::-1]
                    swap_happened = True

        try:
            for child in node.children:
                if PythonParsoParser.oper_swap(child):
                    swap_happened = True
        except:
            pass
        return swap_happened

    def travel(node):
        print(node.get_code())
        try:
            for child in node.children:
                PythonParsoParser.travel(child)
        except:
            pass


class Utils:
    def get_save_dir_path(lang, type, mode, partition="second"):
        path = os.path.join(f"local_data/{partition}/{type}/{mode}/{lang}/CSN")
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def save_new_data(new_data, lang, type_str, mode, partition="second"):
        # 将新的列表转换为pandas DataFrame
        df = pd.DataFrame(new_data)

        # 将pandas DataFrame转换为dataset
        new_dataset = Dataset.from_pandas(df)

        # 将新的dataset保存到文件test.json中
        save_path = Utils.get_save_dir_path(lang, type_str, mode, partition=partition)
        new_dataset.save_to_disk(save_path)
        test_reload = load_from_disk(save_path)
        print(f"save data to {save_path}, size: {test_reload.shape}")

    IF_BODY_STORE = {
        "python": [
            "\n    h = line(y, self._effective_thickness(p), 0.0)\n    return h.sum()",
            "\n    super(Composite,self).state_pop()\n    for gen in self.generators:\n        gen.state_pop()",
        ]
    }

    def display_before_and_after(before_string, after_string):
        # print(f"{Fore.RED}idx = [{idx}]{Fore.RESET}")
        print(
            f"""{Fore.YELLOW}before handle:{Fore.RESET}
{before_string}
{Fore.GREEN}after handle:{Fore.RESET}
{after_string}"""
        )

    def get_dead_if_body(lang, base_code_indent):
        random_case = random.randint(0, 1)
        ret = Utils.IF_BODY_STORE[lang][random_case]
        ret = ret.replace("\n", f"\n{base_code_indent}")
        return ret

    # TODO: 未完成,不要使用，有问题
    def remove_doc_string(lang, code_string, doc_string):
        if lang == "python":
            start_position = code_string.find(doc_string)
            end_position = start_position + len(doc_string)
            try:
                # 向前追溯，找到连续三个引号"""之后，再向前找到第一个\n符号，这个\n符号要保留
                while code_string[start_position : start_position + 3] != '"""':
                    start_position -= 1
                while code_string[start_position] != "\n":
                    start_position -= 1
                start_position += 1
                # 向后追溯，找到连续三个引号"""之后，再向后找到第一个\n符号，这个\n符号也要删去
                while code_string[end_position : end_position + 3] != '"""':
                    end_position += 1
                while code_string[end_position] != "\n":
                    end_position += 1
                end_position += 1
                result_code = code_string[0:start_position] + code_string[end_position:]
            except:
                raise Exception(
                    "remove doc string error,code string is: \n" + code_string,
                    "doc string is: \n" + doc_string,
                )

        return result_code.decode("utf-8")

    def get_body_string(code_string, doc_string):
        end_position = code_string.find(doc_string) + len(doc_string)
        while code_string[end_position : end_position + 3] != '"""':
            end_position += 1
        while code_string[end_position] != "\n":
            end_position += 1
        end_position += 1
        return code_string[end_position:]


class PythonRobustCodeSum:
    def __init__(self) -> None:
        DATASET_PATH = "path_to_code_x/code_x_glue_ct_code_to_text"

        self.dataset = load_dataset(DATASET_PATH, "python")["test"]
        print(f"load data from codex blue, size: {self.dataset.shape}")
        self.lang_str = "python"
        self.langurage = Language("process_data/build/my-languages.so", self.lang_str)
        self.parser = Parser()
        self.parser.set_language(self.langurage)
        pass

    def gen_FNE(self, dataset=None, save_mode=True) -> None:
        """
        找到函数identifier
        替换为v0
        """
        if dataset == None:
            dataset = self.dataset
        new_data = []
        mode_name = "FNE"
        for i in tqdm(range(0, len(dataset)), desc=f"gen {mode_name}"):
            row = dataset[i]
            code_string = row["code"]
            parsed_ast = parso.parse(code_string)
            if len(list(parsed_ast.iter_funcdefs())) == 0:
                continue
            func_node = list(parsed_ast.iter_funcdefs())[0]
            for child in func_node.children:
                if child.type == "name":
                    child.value = "v0"
                    break

            new_code_string = parsed_ast.get_code()
            # 展示结果
            # Utils.display_before_and_after(code_string, new_code_string)
            # 将处理后的数据添加到新的列表中
            new_row = row.copy()  # 复制当前行的数据
            new_row["code"] = new_code_string  # 更新"code"字段
            new_data.append(new_row)  # 添加到新的数据列表中
        if save_mode:
            Utils.save_new_data(new_data, self.lang_str, "semantic", mode_name)
        return new_data

    def gen_IOE(self, dataset=None, save_mode=True):
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

            parsed_ast = parso.parse(code_string)
            if len(list(parsed_ast.iter_funcdefs())) == 0:
                continue
            func_node = list(parsed_ast.iter_funcdefs())[0]
            identifier_name_to_nodes = PythonParsoParser.get_all_identifers(func_node)

            # 从identifier_name_to_nodes中随机选取80%的identifier
            # 使用random.sample
            rate = 0.8
            identifier_list = list(identifier_name_to_nodes.keys())
            final_identifier_list = random.sample(
                identifier_list, int(len(identifier_list) * rate)
            )

            # 生成替换字典
            # 使用v_0,v_1按顺序替换
            replace_dict = {}
            for i, identifier in enumerate(final_identifier_list):
                replace_dict[identifier] = f"v_{i}"
            # print(replace_dict)

            # 替换identifier

            for identifier, replace_str in replace_dict.items():
                nodes = identifier_name_to_nodes[identifier]
                for node in nodes:
                    node.value = replace_str
            new_code_string = parsed_ast.get_code()

            # 展示数据
            # Utils.display_before_and_after(code_string, new_code_string)
            # 将处理后的数据添加到新的列表中
            new_row = row.copy()  # 复制当前行的数据
            new_row["code"] = new_code_string  # 更新"code"字段
            new_data.append(new_row)  # 添加到新的数据列表中

            # break
        if save_mode:
            Utils.save_new_data(new_data, self.lang_str, "semantic", mode_name)
        return new_data

    def gen_IHR(self, dataset=None, save_mode=True):
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

            parsed_ast = parso.parse(code_string)
            if len(list(parsed_ast.iter_funcdefs())) == 0:
                continue
            func_node = list(parsed_ast.iter_funcdefs())[0]
            identifier_name_to_nodes = PythonParsoParser.get_all_identifers(func_node)

            # 获取高频词数组
            high_freq_list = [word[0] for word in python_dict]

            # 获取所有的identifier
            identifier_list = list(identifier_name_to_nodes.keys())

            # 对所有identifier随机匹配一个高频词，且不能有重复的
            # 做法是，先shuffle high_freq_dict，然后按顺序匹配

            # shuffle
            random.shuffle(high_freq_list)

            # 匹配
            replace_dict = {}
            for identifier in identifier_list:
                replace_dict[identifier] = high_freq_list.pop()

            # print(replace_dict)

            # 替换identifier

            for identifier, replace_str in replace_dict.items():
                nodes = identifier_name_to_nodes[identifier]
                for node in nodes:
                    node.value = replace_str

            new_code_string = parsed_ast.get_code()
            # 展示数据
            # Utils.display_before_and_after(code_string, new_code_string)
            # 将处理后的数据添加到新的列表中
            new_row = row.copy()  # 复制当前行的数据
            new_row["code"] = new_code_string  # 更新"code"字段
            new_data.append(new_row)  # 添加到新的数据列表中

            # break
        if save_mode:
            Utils.save_new_data(new_data, self.lang_str, "semantic", mode_name)
        return new_data

    def gen_IRS(self, dataset=None, save_mode=True):
        """
        IRS: 随机调换标识符名字（使用函数片段中已有的标识符名字）
        # 找到所有的identifier_list
        # 打乱ide_list的顺序，然后形成对应，确定替换关系(ide -> ide')
        # 在code中替换对象，作为混淆结果
        """
        if dataset == None:
            dataset = self.dataset
        new_data = []
        mode_name = "IRS"
        for i in tqdm(range(0, len(dataset)), desc=f"gen {mode_name}"):
            row = dataset[i]
            code_string = row["code"]

            parsed_ast = parso.parse(code_string)
            if len(list(parsed_ast.iter_funcdefs())) == 0:
                continue
            func_node = list(parsed_ast.iter_funcdefs())[0]
            identifier_name_to_nodes = PythonParsoParser.get_all_identifers(func_node)

            # 获取所有的identifier
            identifier_list = list(identifier_name_to_nodes.keys())

            # 匹配,使用zip，使用循环替换的方法来做到打乱效果
            replace_dict = dict(zip(identifier_list[:-1], identifier_list[1:]))
            replace_dict.update({identifier_list[-1]: identifier_list[0]})

            # 替换identifier
            for identifier, replace_str in replace_dict.items():
                nodes = identifier_name_to_nodes[identifier]
                for node in nodes:
                    node.value = replace_str

            new_code_string = parsed_ast.get_code()
            # 展示数据
            # Utils.display_before_and_after(code_string, new_code_string)
            # 将处理后的数据添加到新的列表中
            new_row = row.copy()  # 复制当前行的数据
            new_row["code"] = new_code_string  # 更新"code"字段
            new_data.append(new_row)  # 添加到新的数据列表中

            # break
        if save_mode:
            Utils.save_new_data(new_data, self.lang_str, "semantic", mode_name)
        return new_data

    def gen_OOS(self, dataset=None, save_mode=True):
        """
        OOS：计算交换
        """
        if dataset == None:
            dataset = self.dataset
        new_data = []
        mode_name = "OOS"
        for i in tqdm(range(0, len(dataset)), desc=f"gen {mode_name}"):
            row = dataset[i]
            code_string = row["code"]
            parsed_ast = parso.parse(code_string)
            if len(list(parsed_ast.iter_funcdefs())) == 0:
                continue
            func_node = list(parsed_ast.iter_funcdefs())[0]

            cur_swap_happened = PythonParsoParser.oper_swap(func_node)

            new_code_string = parsed_ast.get_code()
            # 展示数据
            # Utils.display_before_and_after(code_string, new_code_string)
            # 将处理后的数据添加到新的列表中
            new_row = row.copy()  # 复制当前行的数据
            new_row["code"] = new_code_string  # 更新"code"字段
            new_row["swap_happened"] = cur_swap_happened
            new_data.append(new_row)  # 添加到新的数据列表中
        if save_mode:
            Utils.save_new_data(new_data, self.lang_str, "synatic", mode_name)
        return new_data

    def gen_DBI(self, dataset=None, save_mode=True):
        """
        DBI：插入死分支
        """
        if dataset == None:
            dataset = self.dataset
        new_data = []
        # 遍历dataset中的每一行
        mode_name = "DBI"
        for i in tqdm(range(0, len(dataset)), desc=f"gen {mode_name}"):
            row = dataset[i]
            code_string = row["code"]
            # code_string = """aynsc def test():\n    print("hello")"""
            # 抽取函数的body，并加上退格
            parsed_ast = parso.parse(code_string)
            if len(list(parsed_ast.iter_funcdefs())) == 0:
                continue
            func_node = list(parsed_ast.iter_funcdefs())[0]
            body_block = PythonParsoParser.find_function_body_block(func_node)
            body_block_string = PythonParsoParser.get_body_block_string(body_block)
            # 每一行加一个退格
            body_block_string = "    " + body_block_string.replace("\n", "\n    ")
            # 获取prefix,找第一行代码，不断向下找child，直到某个child有prefix
            try:
                base_child = func_node.children[-1].children[1]
                while not hasattr(base_child, "prefix"):
                    base_child = base_child.children[0]
                base_code_indent = base_child.prefix
            except:
                print("error~!#")
            # 注意，insert_body首行自带\n
            insert_body = Utils.get_dead_if_body(self.lang_str, base_code_indent)
            random_case = random.randint(0, 1)
            if random_case == 0:
                # 插入一个if true 的语句
                new_body_string = (
                    f"{base_code_indent}if True:\n"
                    + f"{body_block_string}\n"
                    + f"{base_code_indent}else:"
                    + f"{insert_body}"
                )
            else:
                # 插入一个if false 的语句
                new_body_string = (
                    f"{base_code_indent}if False:"
                    + f"{insert_body}\n"
                    + f"{base_code_indent}else:\n"
                    + f"{body_block_string}"
                )

            # 最后和function的头部拼接起来
            prefix_string = PythonParsoParser.get_ast_prefix_string(func_node)

            new_code_string = prefix_string + new_body_string
            # 展示数据
            # Utils.display_before_and_after(code_string, new_code_string)
            # 将处理后的数据添加到新的列表中
            new_row = row.copy()  # 复制当前行的数据
            new_row["code"] = new_code_string  # 更新"code"字段
            new_data.append(new_row)  # 添加到新的数据列表中

            # break
        if save_mode:
            Utils.save_new_data(new_data, self.lang_str, "synatic", mode_name)
        return new_data

    def gen_HVI(self, dataset=None, save_mode=True):
        """
        HVI：加入无意义的高频次的变量声明
        """
        if dataset == None:
            dataset = self.dataset
        new_data = []
        mode_name = "HVI"
        for i in tqdm(range(0, len(dataset)), desc=f"gen {mode_name}"):
            row = dataset[i]
            code_string = row["code"]

            parsed_ast = parso.parse(code_string)
            if len(list(parsed_ast.iter_funcdefs())) == 0:
                continue
            func_node = list(parsed_ast.iter_funcdefs())[0]
            identifier_name_to_nodes = PythonParsoParser.get_all_identifers(func_node)

            insert_list = []
            insert_num = 3
            for i in range(len(python_dict)):
                if not python_dict[i][0] in identifier_name_to_nodes:
                    insert_list.append(python_dict[i][0])
                    insert_num -= 1
                if insert_num == 0:
                    break

            body_node = PythonParsoParser.find_function_body_block(func_node)

            # 统计不是newline的node，并记录在list中
            not_newline_list = []
            for i in range(len(body_node.children)):
                if body_node.children[i].type != "newline":
                    not_newline_list.append(i)

            # 制作一个随机数，从中选择一个位置进行插入，这个位置不能是第一个（第一个是doc）
            insert_index = random.choice(not_newline_list[1:])

            # 制作插入的文本insert_text

            # 获取插入位置的退格信息
            insert_indent = body_node.children[insert_index].get_first_leaf().prefix
            insert_text = ""
            for i in range(len(insert_list)):
                # 插入的是变量赋值语句，值从0-100中随机生成
                insert_text += (
                    insert_indent + insert_list[i] + f"={random.randint(0,100)}" + "\n"
                )

            # 将insert_text插入到body_node中
            body_node.children[insert_index].get_first_leaf().prefix = (
                insert_text + insert_indent
            )

            # print(body_node.children[insert_index].get_first_leaf().prefix)
            # print(body_node.get_code())
            # print(parsed_ast.get_code())

            new_code_string = parsed_ast.get_code()
            # 展示数据
            # Utils.display_before_and_after(code_string, new_code_string)
            # 将处理后的数据添加到新的列表中
            new_row = row.copy()  # 复制当前行的数据
            new_row["code"] = new_code_string  # 更新"code"字段
            new_data.append(new_row)  # 添加到新的数据列表中

            # break
        if save_mode:
            Utils.save_new_data(new_data, self.lang_str, "synatic", mode_name)
        return new_data

    def gen_origin(self, dataset=None, save_mode=True):
        if dataset == None:
            dataset = self.dataset
        new_data = []
        mode_name = "origin"
        for i in tqdm(range(0, len(dataset)), desc=f"gen {mode_name}"):
            row = dataset[i]
            code_string = row["code"]

            parsed_ast = parso.parse(code_string)
            if len(list(parsed_ast.iter_funcdefs())) == 0:
                continue

            new_row = row.copy()  # 复制当前行的数据
            new_data.append(new_row)  # 添加到新的数据列表中

            # break
        if save_mode:
            Utils.save_new_data(new_data, self.lang_str, "origin", mode_name)
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
        Utils.save_new_data(
            FNE_DBI_dataset, self.lang_str, "FNE", "DBI", partition="cross"
        )

        FNE_OOS_dataset = self.gen_OOS(dataset=FNE_dataset, save_mode=False)
        Utils.save_new_data(
            FNE_OOS_dataset, self.lang_str, "FNE", "OOS", partition="cross"
        )

        FNE_HVI_dataset = self.gen_HVI(dataset=FNE_dataset, save_mode=False)
        Utils.save_new_data(
            FNE_HVI_dataset, self.lang_str, "FNE", "HVI", partition="cross"
        )

        print(f"********************************")
        print(f"FNE_based_dataset gen finished~!")
        print(f"********************************")

    def cross_base_IOE(self):
        IOE_dataset = self.gen_IOE(save_mode=False)

        IOE_DBI_dataset = self.gen_DBI(dataset=IOE_dataset, save_mode=False)
        Utils.save_new_data(
            IOE_DBI_dataset, self.lang_str, "IOE", "DBI", partition="cross"
        )

        IOE_OOS_dataset = self.gen_OOS(dataset=IOE_dataset, save_mode=False)
        Utils.save_new_data(
            IOE_OOS_dataset, self.lang_str, "IOE", "OOS", partition="cross"
        )

        IOE_HVI_dataset = self.gen_HVI(dataset=IOE_dataset, save_mode=False)
        Utils.save_new_data(
            IOE_HVI_dataset, self.lang_str, "IOE", "HVI", partition="cross"
        )

        print(f"********************************")
        print(f"IOE_based_dataset gen finished~!")
        print(f"********************************")

    def cross_base_IHR(self):
        IHR_dataset = self.gen_IHR(save_mode=False)

        IHR_DBI_dataset = self.gen_DBI(dataset=IHR_dataset, save_mode=False)
        Utils.save_new_data(
            IHR_DBI_dataset, self.lang_str, "IHR", "DBI", partition="cross"
        )

        IHR_OOS_dataset = self.gen_OOS(dataset=IHR_dataset, save_mode=False)
        Utils.save_new_data(
            IHR_OOS_dataset, self.lang_str, "IHR", "OOS", partition="cross"
        )

        IHR_HVI_dataset = self.gen_HVI(dataset=IHR_dataset, save_mode=False)
        Utils.save_new_data(
            IHR_HVI_dataset, self.lang_str, "IHR", "HVI", partition="cross"
        )

        print(f"********************************")
        print(f"IHR_based_dataset gen finished~!")
        print(f"********************************")

    def cross_base_IRS(self):
        IRS_dataset = self.gen_IRS(save_mode=False)

        IRS_DBI_dataset = self.gen_DBI(dataset=IRS_dataset, save_mode=False)
        Utils.save_new_data(
            IRS_DBI_dataset, self.lang_str, "IRS", "DBI", partition="cross"
        )

        IRS_OOS_dataset = self.gen_OOS(dataset=IRS_dataset, save_mode=False)
        Utils.save_new_data(
            IRS_OOS_dataset, self.lang_str, "IRS", "OOS", partition="cross"
        )

        IRS_HVI_dataset = self.gen_HVI(dataset=IRS_dataset, save_mode=False)
        Utils.save_new_data(
            IRS_HVI_dataset, self.lang_str, "IRS", "HVI", partition="cross"
        )

        print(f"********************************")
        print(f"IRS_based_dataset gen finished~!")
        print(f"********************************")

    def cross_gen(self):
        """
        semantic x synatic
        """

        self.cross_base_FNE()
        self.cross_base_IOE()
        self.cross_base_IHR()
        self.cross_base_IRS()

        print(f"********************************")
        print(f"***********[all finished]**********")
        print(f"********************************")


if __name__ == "__main__":
    robust = PythonRobustCodeSum()
    # robust.gen_FNE()
    # robust.gen_DBI()
    # robust.gen_HVI()
    robust.gen_IOE()
    # robust.gen_IHR()
    # robust.gen_IRS()
    # robust.gen_OOS()
    # robust.gen_origin()
    # robust.batch_gen()
    robust.cross_gen()
