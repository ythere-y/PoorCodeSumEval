from typing import List
from colorama import Fore
import os
from datasets import Dataset, load_from_disk
import pandas as pd
import random
from tree_sitter import Language, Parser, Node


class CommentUtils:
    def strip_c_style_comment_delimiters(comment: str) -> str:
        comment_lines = comment.split("\n")
        cleaned_lines = []
        for l in comment_lines:
            l = l.strip()
            if l.endswith("*/"):
                l = l[:-2]
            if l.startswith("*"):
                l = l[1:]
            elif l.startswith("/**"):
                l = l[3:]
            elif l.startswith("//"):
                l = l[2:]
            cleaned_lines.append(l.strip())
        return "\n".join(cleaned_lines)

    def get_docstring_summary(docstring: str) -> str:
        """Get the first lines of the documentation comment up to the empty lines."""
        if "\n\n" in docstring:
            return docstring.split("\n\n")[0]
        elif "@" in docstring:
            return docstring[
                : docstring.find("@")
            ]  # This usually is the start of a JavaDoc-style @param comment.
        return docstring


class DataUtils:
    def get_save_dir_path(
        lang, type_name, mode_name, partition="second", dataset_part="CSN"
    ):
        if partition == "cross":
            path = (
                f"local_data/second/cross/{type_name}/{mode_name}/{lang}/{dataset_part}"
            )
        else:
            path = (
                f"local_data/{partition}/{type_name}/{mode_name}/{lang}/{dataset_part}"
            )

        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def save_new_data(
        new_data, lang, type_str, mode, partition="second", dataset_part="CSN"
    ):
        # 将新的列表转换为pandas DataFrame
        df = pd.DataFrame(new_data)

        # 将pandas DataFrame转换为dataset
        new_dataset = Dataset.from_pandas(df)

        # 将新的dataset保存到文件test.json中
        save_path = DataUtils.get_save_dir_path(
            lang, type_str, mode, partition=partition, dataset_part=dataset_part
        )
        new_dataset.save_to_disk(save_path)
        test_reload = load_from_disk(save_path)
        print(f"save data to {save_path}, size: {test_reload.shape}")

    IF_BODY_STORE = {
        "python": [
            "\n    h = line(y, self._effective_thickness(p), 0.0)\n    return h.sum()",
            "\n    super(Composite,self).state_pop()\n    for gen in self.generators:\n        gen.state_pop()",
        ],
        "java": [
            "\n    return Math.sin(Math.PI * x) / (Math.PI * x);",
            "\n    return !Strings.isNullOrEmpty(objectName)\n        && objectName.endsWith(GoogleCloudStorage.PATH_DELIMITER);",
        ],
        "go": [
            "\n    return storeTrustedKey(ks.LocalRootPath, r)",
            "\n    return filepath.Join(exitedGarbageDir(p.dataDir), p.UUID.String())",
        ],
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
        ret = DataUtils.IF_BODY_STORE[lang][random_case]
        ret = ret.replace("\n", f"\n{base_code_indent}")
        return ret

    def get_body_string(code_string, doc_string):
        end_position = code_string.find(doc_string) + len(doc_string)
        while code_string[end_position : end_position + 3] != '"""':
            end_position += 1
        while code_string[end_position] != "\n":
            end_position += 1
        end_position += 1
        return code_string[end_position:]


class ParserUtils:
    def get_node_num(node) -> int:
        node_count = 0
        nodes = [node]
        node_count = 0
        while nodes:
            node = nodes.pop()
            if len(node.children) == 0:
                node_count += 1
            else:
                nodes.extend(node.children)
        return node_count

    def shallow_traverse_type(node, result: List, kind: str) -> None:
        """
        浅层遍历，对于kind类型的节点，不会再进入其子节点
        查找binary expression的时候，找最外层最大的expression，忽略内部的
        """
        if node == None:
            return
        if node.type == kind:
            result.append(node)
            return
        if not node.children:
            return
        for n in node.children:
            ParserUtils.shallow_traverse_type(n, result, kind)

    def traverse_type(node, results: List, kind: str) -> None:
        """
        collect a specific type of Node
        """
        if node == None:
            return
        if node.type == kind:
            results.append(node)
        if not node.children:
            return
        for n in node.children:
            ParserUtils.traverse_type(n, results, kind)

    def replace_from_span(position, blob: str, place_taker: str) -> str:
        """
        position[0] = start_point
        position[1] = end_point
        """
        line_start = position[0][0]
        line_end = position[1][0]
        char_start = position[0][1]
        char_end = position[1][1]
        lines = blob.split("\n")
        target_lines = list(lines[line_start:line_end])
        head = lines[line_start][:char_start]
        tail = lines[line_end][char_end:]
        new_insert = head + place_taker + tail
        # 逆序pop是因为，pop会实时地改变后面的元素的下表，正序pop会有问题
        # line_start 位置不进行pop，是因为下面将会对其进行替换，省去pop的步骤
        for _idx in range(line_end, line_start, -1):
            lines.pop(_idx)
        lines[line_start] = new_insert
        new_blob = "\n".join(lines)
        return new_blob

    def replace_node_str(node: Node, blob: str, place_taker: str) -> str:
        return ParserUtils.replace_from_span(
            (node.start_point, node.end_point), blob, place_taker
        )

    def replace_nodes_str(nodes_info: list, blob: str):
        # nodes_info = [(node,place_taker)...]
        # 按照node的位置排序
        nodes_info.sort(key=lambda node: node[0].start_point)
        # 倒序遍历node做替换
        for node, place_taker in nodes_info[::-1]:
            blob = ParserUtils.replace_node_str(node, blob, place_taker)
        return blob

    def add_color_to_nodes(nodes: list, blob: str, color: str) -> str:
        # 先按照node的位置进行排序
        nodes.sort(key=lambda node: node.start_point)
        # 倒序遍历node做替换
        for node in nodes[::-1]:
            blob = ParserUtils.add_color_to_node(node, blob, color)
        return blob

    def add_color_to_node(node: Node, blob: str, color: str) -> str:
        return ParserUtils.replace_node_str(
            node, blob, color + node.text.decode("utf8") + Fore.RESET
        )

    def get_node_indent_count(node: Node, blob: str) -> int:
        code_lines = blob.split("\n")
        prefix = code_lines[node.start_point[0]][: node.start_point[1]]
        count = 0
        for c in prefix:
            if c == "\t":
                count += 4
            else:
                count += 1
        return count
