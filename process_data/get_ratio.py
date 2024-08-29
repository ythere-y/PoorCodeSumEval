import parso
from datasets import load_from_disk
import json
import tree_sitter_java as tsj
import tree_sitter_go as tsg
from tree_sitter import Language, Parser, Node
from utils import ParserUtils, DataUtils
from tqdm import tqdm
from colorama import Fore


def read_local_dataset(lang_name, args):
    file_path = f"local_data/second/origin/origin/origin/{lang_name}/CSN"
    total_len = args["total_len"]
    dataset = load_from_disk(file_path)
    result = []
    for idx in range(total_len):
        sample = dataset[idx]
        code = sample["code"]
        doc_string = sample["docstring"]
        code = code.replace(doc_string, "<FILL_ME>", 1)
        result.append(code)
    return result


def get_score_data(lang_name, args):
    total_len = args["total_len"]
    bert_path = f"scores/{args['model_name']}/origin/origin/origin/{lang_name}/CSN/BERTScore/work_[0-{total_len}].json"
    bleu_path = f"scores/{args['model_name']}/origin/origin/origin/{lang_name}/CSN/BLEUScore/work_[0-{total_len}].json"

    with open(bert_path, "r") as f:
        score_bert = json.load(f)
    with open(bleu_path, "r") as f:
        score_bleu = json.load(f)
    return score_bert, score_bleu


def handle_python(code):
    def get_node_num_python(func_node):
        nodes = [func_node]
        node_count = 0
        while nodes:
            node = nodes.pop()
            if hasattr(node, "children"):
                nodes.extend(node.children)
            else:
                node_count += 1

        return node_count

    def get_id_node_num_python(func_node):
        nodes = [func_node]
        node_count = 0
        while nodes:
            node = nodes.pop()
            if node.type == "name":
                node_count += 1
            else:
                if hasattr(node, "children"):
                    nodes.extend(node.children)
        return node_count

    def get_comment_ratio(code):
        total_counter = len(code)
        comment_counter = 0
        code_lines = code.split("\n")
        for line in code_lines:
            sp_line = line.strip()
            if sp_line.startswith("#"):
                # print(line)
                comment_counter += len(sp_line)
        return comment_counter / total_counter

    ast = parso.parse(code)
    func_node = list(ast.iter_funcdefs())[0]
    # 统计ast下node的数量
    node_count = get_node_num_python(func_node)
    identifier_count = get_id_node_num_python(func_node)
    id_ratio = identifier_count / node_count
    comment_ratio = get_comment_ratio(code)

    result = {"identifier_ratio": id_ratio, "comment_ratio": comment_ratio}
    return result


def sort_by(tag_name, data, args):
    total_len = args["total_len"]

    def divid_and_table(divid_by):
        print(f"* divid by {divid_by} %\n")
        threthold = int(total_len * divid_by / 100)
        first_half = sorted_data[0:threthold]
        second_half = sorted_data[total_len - threthold :]
        first_bleu = sum([x["bleu_score"] for x in first_half]) / threthold
        first_bert = sum([x["bert_score"] for x in first_half]) / threthold
        second_bleu = sum([x["bleu_score"] for x in second_half]) / threthold
        second_bert = sum([x["bert_score"] for x in second_half]) / threthold
        first_mean_tag = sum([x[tag_name] for x in first_half]) / threthold
        second_mean_tag = sum([x[tag_name] for x in second_half]) / threthold
        print(
            f"group by top **{divid_by} %** | **BLEU** | **BERT** | tag = **{tag_name}**"
        )
        print(f"--- | --- | --- | ---")
        print(
            f"down group | {first_bleu:.4f} | {first_bert:.4f} | {first_mean_tag:.4f}"
        )
        print(
            f"up group | {second_bleu:.4f} | {second_bert:.4f} | {second_mean_tag:.4f}"
        )

    print(f"### sort by {tag_name}\n")
    sorted_data = sorted(data, key=lambda x: x[tag_name])
    divid_and_table(50)
    print()

    divid_and_table(25)
    print()

    divid_and_table(10)
    print()
    return sorted_data


def handle_java(code):
    def get_comment_ratio_java(code):
        total_counter = len(code)
        comment_counter = 0
        code_lines = code.split("\n")
        for line in code_lines:
            sp_line = line.strip()
            if sp_line.startswith("//"):
                # print(line)
                comment_counter += len(sp_line)
        return comment_counter / total_counter

    JAVA_LANGUAGE = Language(tsj.language())

    parser = Parser(JAVA_LANGUAGE)

    tree = parser.parse(bytes(code, "utf8"))

    root_node = tree.root_node
    all_ide_list = []
    ParserUtils.traverse_type(root_node, all_ide_list, "identifier")
    total_node_num = ParserUtils.get_node_num(root_node)

    id_ratio = len(all_ide_list) / total_node_num
    comment_ratio = get_comment_ratio_java(code)

    result = {
        "identifier_ratio": id_ratio,
        "comment_ratio": comment_ratio,
    }
    return result


def handle_go(code):

    def get_comment_ratio_go(code):
        total_counter = len(code)
        comment_counter = 0
        code_lines = code.split("\n")
        for line in code_lines:
            sp_line = line.strip()
            if sp_line.startswith("//"):
                # print(line)
                comment_counter += len(sp_line)
        return comment_counter / total_counter

    GO_LANGUAGE = Language(tsg.language())

    parser = Parser(GO_LANGUAGE)

    tree = parser.parse(bytes(code, "utf8"))

    root_node = tree.root_node
    all_ide_list = []
    ParserUtils.traverse_type(root_node, all_ide_list, "identifier")
    total_node_num = ParserUtils.get_node_num(root_node)

    id_ratio = len(all_ide_list) / total_node_num
    comment_ratio = get_comment_ratio_go(code)

    result = {
        "identifier_ratio": id_ratio,
        "comment_ratio": comment_ratio,
    }
    return result


def count_ratio_gen(lang_name, args):
    total_len = args["total_len"]
    score_bert, score_bleu = get_score_data(lang_name, args)
    code = read_local_dataset(lang_name, args)

    bind_data = []
    for idx in tqdm(range(total_len)):
        if lang_name == "java":
            result = handle_java(code[idx])
        elif lang_name == "python":
            result = handle_python(code[idx])
        elif lang_name == "go":
            result = handle_go(code[idx])
        total = result["identifier_ratio"] + result["comment_ratio"] * 5
        cur_data = {
            "identifier_ratio": result["identifier_ratio"],
            "comment_ratio": result["comment_ratio"],
            "bert_score": score_bert[idx],
            "bleu_score": score_bleu[idx] * 100,
            "code": code[idx],
            "total": total,
        }
        bind_data.append(cur_data)
    sort_by("identifier_ratio", bind_data, args)
    sort_by("comment_ratio", bind_data, args)
    sort_by("total", bind_data, args)


if __name__ == "__main__":
    lang_name = "java"
    # model_name = "CodeLlama-7b-hf"
    # model_name = "CodeBERT"
    args = {
        # "model_name": "CodeBERT",
        "model_name": "CodeLlama-7b-hf",
    }
    if args["model_name"] == "CodeBERT":
        if lang_name == "java":
            total_len = 10955
        elif lang_name == "python":
            total_len = 14917
        elif lang_name == "go":
            total_len = 8122
    else:
        total_len = 2000
    args["total_len"] = total_len
    print(f'**********Evaluating {lang_name} with model {args["model_name"]}**********')
    count_ratio_gen(lang_name, args)
