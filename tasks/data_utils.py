# 读取 local_test/first/semantic/FNE/pyhon/CSN/test.jsonl文件
# 然后将内容转为dataset

import json
import os
import sys
import re
import time
from datasets import load_dataset, load_from_disk
import numpy as np
from tasks.path_utils import (
    FinalPath,
    get_input_data_path,
    get_generation_path,
    get_scores_path,
    get_input_data_path_cross,
)

data_name_dict = {
    "semantic": ["FNE", "IHR", "IOE", "IRS"],
    "synatic": ["DBI", "HVI", "OOS"],
}


class FinalDataProcess:
    def handle_reference(sample):
        from mosestokenizer import MosesDetokenizer

        docstring = " ".join(sample["docstring_tokens"]).replace("\n", "")
        if docstring[0] == "r":
            docstring = docstring[1:]
        with MosesDetokenizer("en") as detokenize:
            docstring = detokenize(docstring.strip().split())

        reference = docstring
        return reference

    def handle_generations(lang_name, generations):
        if lang_name == "python":
            return FinalDataProcess.handle_python_gen(generations)
        elif lang_name == "java":
            return FinalDataProcess.handle_java_gen(generations)
        elif lang_name == "go":
            return FinalDataProcess.handle_go_gen(generations)
        else:
            raise Exception(f"lang_name {lang_name} is not supported")

    def handle_java_gen(generations):
        new_generations = [
            gen.strip()
            .replace("The goal of this function is to ", "")
            .strip()
            .strip('"')
            .strip("'")
            for gen in generations
        ]
        return new_generations

    def handle_go_gen(generations):
        new_generations = [
            gen.strip()
            .replace("The goal of this function is to ", "")
            .strip()
            .strip('"')
            .strip("'")
            for gen in generations
        ]
        return new_generations

    def handle_python_gen(generations):
        new_generations = []
        for filling in generations:
            # 获取第一行文字
            first_line = ""
            for line in filling.split("\n"):
                if line.strip() != "":
                    first_line = line
                    break
            # 去除空格
            first_line = first_line.strip()
            new_generations.append(first_line)

        return new_generations

    def load_local_dataset(partition_name, type_name, mode_name, lang_name):
        file_path = FinalPath.get_rob_data_path(
            partition_name, type_name, mode_name, lang_name
        )
        dataset = load_from_disk(file_path)
        return dataset

    def save_ref_and_gen_data(
        model_name: str,
        partition_name: str,
        type_name: str,
        mode_name: str,
        lang_name: str,
        task_name: str,
        start_point: int,
        limit: str,
        fillings,
        references,
    ):
        # Define the path to the JSON file
        gen_path, ref_path = FinalPath.get_generation_path(
            model_name,
            partition_name,
            type_name,
            mode_name,
            lang_name,
            task_name,
            start_point,
            limit,
        )
        # 使用格式化显示时间
        print("**************************************")
        print(
            f"saving reference result to {ref_path}\n"
            + f"saving generations result to {gen_path}\n"
            + f"cur time is {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
        )
        print("**************************************")

        with open(ref_path, "w") as f:
            json.dump(references, f, indent=4)
            f.close()
        with open(gen_path, "w") as f:
            json.dump(fillings, f, indent=4)
            f.close()


def load_cross_dataset(mode_1, mode_2, lang_str):
    path = get_input_data_path_cross(mode_1, mode_2, lang_str)
    dataset = load_from_disk(path)
    return dataset


def get_score_data(model_name, task_name, type_name, mode_name, limit, score_name):
    # Define the path to the JSON file
    data_file_path = f"scores/results/{model_name}/{task_name}_{type_name}_{mode_name}_{limit}_{score_name}.json"
    # Read the JSON file
    with open(data_file_path, "r") as f:
        data = json.load(f)
    # Convert the scores to a numpy array
    scores = np.array(data)
    return scores


def get_gen_data(model_name, task_name, type_name, mode_name, limit):
    # Define the path to the JSON file
    data_file_path = f"ref_and_gen/{model_name}/{task_name}_{type_name}_{mode_name}_{limit}_fill.json"
    # Read the JSON file
    with open(data_file_path, "r") as f:
        data = json.load(f)

    return get_generations(data)


def get_prompt_data(type_name, partition_name, limit):
    dataset = load_local_datasetv2(type_name, partition_name)
    prompt_data = []
    for i in range(limit):
        code = dataset[i]["code"]
        doc = dataset[i]["docstring"]
        prompt = code.replace(doc, "<FILL_ME>", 1)
        prompt_data.append(prompt)
    return prompt_data


def handle_reference(sample):
    from mosestokenizer import MosesDetokenizer

    docstring = " ".join(sample["docstring_tokens"]).replace("\n", "")
    if docstring[0] == "r":
        docstring = docstring[1:]
    with MosesDetokenizer("en") as detokenize:
        docstring = detokenize(docstring.strip().split())

    reference = docstring
    return reference


def get_base_references(start_point, limit):
    from datasets import load_from_disk

    dataset = load_from_disk(f"local_data/second/semantic/FNE/python/CSN")
    references = []
    for i in range(start_point, limit):
        references.append(handle_reference(dataset[i]))
    return references


def get_references_from_tokens_from_first(limit):
    dataset = load_dataset(
        "json", data_files="local_data/first/semantic/FNE/python/CSN/test.jsonl"
    )["train"]

    references = []
    for i in range(limit):
        references.append(handle_reference(dataset[i]))
    return references


def get_references_from_tokens(type_name, partition_name, limit):
    from datasets import load_from_disk

    dataset = load_from_disk(
        f"local_data/second/{type_name}/{partition_name}/python/CSN"
    )
    references = []
    for i in range(limit):
        references.append(handle_reference(dataset[i]))
    return references


def get_references_from_string(type_name, partition_name, limit):
    from datasets import load_from_disk

    dataset = load_from_disk(
        f"local_data/second/{type_name}/{partition_name}/python/CSN"
    )
    references = []
    for i in range(limit):
        references.append(dataset[i]["docstring"])
    return references


def trans_to_list_list(input_list):
    output_list_list = []
    for i in input_list:
        output_list_list.append([i])
    return output_list_list


def load_local_dataset_multi(type_name, mode_name, lang_name):
    file_path = f"local_data/first/{type_name}/{mode_name}/{lang_name}/CSN/test.jsonl"
    dataset = load_dataset("json", data_files=file_path)
    return dataset


def get_generations(fillings):
    generations = []
    for filling in fillings:
        # 获取第一行文字
        first_line = ""
        for line in filling.split("\n"):
            if line != "":
                first_line = line
                break
        # 去除空格
        first_line = first_line.strip()
        generations.append(first_line)
    return generations


def save_ref_and_gen_data(
    model_name,
    task_name,
    type_name,
    mode_name,
    lang_name,
    start_point,
    limit,
    fillings,
    references,
):
    # Define the path to the JSON file
    ref_path, gen_path = get_generation_path(
        model_name, type_name, mode_name, lang_name, task_name, start_point, limit
    )
    # 使用格式化显示时间
    print("**************************************")
    print(
        f"saving reference result to {ref_path}\n"
        + f"saving generations result to {gen_path}\n"
        + f"cur time is {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    )
    print("**************************************")
    with open(ref_path, "w") as f:
        json.dump(references, f, indent=4)
        f.close()
    with open(gen_path, "w") as f:
        json.dump(fillings, f, indent=4)
        f.close()


def load_local_datasetv3(type_name, mode_name, lang_name):
    file_path = get_input_data_path(type_name, mode_name, lang_name)
    dataset = load_from_disk(file_path)
    return dataset


def load_local_datasetv2(type_name, partition_name, lang_name):
    file_path = f"local_data/second/{type_name}/{partition_name}/{lang_name}/CSN"
    dataset = load_from_disk(file_path)
    return dataset


def load_local_dataset(type_name, partition_name):
    file_path = f"local_data/first/{type_name}/{partition_name}/python/CSN/test.jsonl"

    with open(file_path, "r") as f:
        lines = f.readlines()
        print(len(lines))

    dataset = load_dataset("json", data_files=file_path)
    dataset["test"] = dataset["train"]
    print(
        f"load {type_name}/ {partition_name} dataset success, dataset shape = {len(dataset['test'])}"
    )
    return dataset


def test_load_all_datasets():
    for type_name in data_name_dict.keys():
        for partition_name in data_name_dict[type_name]:
            load_local_dataset(type_name, partition_name)


if __name__ == "__main__":
    test_load_all_datasets()
