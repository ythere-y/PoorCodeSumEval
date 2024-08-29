import os
import json
from path_utils import FinalPath
from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np

TEST_MODEL = "codellama/CodeLlama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)


def get_char_num_per_line(code_data):
    result = []
    for code in code_data:
        print(code)
        code_lines = code.split("\n")
        char_num_per_line = 0
        line_counter = 0
        for line in code_lines:
            cur_char_num = len(line)
            print(f"{cur_char_num} -> {line}")
            if cur_char_num <= 2:
                continue
            char_num_per_line += cur_char_num
            line_counter += 1
        char_num_per_line = char_num_per_line / line_counter
        result.append(char_num_per_line)
        break
    return result


def get_code_token_num_per_line(code_data):
    result = []
    for code in code_data:
        code_lines = code.split("\n")
        token_num_per_line = 0
        line_counter = 0
        for line in code_lines:
            cur_token_num = len(tokenizer.tokenize(line))
            if cur_token_num <= 2:
                continue
            token_num_per_line += cur_token_num
            line_counter += 1
        token_num_per_line = token_num_per_line / line_counter
        result.append(token_num_per_line)
    return result


def get_identifier_ratio(code_data):
    result = []
    for code in code_data:
        code_lines = code.split("\n")
        identifier_num = 0
        token_num = 0
        for line in code_lines:
            tokens = tokenizer.tokenize(line)
            for token in tokens:
                token_num += 1
                if token.isidentifier():
                    identifier_num += 1
        result.append(identifier_num / token_num)
    return result


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


def load_local_datasetv2(type_name, partition_name, mode_name, lang_name):
    file_path = (
        f"local_data/second/{type_name}/{partition_name}/{mode_name}/{lang_name}/CSN"
    )
    dataset = load_from_disk(file_path)
    return dataset


from data_utils import (
    FinalDataProcess,
)


def data_proccess():

    language = "python"
    dataset = load_local_datasetv2("origin", "origin", "origin", language)

    # gen_path,ref_path = FinalPath.get_generation_path("CodeLlama-7b-hf","origin",'origin','origin','java',task_name="work",start_point=0,limit=2000)
    gen_path = f"ref_and_gen/CodeLlama-7b-hf/origin/origin/origin/{language}/CSN/work_gen_[0-2000].json"
    ref_path = f"ref_and_gen/CodeLlama-7b-hf/origin/origin/origin/{language}/CSN/work_ref_[0-2000].json"

    with open(ref_path, "r") as f:
        refs = json.load(f)
    with open(gen_path, "r") as f:
        gens = json.load(f)

    gens = FinalDataProcess.handle_generations(language, gens)
    # gens = handle_java_gen(gens)

    score_bert_path = (
        "scores/results/CodeLlama-7b-hf/work_origin_origin_2000_BERTScore.json"
    )
    score_bleu_path = (
        "scores/results/CodeLlama-7b-hf/work_origin_origin_2000_BLEUScore.json"
    )

    with open(score_bert_path, "r") as f:
        score_bert = json.load(f)
    with open(score_bleu_path, "r") as f:
        score_bleu = json.load(f)

    prompt_list = []
    for idx in range(2000):
        sample = dataset[idx]
        code_string = sample["code"]
        doc_string = sample["docstring"]

        prompt = code_string.replace(doc_string, "", 1)
        prompt_list.append(prompt)

    token_num_per_line = get_code_token_num_per_line(prompt_list)
    identifier_ratio = get_identifier_ratio(prompt_list)

    get_char_num_per_line(prompt_list)

    bind_data = []
    for i in range(2000):
        cur_data = {
            "prompt": prompt_list[i],
            "gen": gens[i],
            "ref": refs[i],
            "score_bert": score_bert[i],
            "score_bleu": score_bleu[i],
            "token_num_per_line": token_num_per_line[i],
        }
        bind_data.append(cur_data)

    bert_score = [data["score_bert"] for data in bind_data]
    bleu_score = [data["score_bleu"] for data in bind_data]
    token_num_per_line = [data["token_num_per_line"] for data in bind_data]

    # use corrcoef
    corrcoef_bert = np.corrcoef(bert_score, token_num_per_line)
    corrcoef_bleu = np.corrcoef(bleu_score, token_num_per_line)
    print(f"corrcoef_bert:{corrcoef_bert}")
    print(f"corrcoef_bleu:{corrcoef_bleu}")


if __name__ == "__main__":
    data_proccess()
