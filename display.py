import json
import os
from colorama import Fore
import datasets
from colorama import Fore
from tree_sitter import Language


def display_gen(generation):
    # print(generation)
    # print(len(generation))
    for idx, gen in enumerate(generation):
        # print(generation[0])
        print(f"{Fore.GREEN}[{idx}]{Fore.RESET}")
        print(gen)
        if idx > 2:
            break


def show_human_eval():
    with open(
        "bigcode-evaluation-harness/generations_humanevalsynthesize-python.json",
        "r",
    ) as f:
        generations = json.load(f)
    # print(generations)
    # print(len(generations))
    # print(generations[0])
    hero_numbers = [40]
    display_gen(generations[96])


def show_local_data():
    type_name = "semantic"
    partition_name = "IOE"
    # Language_name = "go"
    Language_name = "python"
    file_name = (
        f"local_data/first/{type_name}/{partition_name}/{Language_name}/CSN/test.jsonl"
    )
    dataset = datasets.load_dataset("json", data_files=file_name)
    data = dataset["train"]
    print(len(data))
    return
    # print(dataset["train"][0]["code"])
    print(dataset["train"][1]["code"])
    print(data.shape)
    left_limit, right_limit = 0, 10
    for i in range(left_limit, right_limit):
        current_data = data[i]
        print(f"{Fore.GREEN}[{i}]code :{Fore.RESET}")
        print(f'{current_data["code"]}')
        print(f"{Fore.RED}**********{Fore.RESET}")

    pass


if __name__ == "__main__":
    # show_human_eval()
    show_local_data()
