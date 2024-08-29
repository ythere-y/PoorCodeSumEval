# check your installation
import os
from bert_score import BERTScorer
from colorama import Fore

import json

import torch
from tasks.data_utils import (
    FinalDataProcess,
)
from tasks.path_utils import FinalPath
import time
from scores.summary_utils import create_summary, reset_summary, update_summary

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)


def setup():
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    # scorer = BERTScorer(lang="en")
    return scorer


scorer = setup()


def single_BERTScore(
    fake=False,
    model_name="CodeLlama-7b-hf",
    partition_name="cross",
    type_name="FNE",
    mode_name="OOS",
    lang_name="python",
    task_name="work",
    start_point=0,
    limit=2000,
    score_name="BERTScore",
):
    gen_path, ref_path = FinalPath.get_generation_path(
        model_name,
        partition_name,
        type_name=type_name,
        mode_name=mode_name,
        lang_name=lang_name,
        task_name=task_name,
        start_point=start_point,
        limit=limit,
    )
    with open(ref_path, "r") as f:
        # print(f"{Fore.GREEN}Loading refer from {ref_path}{Fore.RESET}")
        refs = json.load(f)
    with open(gen_path, "r") as f:
        # print(f"{Fore.GREEN}Loading generation from {gen_path}{Fore.RESET}")
        gens = json.load(f)
    gens = FinalDataProcess.handle_generations(lang_name, gens)

    filtered_gens, filtered_refs = [], []
    # 删除空的
    for gen, ref in zip(gens, refs):
        if len(gen.strip()) != 0:
            filtered_gens.append(gen)
            filtered_refs.append(ref)
    gens, refs = filtered_gens, filtered_refs
    if fake:
        F1 = 0.5
        F1_list = [F1] * limit
    else:
        P, R, F1 = scorer.score(gens, refs)
        F1_list = F1.tolist()
        # 让所有元素乘以100
        F1_list = [i * 100 for i in F1_list]
        F1 = F1.mean().item() * 100

    score_path = FinalPath.get_score_path(
        model_name,
        partition_name,
        type_name,
        mode_name,
        lang_name,
        task_name,
        start_point,
        limit,
        score_name,
    )
    with open(score_path, "w") as f:
        json.dump(F1_list, f, indent=4)

    print(
        f"{model_name}_{partition_name}_{type_name}_{mode_name}_{task_name}_[{start_point}-{limit}] {score_name}  : {F1}"
    )

    update_summary(
        model_name,
        partition_name,
        type_name,
        mode_name,
        lang_name,
        task_name,
        start_point,
        limit,
        score_name,
        F1,
    )


def single_BLEUScore(
    fake=False,
    model_name="CodeLlama-7b-hf",
    partition_name="cross",
    type_name="FNE",
    mode_name="OOS",
    lang_name="python",
    task_name="work",
    start_point=0,
    limit=2000,
    score_name="BLEUScore",
):
    gen_path, ref_path = FinalPath.get_generation_path(
        model_name,
        partition_name,
        type_name=type_name,
        mode_name=mode_name,
        lang_name=lang_name,
        task_name=task_name,
        start_point=start_point,
        limit=limit,
    )
    with open(ref_path, "r") as f:
        # print(f"{Fore.GREEN}Loading refer from {ref_path}{Fore.RESET}")
        refs = json.load(f)
    with open(gen_path, "r") as f:
        # print(f"{Fore.GREEN}Loading generation from {gen_path}{Fore.RESET}")
        gens = json.load(f)
    gens = FinalDataProcess.handle_generations(lang_name, gens)

    if fake:
        bleu = 0.5
        bleu_list = [bleu] * limit
    else:
        from scores.bleu import bleuForList

        bleu, bleu_list = bleuForList(gens, refs)

    score_path = FinalPath.get_score_path(
        model_name,
        partition_name,
        type_name,
        mode_name,
        lang_name,
        task_name,
        start_point,
        limit,
        score_name,
    )
    with open(score_path, "w") as f:
        json.dump(bleu_list, f, indent=4)

    print(
        f"{model_name}_{partition_name}_{type_name}_{mode_name}_{task_name}_[{start_point}-{limit}] {score_name}  : {bleu}"
    )

    update_summary(
        model_name,
        partition_name,
        type_name,
        mode_name,
        lang_name,
        task_name,
        start_point,
        limit,
        score_name,
        bleu,
    )


def AllBERTScore(model_name, lang_name, task_name, start_point, limit):
    score_name = "BERTScore"
    for partition in FinalPath.PATH_NAME_DICT:
        for type_name in FinalPath.PATH_NAME_DICT[partition]:
            for mode_name in FinalPath.PATH_NAME_DICT[partition][type_name]:
                single_BERTScore(
                    fake=False,
                    model_name=model_name,
                    partition_name=partition,
                    type_name=type_name,
                    mode_name=mode_name,
                    lang_name=lang_name,
                    task_name=task_name,
                    start_point=start_point,
                    limit=limit,
                    score_name="BERTScore",
                )


def AllBLEUScore(model_name, lang_name, task_name, start_point, limit):
    score_name = "BLEUScore"
    for partition in FinalPath.PATH_NAME_DICT:
        for type_name in FinalPath.PATH_NAME_DICT[partition]:
            for mode_name in FinalPath.PATH_NAME_DICT[partition][type_name]:
                single_BLEUScore(
                    fake=False,
                    model_name=model_name,
                    partition_name=partition,
                    type_name=type_name,
                    mode_name=mode_name,
                    lang_name=lang_name,
                    task_name=task_name,
                    start_point=start_point,
                    limit=limit,
                    score_name=score_name,
                )


if __name__ == "__main__":
    t0 = time.time()
    # single_BLEUScore(
    #     fake=False,
    #     model_name="CodeLlama-7b-hf",
    #     partition_name="origin",
    #     type_name="origin",
    #     mode_name="origin",
    #     lang_name="python",
    #     task_name="work",
    #     start_point=0,
    #     limit=2000,
    #     score_name="BLEUScore",
    # )
    # reset_summary("CodeLlama-7b-hf")
    # model_name = "previous/PLBart"
    # lang_name = "go"
    # for model_name in ["previous/PLBart", "previous/CodeT5"]:
    for model_name in ["previous/PLBart"]:
        # for model_name in ["previous/CodeT5"]:
        create_summary(model_name)
        reset_summary(model_name)
        for lang_name in ["go", "java", "python"]:
            task_name = "work"
            start_point = 0
            limit = -1
            print(f"start scoring model = {model_name}, lang = {lang_name}")
            AllBLEUScore(model_name, lang_name, task_name, start_point, limit)
            AllBERTScore(model_name, lang_name, task_name, start_point, limit)
    t1 = time.time()

    print("**************************************")
    print("**************************************")
    print(f"**********{Fore.GREEN}All Done!{Fore.RESET}**********")
    print("**************************************")
    print("**************************************")
    print(f"{Fore.RED}cost time: {t1-t0}{Fore.RESET}")
