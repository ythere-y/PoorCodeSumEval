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


def get_other_model_gen_and_ref_path(
    model_name,
    partition_name,
    type_name,
    mode_name,
    lang_name,
):
    result_dir = "/data2/huchao/11.novels/models/results"
    gen_name = "test_0.output"
    ref_name = "test_0.gold"
    data_dir = f"{result_dir}/{model_name}/{partition_name}/{type_name}/{mode_name}/{lang_name}"
    ref_path = f"{data_dir}/{ref_name}"
    gen_path = f"{data_dir}/{gen_name}"
    return gen_path, ref_path


def read_others_data(gen_path, ref_path):
    with open(ref_path, "r") as f:
        lines = f.readlines()
    refs = [line.split("\t") for line in lines]
    refs = [(int(ref[0]), ref[1].strip()) for ref in refs]
    with open(gen_path, "r") as f:
        lines = f.readlines()
    cands = [line.split("\t") for line in lines]
    cands = [(int(cand[0]), cand[1].strip()) for cand in cands]
    # 检查编号是否对齐
    assert len(refs) == len(cands)
    for ref, cand in zip(refs, cands):
        assert ref[0] == cand[0]
    cands = [cand[1] for cand in cands]
    refs = [ref[1] for ref in refs]
    return refs, cands


def single_BERTScore_others(
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
    gen_path, ref_path = get_other_model_gen_and_ref_path(
        model_name,
        partition_name,
        type_name,
        mode_name,
        lang_name,
    )
    refs, gens = read_others_data(gen_path, ref_path)
    limit = len(refs)
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


def single_BLEUScore_others(
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
    gen_path, ref_path = get_other_model_gen_and_ref_path(
        model_name,
        partition_name,
        type_name,
        mode_name,
        lang_name,
    )
    refs, gens = read_others_data(gen_path, ref_path)
    limit = len(refs)
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


def AllBERTScoreOthers(model_name, lang_name):
    score_name = "BERTScore"
    for partition in FinalPath.PATH_NAME_DICT:
        for type_name in FinalPath.PATH_NAME_DICT[partition]:
            for mode_name in FinalPath.PATH_NAME_DICT[partition][type_name]:
                single_BERTScore_others(
                    fake=False,
                    model_name=model_name,
                    partition_name=partition,
                    type_name=type_name,
                    mode_name=mode_name,
                    lang_name=lang_name,
                    score_name=score_name,
                )


def AllBLEUScoreOthers(model_name, lang_name):
    score_name = "BLEUScore"
    for partition in FinalPath.PATH_NAME_DICT:
        for type_name in FinalPath.PATH_NAME_DICT[partition]:
            for mode_name in FinalPath.PATH_NAME_DICT[partition][type_name]:
                single_BLEUScore_others(
                    fake=False,
                    model_name=model_name,
                    partition_name=partition,
                    type_name=type_name,
                    mode_name=mode_name,
                    lang_name=lang_name,
                    score_name=score_name,
                )


if __name__ == "__main__":
    t0 = time.time()
    model_name = "CodeBERT"
    lang_name = "java"
    partition = "origin"
    type_name = "origin"
    mode_name = "origin"
    print(f"start scoring model = {model_name}, lang = {lang_name}")
    # reset_summary(model_name)
    # single_BLEUScore_others(
    #     fake=False,
    #     model_name=model_name,
    #     partition_name=partition,
    #     type_name=type_name,
    #     mode_name=mode_name,
    #     lang_name=lang_name,
    #     score_name=score_name,
    # )
    AllBLEUScoreOthers(model_name, lang_name)
    AllBERTScoreOthers(model_name, lang_name)
    # lang_name = "go"
    # print(f"start scoring model = {model_name}, lang = {lang_name}")
    # AllBLEUScoreOthers(model_name, lang_name)
    # AllBERTScoreOthers(model_name, lang_name)
    t1 = time.time()

    print("**************************************")
    print("**************************************")
    print(f"**********{Fore.GREEN}All Done!{Fore.RESET}**********")
    print("**************************************")
    print("**************************************")
    print(f"{Fore.RED}cost time: {t1-t0}{Fore.RESET}")
