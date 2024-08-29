import os
import json
import time

from tasks.path_utils import FinalPath


def create_summary(model_name):
    summary_path = f"scores/{model_name}/summary.json"
    if os.path.exists(summary_path):
        return
    summary = {"best": {}, "history": {}}
    # for score_name in [FinalPath.SCORE_NAME_LIST]:
    for partition in FinalPath.PATH_NAME_DICT:
        for type_name in FinalPath.PATH_NAME_DICT[partition]:
            for mode_name in FinalPath.PATH_NAME_DICT[partition][type_name]:
                for lang_name in FinalPath.LANG_LIST:
                    key = f"{lang_name}_{partition}_{type_name}_{mode_name}"
                    summary["best"][key] = {}
                    summary["history"][key] = {}
                    summary["history"][key]["BLEUScore"] = {}
                    summary["history"][key]["BERTScore"] = {}
                    for score_name in FinalPath.SCORE_NAME_LIST:
                        summary["best"][key][score_name] = 0.0

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)


def reset_summary(model_name):
    # 删除summary文件
    summary_path = f"scores/{model_name}/summary.json"
    if os.path.exists(summary_path):
        os.remove(summary_path)
    create_summary(model_name)


def reset_summary_partly(model_name, score_name):
    pass


def update_summary(
    model_name,
    partition_name,
    type_name,
    mode_name,
    lang_name,
    task_name,
    start_point,
    limit,
    score_name,
    new_score,
):
    create_summary(model_name)

    summary_path = f"scores/{model_name}/summary.json"

    with open(summary_path, "r") as f:
        summary = json.load(f)
    key = f"{lang_name}_{partition_name}_{type_name}_{mode_name}"

    # try to update best score
    history_dict = summary["history"][key][score_name]
    score_list = [info["score"] for info in history_dict.values()]
    if len(score_list) == 0 or new_score > max(score_list):
        summary["best"][key][score_name] = new_score

    # update history
    cur_time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    info = {"score": new_score, "info": f"{task_name}_[{start_point}-{limit}]"}
    summary["history"][key][cur_time_stamp] = info

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
