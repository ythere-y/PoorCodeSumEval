# 统一数据路径
import os
from colorama import Fore


class FinalPath:
    PATH_NAME_DICT = {
        "origin": {
            "origin": ["origin"],
        },
        "single": {
            "semantic": ["FNE", "IHR", "IOE", "IRS"],
            "synatic": ["DBI", "HVI", "OOS"],
        },
        "cross": {
            "FNE": ["DBI", "HVI", "OOS"],
            "IHR": ["DBI", "HVI", "OOS"],
            "IOE": ["DBI", "HVI", "OOS"],
            "IRS": ["DBI", "HVI", "OOS"],
        },
    }
    LANG_LIST = ["java", "go", "python"]
    MODEL_LIST = ["CodeLlama-7b-hf", "CodeT5", "CodeBERT", "PLBart"]
    SCORE_NAME_LIST = ["BLEUScore", "BERTScore"]

    def check_file_exist(
        part_name,
        model_name=None,
        task_name=None,
        start_point=None,
        limit=None,
        set_lang=None,
    ):
        """
        检查路径下是否存在指定对象
        """
        if part_name == "data":
            fine_flag = True
            print(f"{Fore.BLUE}********** check data path **********{Fore.RESET}")
            for partition in FinalPath.PATH_NAME_DICT:
                for type_name in FinalPath.PATH_NAME_DICT[partition]:
                    for mode_name in FinalPath.PATH_NAME_DICT[partition][type_name]:
                        for lang_name in FinalPath.LANG_LIST:
                            path = FinalPath.get_rob_data_path(
                                partition,
                                type_name,
                                mode_name,
                                lang_name,
                            )
                            # 检查该目录下有没有数据文件
                            data_file = f"{path}/dataset_info.json"
                            if not os.path.exists(data_file):
                                print(f'path "{path}" not exist xxxxxxxxxxxxxxxxxxxx')
                                fine_flag = False
            if not fine_flag:
                print(
                    f"{Fore.RED}***************** data path loss *****************{Fore.RESET}"
                )
            else:
                print(
                    f"{Fore.GREEN}***************** data path fine *****************{Fore.RESET}"
                )
        elif part_name == "generations":
            fine_flag = True
            print(
                f"{Fore.BLUE}********** check generations for {model_name} path **********{Fore.RESET}"
            )
            # 检查参数
            if model_name is None:
                print(
                    f"{Fore.RED}***************** model_name is None *****************{Fore.RESET}"
                )
                return
            if task_name is None:
                print(
                    f"{Fore.RED}***************** task_name is None *****************{Fore.RESET}"
                )
                return
            if start_point is None:
                print(
                    f"{Fore.RED}***************** start_point is None *****************{Fore.RESET}"
                )
                return
            if limit is None:
                print(
                    f"{Fore.RED}***************** limit is None *****************{Fore.RESET}"
                )
                return
            for partition in FinalPath.PATH_NAME_DICT:
                for type_name in FinalPath.PATH_NAME_DICT[partition]:
                    for mode_name in FinalPath.PATH_NAME_DICT[partition][type_name]:
                        for lang_name in FinalPath.LANG_LIST:
                            if set_lang != None and lang_name != set_lang:
                                continue
                            gen_path, ref_path = FinalPath.get_generation_path(
                                model_name,
                                partition,
                                type_name,
                                mode_name,
                                lang_name,
                                task_name,
                                start_point,
                                limit,
                            )
                            if not os.path.exists(gen_path):
                                print(
                                    f'path "{gen_path}" not exist xxxxxxxxxxxxxxxxxxxx'
                                )
                                fine_flag = False
                            if not os.path.exists(ref_path):
                                print(
                                    f'path "{ref_path}" not exist xxxxxxxxxxxxxxxxxxxx'
                                )
                                fine_flag = False
            if not fine_flag:
                print(
                    f"{Fore.RED}***************** generations path loss *****************{Fore.RESET}"
                )
            else:
                print(
                    f"{Fore.GREEN}***************** generations path fine *****************{Fore.RESET}"
                )
        elif part_name == "scores":
            fine_flag = True
            print(
                f"{Fore.BLUE}********** check scores for {model_name} path **********{Fore.RESET}"
            )
            # 检查参数
            if model_name is None:
                print(
                    f"{Fore.RED}***************** model_name is None *****************{Fore.RESET}"
                )
                return
            if task_name is None:
                print(
                    f"{Fore.RED}***************** task_name is None *****************{Fore.RESET}"
                )
                return
            if start_point is None:
                print(
                    f"{Fore.RED}***************** start_point is None *****************{Fore.RESET}"
                )
                return
            if limit is None:
                print(
                    f"{Fore.RED}***************** limit is None *****************{Fore.RESET}"
                )
                return
            for partition in FinalPath.PATH_NAME_DICT:
                for type_name in FinalPath.PATH_NAME_DICT[partition]:
                    for mode_name in FinalPath.PATH_NAME_DICT[partition][type_name]:
                        for lang_name in FinalPath.LANG_LIST:
                            if set_lang != None and lang_name != set_lang:
                                continue
                            for score_name in FinalPath.SCORE_NAME_LIST:
                                score_path = FinalPath.get_score_path(
                                    model_name,
                                    partition,
                                    type_name,
                                    mode_name,
                                    lang_name,
                                    task_name,
                                    start_point,
                                    limit,
                                    score_name,
                                )
                                if not os.path.exists(score_path):
                                    print(
                                        f'path "{score_path}" not exist xxxxxxxxxxxxxxxxxxxx'
                                    )
                                    fine_flag = False
            if not fine_flag:
                print(
                    f"{Fore.RED}***************** scores path loss *****************{Fore.RESET}"
                )
            else:
                print(
                    f"{Fore.GREEN}***************** scores path fine *****************{Fore.RESET}"
                )

    def get_rob_data_path(partition_name, type_name, mode_name, lang_name):
        """
        唯一指定的混淆数据源读取方式
        对于元数据，输入 (origin, origin, origin, python)
        对于单一混淆数据, 输入(single, semantic, FNE, python)
        对于交叉混淆数据, 输入(cross, FNE, OOS, python)
        """
        res_dir = f"local_data/second/{partition_name}/{type_name}/{mode_name}/{lang_name}/CSN"
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        return res_dir

    def get_generation_path(
        model_name,
        partition_name,
        type_name,
        mode_name,
        lang_name,
        task_name,
        start_point,
        limit,
    ):
        """
        最终唯一指定的推理生成数据存储路径
        对于元数据，输入 (codellama,origin,origin,origin,python,work_name,0,2000)
        对于单一混淆数据, 输入(codellama,single,semantic,FNE,python,work_name,0,2000)
        对于交叉混淆数据, 输入(codellama,cross,FNE,OOS,python,work_name,0,2000)
        """
        gen_dir = f"ref_and_gen/{model_name}/{partition_name}/{type_name}/{mode_name}/{lang_name}/CSN"
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
        gen_path = f"{gen_dir}/{task_name}_gen_[{start_point}-{limit}].json"
        ref_path = f"{gen_dir}/{task_name}_ref_[{start_point}-{limit}].json"
        return gen_path, ref_path

    def get_score_path(
        model_name,
        partition_name,
        type_name,
        mode_name,
        lang_name,
        task_name,
        start_point,
        limit,
        score_name,
    ):
        """
        最终唯一指定的评分结果存储路径
        对于元数据，输入 (codellama,origin,origin,origin,work_name,python,0,2000,bleu)
        对于单一混淆数据, 输入(codellama,single,semantic,FNE,work_name,python,0,2000,bleu)
        对于交叉混淆数据, 输入(codellama,cross,FNE,OOS,work_name,python,0,2000,bleu)
        """
        score_dir = f"scores/{model_name}/{partition_name}/{type_name}/{mode_name}/{lang_name}/CSN/{score_name}"
        if not os.path.exists(score_dir):
            os.makedirs(score_dir)
        score_path = f"{score_dir}/{task_name}_[{start_point}-{limit}].json"
        return score_path


def get_input_data_path(type_name, mode_name, lang_name):
    return f"local_data/second/{type_name}/{mode_name}/{lang_name}/CSN"


def get_input_data_path_cross(mode_1_name, mode_2_name, lang_name):
    return f"local_data/second/cross/{mode_1_name}/{mode_2_name}/{lang_name}/CSN"


def get_generation_path(
    model_name, type_name, mode_name, lang_name, task_name, start_point, limit
):
    dir_path = f"ref_and_gen/{model_name}/{type_name}/{mode_name}/{lang_name}/CSN"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    ref_path = f"{dir_path}/{task_name}_ref_[{start_point}-{limit}].json"
    gen_path = f"{dir_path}/{task_name}_gen_[{start_point}-{limit}].json"
    return ref_path, gen_path


def get_scores_path(
    model_name, type_name, mode_name, lang_name, start_point, limit, score_name
):
    dir_path = (
        f"scores/{model_name}/{type_name}/{mode_name}/{lang_name}/CSN/{score_name}"
    )
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    score_path = f"{dir_path}/score_[{start_point}-{limit}].json"
    return score_path


if __name__ == "__main__":
    # FinalPath.check_file_exist("data")
    # FinalPath.check_file_exist(
    #     "generations", "CodeLlama-7b-hf", "work", 0, 2000, set_lang="python"
    # )
    # FinalPath.check_file_exist(
    #     "scores", "CodeLlama-7b-hf", "work", 0, 2000, set_lang="python"
    # )

    FinalPath.check_file_exist(
        "generations", "CodeLlama-7b-hf", "work", 0, 2000, set_lang="go"
    )
