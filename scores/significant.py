# check your installation
import os
from bert_score import BERTScorer
from colorama import Fore
import sys, time

sys.path.append(
    "/data2/huchao/archived_11.2024_1_confused_novel/bigcode-evaluation-harness"
)
import json
from scipy.stats import f_oneway, ttest_ind
from tasks.path_utils import FinalPath


def significant_analysis(data1, data2):
    statistic, pvalue = f_oneway(data1, data2)
    return statistic, pvalue


def ALLSignificant(model_name, lang_name, task_name, start_point, score_name, limit):
    semantic_bases = {}
    for partition in FinalPath.PATH_NAME_DICT:
        for type_name in FinalPath.PATH_NAME_DICT[partition]:
            cross_list = []
            for mode_name in FinalPath.PATH_NAME_DICT[partition][type_name]:
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
                print("*********************gap*********************")
                print(f"{Fore.GREEN} loading data from {score_path}{Fore.RESET}")
                with open(score_path, "r") as f:
                    data = json.load(f)
                # continue
                if partition == "origin":
                    semantic_bases["origin"] = data
                    continue
                elif partition == "single":
                    semantic_bases[mode_name] = data
                    compare_name = "origin"
                    compare_data = semantic_bases[compare_name]
                elif partition == "cross":
                    compare_name = type_name
                    compare_data = semantic_bases[compare_name]
                    cross_list.append(data)
                print(
                    f"compare {Fore.RED}{partition}-{type_name}-{mode_name}{Fore.RESET} with {compare_name}"
                )
                average = sum(data) / len(data)
                compare_average = sum(compare_data) / len(compare_data)
                if average < 1.0:
                    print(
                        f"current average: {round(average*100,2)}, compare average: {round(compare_average*100,2)}"
                    )
                else:
                    print(
                        f"current average: {round(average,2)}, compare average: {round(compare_average,2)}"
                    )

                statistic, pvalue = significant_analysis(data, compare_data)
                print(
                    f"statistic: {statistic}, {Fore.BLUE}pvalue: {pvalue:.4f}{Fore.RESET}"
                )
            if partition == "cross":
                continue
                print("*********************gap*********************")
                print(
                    f"compare average data of {Fore.YELLOW}cross{Fore.RED} of {type_name}*3{Fore.RESET} with {compare_name}"
                )
                # 计算3个list得到一个平均值list
                average_list = []
                for i in range(len(cross_list[0])):
                    average_list.append(
                        (cross_list[0][i] + cross_list[1][i] + cross_list[2][i]) / 3
                    )
                current_average = sum(average_list) / len(average_list)
                compare_average = sum(compare_data) / len(compare_data)
                if average < 1.0:
                    print(
                        f"current average: {round(current_average*100,2)}, compare average: {round(compare_average*100,2)}"
                    )
                else:
                    print(
                        f"current average: {round(current_average,2)}, compare average: {round(compare_average,2)}"
                    )
                statistic, pvalue = significant_analysis(average_list, compare_data)
                print(
                    f"statistic: {statistic}, {Fore.BLUE}pvalue: {pvalue:.4f}{Fore.RESET}"
                )

    pass


def generate_table(args):
    pattern_str = """
\begin{table*}[htbp]  
\centering
%\small
\caption{Evaluation results on the obfuscated datasets (BL=BLEU, BS=BERTScore).}
\setlength{\tabcolsep}{0.6mm}{
\begin{tabular}{l cccccc}  
\toprule

\multirow{3}{*}{\bf Dataset} 
& \multicolumn{6}{c}{\bf rp_model_name}\\
        
& \multicolumn{2}{c}{\bf Python} & \multicolumn{2}{c}{\bf Go} & \multicolumn{2}{c}{\bf Java} \\

& BL & BS & BL & BS& BL & BS \\


\hline
\bf Primary &&&\\

\;\;&rp_pr_python_bl& rp_pr_python_bs &rp_pr_go_bl& rp_pr_go_bs & rp_pr_java_bl& rp_pr_java_bs 
\\

\midrule    
\multicolumn{2}{l}{\bf Semantic Perturb.} &&\\

\;\;{IOE}
&rp_ioe_python_bl(rp_p_ioe_python_bl)&pr_ioe_python_bs(rp_p_ioe_python_bs) & rp_ioe_go_bs(rp_p_ioe_) (0.0000)& \bf 16.26 &13.85&21.53 
\\
\;\;{IS}
&14.70(0.0000)&22.82 &13.07(0.0000)&23.28 &15.42&25.50 
\\
\;\;{IHR}
&\bf 12.54 (0.0000)&\bf 17.53&12.10(0.0000)&22.38 &\bf 12.84&\bf 19.61 
\\
\;\;{FNE}
&14.74(0.0000)& 22.63 &12.95(0.0000)& 21.48 &15.40& 25.17 
\\
        
\midrule
\multicolumn{2}{l}{\bf Syntactic Perturb.}&&\\

\;\;{OOS}
&17.94(0.9632)& 29.63 &17.79& 40.14 &18.61& 31.90 
\\
\;\;{HVI}
&17.53(0.0133)& 28.69 &17.75& 40.08 &18.15& 30.99 
\\
\;\;{DBI}
&17.34(0.0003)& 28.27 &17.87(0.7555)& 40.40 &18.26& 31.41 
\\

    \bottomrule
 \end{tabular}
}
\end{table*}
"""
    prefix_string = """
\\begin{table*}[htbp]  
\centering
\caption{Evaluation results on the obfuscated datasets (BL=BLEU, BS=BERTScore).}
\setlength{\\tabcolsep}{0.6mm}{
\\begin{tabular}{l cccccc}  
\\toprule

\multirow{3}{*}{\bf Dataset} 
& \multicolumn{6}{c}{\\bf rp_model_name}\\
        
& \multicolumn{2}{c}{\\bf Python} & \multicolumn{2}{c}{\\bf Go} & \multicolumn{2}{c}{\\bf Java} \\

& BL & BS & BL & BS& BL & BS \\


\hline
\\bf Primary &&&\\

\;\;&rp_pr_python_bl& rp_pr_python_bs &rp_pr_go_bl& rp_pr_go_bs & rp_pr_java_bl& rp_pr_java_bs 
\\
"""
    tail_string = """
    \bottomrule
\end{tabular}
}
\end{table*}
    """
    add_string = ""
    for mode_name in ["IOE", "IS", "IHR", "FNE"]:
        # add_string = "\;\;{IS}"
        add_string += "\;\;{" + mode_name + "}\n"
        for lang_name in ["python", "go", "java"]:
            for score_name in ["bl", "bs"]:
                add_string += f"&rp_{mode_name}_{lang_name}_{score_name}(rp_p_{mode_name}_{lang_name}_{score_name})"
            add_string += "\\"
        prefix_string += add_string
        # print(add_string)
        add_string = ""
    prefix_string += """
\midrule
\multicolumn{2}{l}{\bf Syntactic Perturb.}&&\\
"""
    for model_name in ["OOS", "HVI", "DBI"]:
        add_string += "\;\;{" + model_name + "}\n"
        for lang_name in ["python", "go", "java"]:
            for score_name in ["bl", "bs"]:
                add_string += f"&rp_{model_name}_{lang_name}_{score_name}(rp_p_{model_name}_{lang_name}_{score_name})"
            add_string += "\\"
        prefix_string += add_string
        # print(add_string)
        add_string = ""
    prefix_string += tail_string
    print(prefix_string)


def analysis_and_log():
    t0 = time.time()

    model_name = "CodeBERT"
    # model_name = "CodeT5"
    # model_name = "CodeLlama-7b-hf"
    # lang_name = "go"
    task_name = "work"
    # score_name = "BLEUScore"
    score_name = "BERTScore"
    start_point = 0
    limit = 2000
    for lang_name in ["python", "go", "java"]:
        # for lang_name in ["java"]:
        print(
            f"{Fore.MAGENTA}start Analysising model : {model_name}, lang : {lang_name}{Fore.RESET}"
        )

        ALLSignificant(model_name, lang_name, task_name, start_point, score_name, limit)
        print("******************************")
        print("******************************")
        print("******************************")
        print("******************************")
        # AllBLEUScore(model_name, lang_name, task_name, start_point, limit)
        # AllBERTScore(model_name, lang_name, task_name, start_point, limit)

    t1 = time.time()

    print("**************************************")
    print("**************************************")
    print(f"**********{Fore.GREEN}All Done!{Fore.RESET}**********")
    print("**************************************")
    print("**************************************")
    print(f"{Fore.RED}cost time: {t1-t0}{Fore.RESET}")


if __name__ == "__main__":

    # generate_table(None)
    analysis_and_log()
