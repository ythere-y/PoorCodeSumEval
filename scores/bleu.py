import sys, math, re, xml.sax.saxutils

sys.path.append(
    "/data2/huchao/archived_11.2024_1_confused_novel/bigcode-evaluation-harness"
)
import os
from colorama import Fore
import json

from tasks.data_utils import (
    get_base_references,
    handle_reference,
    get_references_from_tokens_from_first,
    get_references_from_string,
    get_references_from_tokens,
    trans_to_list_list,
    get_generations,
    data_name_dict,
)

# Added to bypass NIST-style pre-processing of hyp and ref files -- wade
nonorm = 0

preserve_case = False
eff_ref_len = "shortest"

normalize1 = [
    ("<skipped>", ""),  # strip "skipped" tags
    (r"-\n", ""),  # strip end-of-line hyphenation and join lines
    (r"\n", " "),  # join lines
    #    (r'(\d)\s+(?=\d)', r'\1'), # join digits
]
normalize1 = [(re.compile(pattern), replace) for (pattern, replace) in normalize1]

normalize2 = [
    (
        r"([\{-\~\[-\` -\&\(-\+\:-\@\/])",
        r" \1 ",
    ),  # tokenize punctuation. apostrophe is missing
    (
        r"([^0-9])([\.,])",
        r"\1 \2 ",
    ),  # tokenize period and comma unless preceded by a digit
    (
        r"([\.,])([^0-9])",
        r" \1 \2",
    ),  # tokenize period and comma unless followed by a digit
    (r"([0-9])(-)", r"\1 \2 "),  # tokenize dash when preceded by a digit
]
normalize2 = [(re.compile(pattern), replace) for (pattern, replace) in normalize2]


def normalize(s):
    """Normalize and tokenize text. This is lifted from NIST mteval-v11a.pl."""
    # Added to bypass NIST-style pre-processing of hyp and ref files -- wade
    if nonorm:
        return s.split()
    if type(s) is not str:
        s = " ".join(s)
    # language-independent part:
    for pattern, replace in normalize1:
        s = re.sub(pattern, replace, s)
    s = xml.sax.saxutils.unescape(s, {"&quot;": '"'})
    # language-dependent part (assuming Western languages):
    s = " %s " % s
    if not preserve_case:
        s = s.lower()  # this might not be identical to the original
    for pattern, replace in normalize2:
        s = re.sub(pattern, replace, s)
    return s.split()


def count_ngrams(words, n=4):
    counts = {}
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i : i + k])
            counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def cook_refs(refs, n=4):
    """Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them."""

    refs = [normalize(ref) for ref in refs]
    maxcounts = {}
    for ref in refs:
        counts = count_ngrams(ref, n)
        for ngram, count in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)
    return ([len(ref) for ref in refs], maxcounts)


def cook_test(test, item, n=4):
    """Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it."""
    (reflens, refmaxcounts) = item
    test = normalize(test)
    result = {}
    result["testlen"] = len(test)

    # Calculate effective reference sentence length.

    if eff_ref_len == "shortest":
        result["reflen"] = min(reflens)
    elif eff_ref_len == "average":
        result["reflen"] = float(sum(reflens)) / len(reflens)
    elif eff_ref_len == "closest":
        min_diff = None
        for reflen in reflens:
            if min_diff is None or abs(reflen - len(test)) < min_diff:
                min_diff = abs(reflen - len(test))
                result["reflen"] = reflen

    result["guess"] = [max(len(test) - k + 1, 0) for k in range(1, n + 1)]

    result["correct"] = [0] * n
    counts = count_ngrams(test, n)
    for ngram, count in counts.items():
        result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)

    return result


def score_cooked(allcomps, n=4, ground=0, smooth=1):
    totalcomps = {"testlen": 0, "reflen": 0, "guess": [0] * n, "correct": [0] * n}
    for comps in allcomps:
        for key in ["testlen", "reflen"]:
            totalcomps[key] += comps[key]
        for key in ["guess", "correct"]:
            for k in range(n):
                totalcomps[key][k] += comps[key][k]
    logbleu = 0.0
    all_bleus = []
    for k in range(n):
        correct = totalcomps["correct"][k]
        guess = totalcomps["guess"][k]
        addsmooth = 0
        if smooth == 1 and k > 0:
            addsmooth = 1
        logbleu += math.log(correct + addsmooth + sys.float_info.min) - math.log(
            guess + addsmooth + sys.float_info.min
        )
        if guess == 0:
            all_bleus.append(-10000000)
        else:
            all_bleus.append(math.log(correct + sys.float_info.min) - math.log(guess))

    logbleu /= float(n)
    all_bleus.insert(0, logbleu)

    brevPenalty = min(
        0, 1 - float(totalcomps["reflen"] + 1) / (totalcomps["testlen"] + 1)
    )
    for i in range(len(all_bleus)):
        if i == 0:
            all_bleus[i] += brevPenalty
        all_bleus[i] = math.exp(all_bleus[i])
    return all_bleus


def bleu(refs, candidate, ground=0, smooth=1):
    refs = cook_refs(refs)
    test = cook_test(candidate, refs)
    return score_cooked([test], ground=ground, smooth=smooth)


def splitPuncts(line):
    return " ".join(re.findall(r"[\w]+|[^\s\w]", line))


def computeMaps(predictions, goldfile):
    predictionMap = {}
    goldMap = {}
    gf = open(goldfile, "r")

    for row in predictions:
        cols = row.strip().split("\t")
        if len(cols) == 1:
            (rid, pred) = (cols[0], "")
        else:
            (rid, pred) = (cols[0], cols[1])
        predictionMap[rid] = [splitPuncts(pred.strip().lower())]

    for row in gf:
        (rid, pred) = row.split("\t")
        if rid in predictionMap:  # Only insert if the id exists for the method
            if rid not in goldMap:
                goldMap[rid] = []
            goldMap[rid].append(splitPuncts(pred.strip().lower()))

    sys.stderr.write("Total: " + str(len(goldMap)) + "\n")
    return (goldMap, predictionMap)


def bleuFromMaps(m1, m2):
    score = 0
    num = 0.0
    bleu_list = []
    for s1, s2 in zip(m1, m2):
        # print(s1)
        # print(s2)
        bl = bleu(s1, s2)
        bleu_list.append(bl[0])
        score = score + bl[0]
        num += 1
    return score * 100.0 / num, bleu_list


def bleuForList(m1, m2):
    m1 = trans_to_list_list(m1)
    m2 = trans_to_list_list(m2)
    return bleuFromMaps(m2, m1)


def static_test():
    m1 = [
        ["Build a list of commic objects"],
        ["Generates a Cartesia."],
        ["Returns the parametearameters."],
    ]
    m2 = [
        ["Build a list of comm...ic objects"],
        ["Generates a Cartesia...ictionary."],
        ["Returns the paramete...arameters."],
    ]

    print(bleuFromMaps(m1, m2)[0])


def batch_bleu():
    model_name = "CodeLlama-7b-hf"
    task_name = "work"
    start_point = 0
    limit = 2000

    result = {}

    refs_tokens = get_references_from_tokens("semantic", "FNE", limit)
    refs_tokens = trans_to_list_list(refs_tokens)
    origin_fill_file_name = f"ref_and_gen/{model_name}/origin_origin_{limit}_fill.json"
    # origin_fill_file_name = (
    #     f"ref_and_gen/{model_name}/origin_origin_[{start_point}-{limit}]_fill.json"
    # )
    if os.path.exists(origin_fill_file_name):
        with open(origin_fill_file_name, "r") as f:
            origin_fillings = json.load(f)
        print(f"{Fore.BLUE}origin_fillings{Fore.RESET} : {origin_fill_file_name}")
        origin_first_lines = get_generations(origin_fillings)
        origin_first_lines = trans_to_list_list(origin_first_lines)
        origin_bleu = bleuFromMaps(refs_tokens, origin_first_lines)[0]
        result["origin"] = origin_bleu
    for type_name in data_name_dict:
        for mode_name in data_name_dict[type_name]:
            fill_file_name = f"ref_and_gen/{model_name}/{task_name}_{type_name}_{mode_name}_{limit}_fill.json"
            if os.path.exists(fill_file_name):
                with open(fill_file_name, "r") as f:
                    fillings = json.load(f)
                print(f"{Fore.BLUE}fillings{Fore.RESET} : {fill_file_name}")
                first_lines = get_generations(fillings)

                first_lines = trans_to_list_list(first_lines)
                cur_bleu, cur_bleu_list = bleuFromMaps(refs_tokens, first_lines)
                with open(
                    f"scores/results/{model_name}/{task_name}_{type_name}_{mode_name}_{limit}_BLEUScore.json",
                    "w",
                ) as f:
                    json.dump(cur_bleu_list, f, indent=4)
                result[f"{type_name}_{mode_name}"] = cur_bleu
    # display all results

    for key in result:
        print(f"{Fore.BLUE}{key}{Fore.RESET} : {result[key]}")

    # save result
    # save_dir = f"scores/results/bleu_{limit}.json"
    # with open(save_dir, "w") as f:
    # json.dump(result, f, indent=4)


def single_bleu():
    model_name = "CodeLlama-7b-hf"
    task_name = ""
    start_point = 0
    limit = 14918

    result = {}

    # refs_tokens = get_references_from_tokens("semantic", "FNE", limit)
    refs_tokens = get_references_from_tokens_from_first(limit)
    refs_tokens = trans_to_list_list(refs_tokens)
    type_name = "origin"
    mode_name = "origin"
    fill_file_name = f"ref_and_gen/{model_name}/{task_name}_{type_name}_{mode_name}_{limit}_fill.json"
    with open(fill_file_name, "r") as f:
        fillings = json.load(f)
    print(f"{Fore.BLUE}fillings{Fore.RESET} : {fill_file_name}")
    first_lines = get_generations(fillings)
    first_lines = trans_to_list_list(first_lines)
    cur_bleu, cur_bleu_list = bleuFromMaps(refs_tokens, first_lines)
    with open(
        f"scores/results/{model_name}/{task_name}_{type_name}_{mode_name}_{limit}_BLEUScore.json",
        "w",
    ) as f:
        json.dump(cur_bleu_list, f, indent=4)
    result[f"{type_name}_{mode_name}"] = cur_bleu

    for key in result:
        print(f"{Fore.BLUE}{key}{Fore.RESET} : {result[key]}")


def get_bleu_test():
    refs_tokens = get_references_from_tokens("semantic", "FNE", 200)
    # refs_string = get_references_from_string("semantic", "FNE", 200)
    # fill_file_name = "ref_and_gen/CodeLlama-7b-hf/batch_test_semantic_FNE_200_fill.json"
    fill_file_name = "ref_and_gen/CodeLlama-7b-hf/origin_origin_200_fill.json"
    with open(fill_file_name, "r") as f:
        fillings = json.load(f)
    first_lines = get_generations(fillings)
    # show_idx = 0
    # print(f"{Fore.GREEN}refs_token{Fore.RESET} : {refs_tokens[show_idx]}")
    # print(f"{Fore.GREEN}refs_string{Fore.RESET}  : {refs_string[show_idx]}")
    # print(f"{Fore.GREEN}fillings{Fore.RESET}  : {fillings[show_idx]}")
    # print(f"{Fore.GREEN}first_lines{Fore.RESET}  : {first_lines[show_idx]}")

    refs_tokens = trans_to_list_list(refs_tokens)
    # refs_string = trans_to_list_list(refs_string)
    # fillings = trans_to_list_list(fillings)
    first_lines = trans_to_list_list(first_lines)

    # name_1 = "doc_string"
    # name_2 = "doc_token_combine"
    # ground_truth = refs_tokens
    # prediction = refs_string
    # bleu_score = bleuFromMaps(ground_truth, prediction)[0]
    # print(
    #     f"cmp : {Fore.BLUE}{name_1}{Fore.RESET} -> {Fore.BLUE}{name_2}{Fore.RESET} , BLEU : {bleu_score}"
    # )

    # name_1 = "filling(no split)"
    # name_2 = "doc_token_combine"
    # ground_truth = refs_tokens
    # prediction = fillings
    # bleu_score = bleuFromMaps(ground_truth, prediction)[0]
    # print(
    #     f"cmp : {Fore.BLUE}{name_1}{Fore.RESET} -> {Fore.BLUE}{name_2}{Fore.RESET} , BLEU : {bleu_score}"
    # )

    name_1 = "first line of filling"
    name_2 = "doc_token_combine"
    ground_truth = refs_tokens
    prediction = first_lines
    bleu_score, bleu_list = bleuFromMaps(ground_truth, prediction)

    print(
        f"cmp : {Fore.BLUE}{name_1}{Fore.RESET} -> {Fore.BLUE}{name_2}{Fore.RESET} , BLEU : {bleu_score}"
    )
    print(bleu_list)


def get_score_path(
    model_name, task_name, mode_1, mode_2, limit, score_name, type_name="single"
):
    path = ""
    if type_name == "single":
        path = f"scores/results/{model_name}/{task_name}_{mode_1}_{mode_2}_{limit}_{score_name}.json"
    elif type_name == "cross":
        path = f"scores/results/{model_name}/{type_name}/{task_name}_{mode_1}_{mode_2}_{limit}_{score_name}.json"
    if path == "":
        raise Exception("wrong type_name")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path))
    return path
