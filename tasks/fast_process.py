import re
import os
import glob
import json


def parse_java_comment_and_code(java_code):
    # 正则表达式来匹配注释块
    comment_pattern = re.compile(r"/\*\*(.*?)\*/", re.DOTALL)
    # 正则表达式来匹配Java注释的第一句话
    first_sentence_pattern = re.compile(r"\*\s+(.*?)(?:\n|$)", re.DOTALL)

    # 匹配所有注释块
    comments = comment_pattern.findall(java_code)

    first_sentences = []
    for comment in comments:
        match = first_sentence_pattern.search(comment)
        if match:
            first_sentences.append(match.group(1).strip())

    # 删除所有注释块
    code_without_comments = comment_pattern.sub("", java_code).strip()

    return first_sentences[0], code_without_comments


def batch_test():
    code_files = glob.glob(
        os.path.join("local_data/one_more/Snippets/with_documents", "**", "*.jsnp"),
        recursive=True,
    )
    totla_res = []
    for file in code_files:
        with open(file, "r") as f:
            java_code = f.read()
        first_sentences, code_without_comments = parse_java_comment_and_code(java_code)
        result = {"comments": first_sentences, "code": code_without_comments}
        totla_res.append(result)
    with open("local_data/one_more/Snippets/filtered.jsonl", "w+") as f:
        for res in totla_res:
            f.write(json.dumps(res) + "\n")


def inference():
    import os
    import tempfile
    from tqdm import tqdm
    from accelerate import Accelerator
    from accelerate.utils import write_basic_config
    from click import prompt
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    import json
    from colorama import Fore, Style

    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.cuda.set_device(device)
    # Tasks for generation test
    # GEN_TASKS = ["humaneval", "mbpp"]
    GEN_TASKS = ["codexglue_code_to_text-python-left"]
    # Tasks for evaluator tests
    EVAL_TASKS = ["humaneval", "mbpp", "pal-gsm8k-greedy"]
    TMPDIR = tempfile.mkdtemp()
    # TEST_MODEL = "hf-internal-testing/tiny-random-gpt2"
    # TEST_MODEL = "bigcode/santacoder"
    TEST_MODEL = "codellama/CodeLlama-7b-hf"
    # TEST_MODEL = "codellama/CodeLlama-7b-Instruct-hf"
    REF_EVAL_SCORES = {
        "humaneval": {"pass@1": 0.25},
        "mbpp": {"pass@1": 0.25},
        "pal-gsm8k-greedy": {"accuracy": 1.0, "num_failed_execution": 0},
    }

    def update_args(args):
        # args.model = "bigcode/santacoder"
        args.model = TEST_MODEL
        # the executed code for the tests is safe (see tests/data/*_eval_gens.json)
        args.allow_code_execution = True
        args.save_generations = False
        args.save_generations_path = ""
        args.save_references = False
        args.save_references_path = ""
        args.metric_output_path = TMPDIR
        args.load_generations_path = None
        args.generation_only = False
        args.check_references = False
        # postprocessing for HumanEval and MBPP makes generations
        # with dummy model not distinctive
        args.postprocess = False
        args.instruction_tokens = None

        args.limit = 2
        args.limit_start = 0
        args.batch_size = 1
        args.max_length_generation = 300
        args.do_sample = False
        args.top_p = 0
        args.n_samples = 1
        args.seed = 0
        args.prompt = None
        args.precision = None
        args.modeltype = None
        args.max_memory_per_gpu = None
        return args

    def setup():
        print(f"{Fore.GREEN}loading model of {TEST_MODEL}... {Style.RESET_ALL}")
        # model = AutoModelForCausalLM.from_pretrained(TEST_MODEL)
        cache_id = "7f22f0a5f7991355a2c3867923359ec4ed0b58bf"
        cache_id = "f02cb64b091c07f4e96b79960fd89caf434578e0"
        model_name = "models--codellama--CodeLlama-7b-Instruct-hf"
        model = AutoModelForCausalLM.from_pretrained(
            f"/data3/.cache/huggingface/hub/{model_name}/snapshots/{cache_id}"
        )

        print(f"{Fore.GREEN}loading tokenizer of {TEST_MODEL}...{Fore.RESET}")
        # tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(
            f"/data3/.cache/huggingface/hub/{model_name}/snapshots/{cache_id}"
        )

        tokenizer.pad_token = tokenizer.eos_token
        configPath = os.path.join(TMPDIR, "default_config.yml")
        write_basic_config(save_location=configPath)
        accelerator = Accelerator()
        model = model.to(accelerator.device)
        return model, tokenizer, accelerator

    model, tokenizer, accelerator = setup()
    print("DONE")
    file_path = "local_data/one_more/Snippets/filtered.jsonl"
    total_data = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            total_data.append(json.loads(line))
    print(len(total_data))

    generations = []
    references = []
    for i in tqdm(range(len(total_data))):
        code_string = total_data[i]["code"]
        system = "Generate docstring for the code in 12 words. "
        user = (
            code_string
            + '\n"""Please fill this sentence "The goal of this function is to " in 12 words: '
        )

        prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user}[/INST]"
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(accelerator.device)
        generated_ids = model.generate(input_ids, max_new_tokens=128)
        filling = tokenizer.batch_decode(
            generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]
        # print(filling)
        generated = filling.replace("The goal of this function is to ", "")
        generations.append(filling)
        reference = total_data[i]["comments"]
        references.append(reference)
    with open("tmp/test_gen_java_one_more.json", "w") as f:
        json.dump(generations, f, indent=4)
        # print(generated)
    with open("tmp/test_ref_java_one_more.json", "w") as f:
        json.dump(references, f, indent=4)


def fast_score():
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

    def trans_to_list_list(input_list):
        output_list_list = []
        for i in input_list:
            output_list_list.append([i])
        return output_list_list

    def bleuForList(m1, m2):
        m1 = trans_to_list_list(m1)
        m2 = trans_to_list_list(m2)
        return bleuFromMaps(m2, m1)

    def bleu(refs, candidate, ground=0, smooth=1):
        refs = cook_refs(refs)
        test = cook_test(candidate, refs)
        return score_cooked([test], ground=ground, smooth=smooth)

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

    gen_path = "tmp/test_gen_java_one_more.json"
    ref_path = "tmp/test_ref_java_one_more.json"
    with open(ref_path, "r") as f:
        refs = json.load(f)
    with open(gen_path, "r") as f:
        gens = json.load(f)
    gens = handle_java_gen(gens)
    print(len(gen_path))

    pass


def bind_data():
    code_files = glob.glob(
        os.path.join("local_data/one_more/Snippets/with_documents", "**", "*.jsnp"),
        recursive=True,
    )
    idx = 0
    ref_path = "tmp/test_ref_java_one_more.json"
    gen_path = "tmp/test_gen_java_one_more.json"
    bleu_score_path = "tmp/test_bleu_java_one_more.json"
    bert_socre_path = "tmp/test_bert_java_one_more.json"

    with open(ref_path, "r") as f:
        refs = json.load(f)

    with open(gen_path, "r") as f:
        gens = json.load(f)
    with open(bleu_score_path, "r") as f:
        bleu_score = json.load(f)
    with open(bert_socre_path, "r") as f:
        bert_score = json.load(f)
    readability_score = read_readability_score()
    total_data = []
    for idx, file in enumerate(code_files):
        with open(file, "r") as f:
            java_code = f.read()
        first_sentences, code_without_comments = parse_java_comment_and_code(java_code)
        file_id = file.split("/")[-1].split(".")[0]
        sp_name = f"Snippet{file_id}"

        cur_data = {
            "file": file,
            "comments": first_sentences,
            "code": code_without_comments,
            "ref": refs[idx],
            "gen": gens[idx],
            "bleu": bleu_score[idx],
            "bert": bert_score[idx],
            "readability": readability_score[sp_name],
        }
        total_data.append(cur_data)
    # save total data into xlsx

    import pandas as pd

    df = pd.DataFrame(total_data)
    df.to_excel("tmp/java_one_more.xlsx", index=False)
    with open("tmp/json_data.json", "w+") as f:
        json.dump(total_data, f, indent=4)


def read_readability_score():
    # data format:
    #     ,Snippet1,Snippet2,Snippet3,Snippet4,Snippet5,Snippet6,Snippet7,Snippet8,Snippet9,Snippet10,Snippet11,Snippet12,Snippet13,Snippet14,Snippet15,Snippet16,Snippet17,Snippet18,Snippet19,Snippet20,Snippet21,Snippet22,Snippet23,Snippet24,Snippet25,Snippet26,Snippet27,Snippet28,Snippet29,Snippet30,Snippet31,Snippet32,Snippet33,Snippet34,Snippet35,Snippet36,Snippet37,Snippet38,Snippet39,Snippet40,Snippet41,Snippet42,Snippet43,Snippet44,Snippet45,Snippet46,Snippet47,Snippet48,Snippet49,Snippet50,Snippet51,Snippet52,Snippet53,Snippet54,Snippet55,Snippet56,Snippet57,Snippet58,Snippet59,Snippet60,Snippet61,Snippet62,Snippet63,Snippet64,Snippet65,Snippet66,Snippet67,Snippet68,Snippet69,Snippet70,Snippet71,Snippet72,Snippet73,Snippet74,Snippet75,Snippet76,Snippet77,Snippet78,Snippet79,Snippet80,Snippet81,Snippet82,Snippet83,Snippet84,Snippet85,Snippet86,Snippet87,Snippet88,Snippet89,Snippet90,Snippet91,Snippet92,Snippet93,Snippet94,Snippet95,Snippet96,Snippet97,Snippet98,Snippet99,Snippet100,Snippet101,Snippet102,Snippet103,Snippet104,Snippet105,Snippet106,Snippet107,Snippet108,Snippet109,Snippet110,Snippet111,Snippet112,Snippet113,Snippet114,Snippet115,Snippet116,Snippet117,Snippet118,Snippet119,Snippet120,Snippet121,Snippet122,Snippet123,Snippet124,Snippet125,Snippet126,Snippet127,Snippet128,Snippet129,Snippet130,Snippet131,Snippet132,Snippet133,Snippet134,Snippet135,Snippet136,Snippet137,Snippet138,Snippet139,Snippet140,Snippet141,Snippet142,Snippet143,Snippet144,Snippet145,Snippet146,Snippet147,Snippet148,Snippet149,Snippet150,Snippet151,Snippet152,Snippet153,Snippet154,Snippet155,Snippet156,Snippet157,Snippet158,Snippet159,Snippet160,Snippet161,Snippet162,Snippet163,Snippet164,Snippet165,Snippet166,Snippet167,Snippet168,Snippet169,Snippet170,Snippet171,Snippet172,Snippet173,Snippet174,Snippet175,Snippet176,Snippet177,Snippet178,Snippet179,Snippet180,Snippet181,Snippet182,Snippet183,Snippet184,Snippet185,Snippet186,Snippet187,Snippet188,Snippet189,Snippet190,Snippet191,Snippet192,Snippet193,Snippet194,Snippet195,Snippet196,Snippet197,Snippet198,Snippet199,Snippet200
    # Evaluator1,5,4,5,5,5,4,2,3,5,3,5,3,3,4,2,4,3,4,5,5,5,4,4,4,5,3,3,2,4,4,3,5,4,3,5,4,4,5,5,4,3,5,3,3,5,2,4,4,3,4,3,3,4,4,5,3,4,4,5,4,2,4,5,5,4,3,5,2,2,2,3,3,4,1,4,4,5,3,4,3,4,4,5,3,5,5,2,5,2,4,4,5,3,3,4,3,3,3,3,3,5,3,5,4,5,4,4,4,4,2,4,4,5,5,3,4,5,4,5,4,4,4,5,3,3,3,4,4,3,3,4,3,5,3,3,3,3,4,4,2,3,3,4,5,3,3,3,4,3,3,3,4,5,3,3,3,3,2,3,4,3,3,2,3,5,5,3,4,3,2,3,3,2,4,4,4,4,3,3,3,4,5,4,4,4,4,5,4,3,4,5,3,4,4,4,2,2,3,3,3
    # Evaluator2,3,3,2,3,3,5,3,2,2,3,3,2,3,2,1,4,3,4,3,4,4,4,3,4,4,4,3,2,3,2,2,3,3,2,4,1,3,4,4,3,3,5,3,4,4,4,3,4,1,3,3,3,4,3,5,4,4,4,4,3,3,3,4,4,5,3,4,3,3,2,4,3,4,3,4,4,4,3,4,4,4,2,3,3,3,4,3,3,4,4,4,4,3,3,4,2,2,2,3,4,2,3,3,3,4,3,4,3,3,4,3,3,4,4,5,3,4,4,4,3,3,3,5,3,3,3,3,3,3,3,3,4,5,5,3,3,3,4,4,3,4,4,4,5,4,3,4,4,3,5,5,4,5,3,3,4,3,3,3,4,4,3,3,3,4,4,3,3,3,3,3,4,3,3,5,4,4,4,3,3,3,3,4,5,4,3,5,3,4,4,5,3,4,4,3,2,3,3,3,4
    # Evaluator3,4,4,2,3,4,3,3,3,3,2,4,2,4,3,3,3,4,4,3,3,4,4,2,3,4,4,4,3,4,4,3,4,3,3,4,2,4,3,4,4,4,4,4,3,4,3,3,4,3,4,3,4,3,4,4,3,4,4,3,3,3,3,3,4,4,4,4,3,2,3,3,2,4,1,3,4,4,4,3,3,4,3,4,3,4,4,4,3,3,4,4,4,3,3,4,4,3,3,3,3,3,3,3,4,4,4,4,4,3,3,3,3,4,4,3,2,5,3,4,3,4,3,4,3,3,4,3,4,4,4,3,4,3,3,3,4,4,4,4,3,4,3,4,5,4,3,4,4,3,5,5,5,4,3,4,4,3,3,4,4,4,3,3,5,4,4,3,4,4,4,4,4,4,4,4,3,2,4,3,2,3,3,4,4,4,4,3,4,4,4,4,3,4,3,4,3,4,4,3,4
    # Evaluator4,5,4,4,5,5,5,4,5,5,5,5,5,5,5,4,5,5,5,5,5,5,5,5,5,5,5,5,5,4,5,5,5,5,3,5,2,4,5,5,5,5,5,4,5,5,5,5,5,4,5,5,5,5,5,5,5,5,5,4,5,5,4,3,3,5,5,5,4,5,4,5,5,5,3,5,5,5,5,5,5,5,5,4,4,5,5,5,5,4,5,4,5,4,3,4,4,4,4,4,4,5,5,5,5,4,5,5,5,5,5,4,5,5,5,5,5,5,4,4,5,4,4,5,5,4,5,4,4,5,4,4,4,5,4,4,5,5,5,5,3,5,4,4,5,4,4,5,4,4,5,5,5,5,4,4,3,3,3,4,5,4,3,3,4,5,4,3,5,5,4,4,2,4,4,5,5,4,4,4,3,3,4,5,5,4,4,5,4,4,5,5,4,5,4,5,3,4,3,3,5
    data_path = "local_data/one_more/scores.csv"
    import pandas as pd

    df = pd.read_csv(data_path)
    # 用同样的做法，给每个snappet计算平均值，返回一个map
    snippet_avg_score = {}
    # 第一列要跳过
    for col in df.columns[1:]:
        snippet_data = df[col]
        snippet_data = snippet_data.dropna()
        avg_score = snippet_data.mean()
        snippet_avg_score[col] = avg_score
    print(snippet_avg_score)
    return snippet_avg_score


def fast_analysis():
    file_path = "tmp/json_data.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    # 首先按照readability score排序
    data = sorted(data, key=lambda x: x["readability"])
    # 按照readability分成两组计算bleu和bert score
    down_data = data[: len(data) // 2]
    up_data = data[len(data) // 2 :]

    down_bleu = [x["bleu"] for x in down_data]
    down_bert = [x["bert"] for x in down_data]
    up_bleu = [x["bleu"] for x in up_data]
    up_bert = [x["bert"] for x in up_data]

    # 计算平均值并展示
    down_bleu_avg = sum(down_bleu) / len(down_bleu)
    down_bert_avg = sum(down_bert) / len(down_bert)
    up_bleu_avg = sum(up_bleu) / len(up_bleu)
    up_bert_avg = sum(up_bert) / len(up_bert)

    down_readability_avg = sum([x["readability"] for x in down_data]) / len(down_data)
    up_readability_avg = sum([x["readability"] for x in up_data]) / len(up_data)
    print(
        f"down ** bleu_avg: {round(down_bleu_avg, 4)}, bert avg: {round(down_bert_avg, 4)}, readability avg: {round(down_readability_avg, 4)}"
    )

    print(
        f"up ** bleu_avg: {round(up_bleu_avg, 4)}, bert avg: {round(up_bert_avg, 4)}, readability avg: {round(up_readability_avg, 4)}"
    )


if __name__ == "__main__":
    # single_test()
    # batch_test()
    # inference()
    # fast_score()
    bind_data()
    # read_readability_score()
    fast_analysis()
