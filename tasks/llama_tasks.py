import json
import os
import tempfile
from turtle import filling
from numpy import save
from datasets import load_dataset
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import write_basic_config
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from bigcode_eval.arguments import EvalArguments

from colorama import Fore, Style
from tests.data_utils import (
    data_name_dict,
    load_local_dataset,
    load_local_datasetv2,
)

# TODO add more tasks
import torch

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# Tasks for generation test
# GEN_TASKS = ["humaneval", "mbpp"]
GEN_TASKS = ["codexglue_code_to_text-python-left"]
# Tasks for evaluator tests
EVAL_TASKS = ["humaneval", "mbpp", "pal-gsm8k-greedy"]
TMPDIR = tempfile.mkdtemp()
TEST_MODEL = "hf-internal-testing/tiny-random-gpt2"
TEST_MODEL = "bigcode/santacoder"
TEST_MODEL = "codellama/CodeLlama-7b-hf"
REF_EVAL_SCORES = {
    "humaneval": {"pass@1": 0.25},
    "mbpp": {"pass@1": 0.25},
    "pal-gsm8k-greedy": {"accuracy": 1.0, "num_failed_execution": 0},
}

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    model = AutoModelForCausalLM.from_pretrained(TEST_MODEL)
    print(f"{Fore.GREEN}loading tokenizer of {TEST_MODEL}...{Fore.RESET}")
    tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    configPath = os.path.join(TMPDIR, "default_config.yml")
    write_basic_config(save_location=configPath)
    accelerator = Accelerator()
    model = model.to(accelerator.device)
    return model, tokenizer, accelerator


def test_code_llama_7b():
    args = update_args(EvalArguments())
    set_seed(args.seed)
    model, tokenizer, accelerator = setup()
    PROMPT = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''
    PROMPT = 'def sina_xml_to_url_list(xml_data):\n    """<FILL_ME>\n"""\n    rawurl = []\n    dom = parseString(xml_data)\n    for node in dom.getElementsByTagName(\'durl\'):\n        url = node.getElementsByTagName(\'url\')[0]\n        rawurl.append(url.childNodes[0].data)\n    return rawurl'
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    generated_ids = model.generate(input_ids, max_new_tokens=128)
    filling = tokenizer.batch_decode(
        generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )[0]
    print(f"{Fore.GREEN} input code :  \n{PROMPT}{Style.RESET_ALL}")
    print(f"{Fore.BLUE} generated code : \n{filling}{Style.RESET_ALL}")
    print(PROMPT.replace("<FILL_ME>", filling))


def single_dataset_gen(
    task_name: str = "", type_name=None, mode_name=None, start_point=0, limit=20
):
    args = update_args(EvalArguments())
    set_seed(args.seed)
    model, tokenizer, accelerator = setup()
    if type_name == None and mode_name == None:
        type_name = "origin"
        mode_name = "origin"
        dataset = load_dataset(
            "/data2/huchao/11.novels/download/code_x_glue_ct_code_to_text", "python"
        )["test"]
    else:
        # dataset = load_local_dataset(type_name, partition_name)["test"]
        dataset = load_local_datasetv2(type_name, mode_name)
    if limit == -1:
        limit = dataset.shape[0]
    references, fillings = evaluation_process(
        model,
        tokenizer,
        accelerator,
        dataset,
        start_point=start_point,
        limit=limit,
        process_name=f"{type_name}_{mode_name}",
    )
    save_json_files(
        type_name, mode_name, references, fillings, start_point, limit, task_name
    )


# TODO:HC:reference
def handle_reference(sample):
    from mosestokenizer import MosesDetokenizer

    docstring = " ".join(sample["docstring_tokens"]).replace("\n", "")
    if docstring[0] == "r":
        docstring = docstring[1:]
    with MosesDetokenizer("en") as detokenize:
        docstring = detokenize(docstring.strip().split())

    reference = docstring
    return reference


def evaluation_process(
    model, tokenizer, accelerator, dataset, start_point=0, limit=20, process_name=""
):
    references, fillings = [], []
    for index in tqdm(range(0, limit), desc=f"Generating {process_name}"):
        sample = dataset[index]
        code_string = sample["code"]
        doc_string = sample["docstring"]
        reference = doc_string

        prompt = code_string.replace(doc_string, "<FILL_ME>", 1)
        input_ids = tokenizer(prompt, return_tensors="pt", padding=False)["input_ids"]
        input_ids = input_ids.to(accelerator.device)
        generated_ids = model.generate(input_ids, max_new_tokens=128)
        filling = tokenizer.batch_decode(
            generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]

        torch.cuda.empty_cache()  # 这段是为了释放显存，避免评测过程中显存不断增大的问题
        if index % 100 == 0:
            print(f"{Fore.YELLOW}prompt = {Fore.RESET}{prompt}")
            print(f"{Fore.GREEN}reference = {Fore.RESET}{reference}")
            print(f"{Fore.RED}filling = {Fore.RESET}{filling}")
        references.append(reference)
        fillings.append(filling)
    return references, fillings


def save_json_files(
    type_name, partition_name, references, fillings, start_point, limit, task_name
):
    save_dir = f'ref_and_gen/{TEST_MODEL.split("/")[-1]}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(
        f"{save_dir}/{task_name}_{type_name}_{partition_name}_[{start_point}-{limit}]_ref.json",
        "w",
    ) as f:
        json.dump(references, f, indent=4)
        f.close()
    with open(
        f"{save_dir}/{task_name}_{type_name}_{partition_name}_[{start_point}-{limit}]_fill.json",
        "w",
    ) as f:
        json.dump(fillings, f, indent=4)
        f.close()


def batch_dataset_test(task_name: str = "", start_point=0, limit=200):
    args = update_args(EvalArguments())
    set_seed(args.seed)
    model, tokenizer, accelerator = setup()

    for type_name in data_name_dict.keys():
        for mode_name in data_name_dict[type_name]:
            print("**************************************")
            print(
                f"**********Evaluating {type_name} / {mode_name} with model {TEST_MODEL}**********"
            )
            print("**************************************")
            # dataset = load_local_dataset(type_name, partition_name)["test"]
            dataset = load_local_datasetv2(type_name, mode_name)
            references, fillings = evaluation_process(
                model,
                tokenizer,
                accelerator,
                dataset,
                start_point=start_point,
                limit=limit,
                process_name=f"{type_name}_{mode_name}",
            )
            save_json_files(
                type_name,
                mode_name,
                references,
                fillings,
                start_point,
                limit,
                task_name,
            )


if __name__ == "__main__":
    # test_code_llama_7b()
    single_dataset_gen(limit=2000)
    # single_dataset_gen(
    # task_name="cuda_3_work", type_name="semantic", partition_name="FNE", limit=2000
    # )
    # single_dataset_gen(
    #     task_name="work", type_name="semantic", partition_name="FNE", limit=2000
    # )
    # single_dataset_gen(
    #     task_name="work", type_name="semantic", partition_name="IOE", limit=2000
    # )
    # single_dataset_gen(
    #     task_name="work", type_name="semantic", partition_name="IRS", limit=2000
    # )
    # single_dataset_gen(
    #     task_name="batch_test", type_name="synatic", partition_name="DBI", limit=2000
    # )
    # single_dataset_gen(
    #     task_name="batch_test", type_name="synatic", partition_name="HVI", limit=2000
    # )
    # single_dataset_gen(
    #     task_name="batch_test", type_name="synatic", partition_name="OOS", limit=2000
    # )
    # batch_dataset_test(task_name="batch_test", limit=200)
    # single_dataset_gen("semantic", "IHR")
    print("**************************************")
    print("**************************************")
    print(f"**********{Fore.GREEN}All Done!{Style.RESET_ALL}**********")
    print("**************************************")
    print("**************************************")
