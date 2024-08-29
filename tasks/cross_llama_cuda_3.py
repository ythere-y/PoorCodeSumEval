import json
import os
import tempfile
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import write_basic_config
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from bigcode_eval.arguments import EvalArguments

from colorama import Fore, Style
from tests.data_utils import (
    load_cross_dataset,
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
TEST_MODEL = "codellama/CodeLlama-7b-hf"
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


def cross_dataset_gen(
    task_name: str = "", mode_1=None, mode_2=None, start_point=0, limit=20
):
    args = update_args(EvalArguments())
    set_seed(args.seed)
    model, tokenizer, accelerator = setup()

    dataset = load_cross_dataset(mode_1, mode_2)

    if limit == -1:
        limit = dataset.shape[0]
    fillings = evaluation_process(
        model,
        tokenizer,
        accelerator,
        dataset,
        start_point=start_point,
        limit=limit,
        process_name=f"{mode_1}_{mode_2}",
    )
    save_cross_gen_results(mode_1, mode_2, fillings, start_point, limit, task_name)


def evaluation_process(
    model, tokenizer, accelerator, dataset, start_point=0, limit=20, process_name=""
):
    fillings = []
    for index in tqdm(range(start_point, limit), desc=f"Generating {process_name}"):
        sample = dataset[index]
        code_string = sample["code"]
        doc_string = sample["docstring"]

        prompt = code_string.replace(doc_string, "<FILL_ME>", 1)
        input_ids = tokenizer(prompt, return_tensors="pt", padding=False)["input_ids"]
        input_ids = input_ids.to(accelerator.device)
        generated_ids = model.generate(input_ids, max_new_tokens=128)
        filling = tokenizer.batch_decode(
            generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]
        torch.cuda.empty_cache()
        if index % 100 == 0:
            print(f"{Fore.YELLOW}prompt = {Fore.RESET}{prompt}")
            print(f"{Fore.RED}filling = {Fore.RESET}{filling}")

        fillings.append(filling)
    return fillings


def save_cross_gen_results(mode_1, mode_2, fillings, start_point, limit, task_name):
    save_dir = f'ref_and_gen/{TEST_MODEL.split("/")[-1]}/cross'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(
        f"{save_dir}/{task_name}_{mode_1}_{mode_2}_[{start_point}-{limit}]_fill.json",
        "w",
    ) as f:
        json.dump(fillings, f, indent=4)
        f.close()


if __name__ == "__main__":
    # cross_dataset_gen(task_name="cross_work", mode_1="FNE", mode_2="DBI", limit=2000)
    # cross_dataset_gen(task_name="cross_work", mode_1="FNE", mode_2="OOS", limit=2000)
    # cross_dataset_gen(task_name="cross_work", mode_1="FNE", mode_2="HVI", limit=2000)

    # cross_dataset_gen(task_name="cross_work", mode_1="IOE", mode_2="DBI", limit=2000)
    # cross_dataset_gen(task_name="cross_work", mode_1="IOE", mode_2="OOS", limit=2000)
    # cross_dataset_gen(task_name="cross_work", mode_1="IOE", mode_2="HVI", limit=2000)

    # cross_dataset_gen(task_name="cross_work", mode_1="IHR", mode_2="DBI", limit=2000)
    # cross_dataset_gen(task_name="cross_work", mode_1="IHR", mode_2="OOS", limit=2000)
    # cross_dataset_gen(task_name="cross_work", mode_1="IHR", mode_2="HVI", limit=2000)

    # cross_dataset_gen(task_name="cross_work", mode_1="IRS", mode_2="DBI", limit=2000)
    # cross_dataset_gen(task_name="cross_work", mode_1="IRS", mode_2="OOS", limit=2000)
    cross_dataset_gen(task_name="cross_work", mode_1="IRS", mode_2="HVI", limit=2000)

    print("**************************************")
    print("**************************************")
    print(f"**********{Fore.GREEN}All Done!{Style.RESET_ALL}**********")
    print("**************************************")
    print("**************************************")
