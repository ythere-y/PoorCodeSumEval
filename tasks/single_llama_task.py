import json
import os
import tempfile
from turtle import filling
from numpy import save
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import write_basic_config
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from torch.utils.data import DataLoader
from bigcode_eval.arguments import EvalArguments

from colorama import Fore, Style
from tests.data_utils import (
    FinalDataProcess,
    data_name_dict,
    load_local_dataset,
    load_local_datasetv2,
    load_local_datasetv3,
    save_ref_and_gen_data,
    handle_reference,
)

import torch

from tests.path_utils import FinalPath

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

TMPDIR = tempfile.mkdtemp()
TEST_MODEL = "codellama/CodeLlama-7b-hf"
TEST_MODEL = "codellama/CodeLlama-7b-Instruct-hf"
model_name = "CodeLlama-7b-hf"


os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


model, tokenizer, accelerator = setup()


def single_dataset_gen(
    partition_name: str,
    task_name: str,
    type_name: str,
    mode_name: str,
    lang_name: str,
    start_point=0,
    limit=20,
):
    set_seed(0)
    dataset = FinalDataProcess.load_local_dataset(
        partition_name,
        type_name,
        mode_name,
        lang_name,
    )

    references, fillings = evaluation_process(
        lang_name,
        model,
        tokenizer,
        accelerator,
        dataset,
        start_point=start_point,
        limit=limit,
        process_name=f"{type_name}_{mode_name}",
    )
    FinalDataProcess.save_ref_and_gen_data(
        model_name,
        partition_name,
        type_name,
        mode_name,
        lang_name,
        task_name,
        start_point,
        limit,
        fillings,
        references,
    )


def prompt_builder(lang_name, sample):
    if lang_name == "java" or lang_name == "go":
        code_string = sample["code"]
        system = "Generate docstring for the code in 12 words. "
        user = (
            code_string
            + '\n"""Please fill this sentence "The goal of this function is to " in 12 words: '
        )

        prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user}[/INST]"
    elif lang_name == "python":
        code_string = sample["code"]
        doc_string = sample["docstring"]
        prompt = code_string.replace(doc_string, "<FILL_ME>", 1)
    return prompt


def evaluation_process(
    lang_name,
    model,
    tokenizer,
    accelerator,
    dataset,
    start_point=0,
    limit=20,
    process_name="",
):
    references, fillings = [], []
    for index in tqdm(range(start_point, limit), desc=f"Generating {process_name}"):
        sample = dataset[index]
        reference = handle_reference(sample)
        prompt = prompt_builder(lang_name, sample)
        input_ids = tokenizer(prompt, return_tensors="pt", padding=False)["input_ids"]
        input_ids = input_ids.to(accelerator.device)
        generated_ids = model.generate(input_ids, max_new_tokens=128)
        filling = tokenizer.batch_decode(
            generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]

        if index % 100 == 0:
            # print(f"{Fore.YELLOW}prompt = {Fore.RESET}{prompt}")
            # print(f"{Fore.GREEN}reference = {Fore.RESET}{reference}")
            # print(f"{Fore.RED}filling = {Fore.RESET}{filling}")
            torch.cuda.empty_cache()  # 这段是为了释放显存，避免评测过程中显存不断增大的问题

        fillings.append(filling)
        references.append(reference)
    return references, fillings


def run_cross(lang_name, limit):
    partition = "cross"
    for type_name in FinalPath.PATH_NAME_DICT[partition]:
        for mode_name in FinalPath.PATH_NAME_DICT[partition][type_name]:
            single_dataset_gen(
                partition_name=partition,
                type_name=type_name,
                mode_name=mode_name,
                task_name="work",
                lang_name=lang_name,
                limit=limit,
            )


def run_all(lang_name, limit):
    for partition in FinalPath.PATH_NAME_DICT:
        for type_name in FinalPath.PATH_NAME_DICT[partition]:
            for mode_name in FinalPath.PATH_NAME_DICT[partition][type_name]:
                single_dataset_gen(
                    partition_name=partition,
                    type_name=type_name,
                    mode_name=mode_name,
                    task_name="work",
                    lang_name=lang_name,
                    limit=limit,
                )


if __name__ == "__main__":
    lang_name = "go"
    limit = 2000
    run_cross(lang_name, limit)
    single_dataset_gen(
        partition_name="origin",
        type_name="origin",
        mode_name="origin",
        task_name="work",
        lang_name=lang_name,
        limit=limit,
    )
    single_dataset_gen(
        partition_name="single",
        type_name="semantic",
        mode_name="FNE",
        task_name="work",
        lang_name=lang_name,
        limit=limit,
    )

    single_dataset_gen(
        partition_name="single",
        type_name="semantic",
        mode_name="IHR",
        task_name="work",
        lang_name=lang_name,
        limit=limit,
    )

    single_dataset_gen(
        partition_name="single",
        type_name="semantic",
        mode_name="IOE",
        task_name="work",
        lang_name=lang_name,
        limit=limit,
    )

    single_dataset_gen(
        partition_name="single",
        type_name="semantic",
        mode_name="IRS",
        task_name="work",
        lang_name=lang_name,
        limit=limit,
    )

    single_dataset_gen(
        partition_name="single",
        type_name="semantic",
        mode_name="IRS",
        task_name="work",
        lang_name=lang_name,
        limit=limit,
    )

    single_dataset_gen(
        partition_name="single",
        type_name="synatic",
        mode_name="DBI",
        task_name="work",
        lang_name=lang_name,
        limit=limit,
    )

    single_dataset_gen(
        partition_name="single",
        type_name="synatic",
        mode_name="HVI",
        task_name="work",
        lang_name=lang_name,
        limit=limit,
    )

    single_dataset_gen(
        partition_name="single",
        type_name="synatic",
        mode_name="OOS",
        task_name="work",
        lang_name=lang_name,
        limit=limit,
    )
    # batch_dataset_test(lang_name="java", task_name="batch_work_java", limit=2000)
    # single_dataset_gen("semantic", "IHR")
    print("**************************************")
    print("**************************************")
    print(f"**********{Fore.GREEN}All Done!{Style.RESET_ALL}**********")
    print("**************************************")
    print("**************************************")
