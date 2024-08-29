import json
import os
import tempfile
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import write_basic_config
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from bigcode_eval.arguments import EvalArguments

from colorama import Fore, Style, Back
from tests.data_utils import data_name_dict, load_local_dataset

# TODO add more tasks
import torch

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# Tasks for generation test
# GEN_TASKS = ["humaneval", "mbpp"]
GEN_TASKS = ["codexglue_code_to_text-python-left"]
# Tasks for evaluator tests
EVAL_TASKS = ["humaneval", "mbpp", "pal-gsm8k-greedy"]
TMPDIR = tempfile.mkdtemp()
TEST_MODEL = "hf-internal-testing/tiny-random-gpt2"
TEST_MODEL = "codellama/CodeLlama-7b-hf"
TEST_MODEL = "bigcode/santacoder"
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


def single_dataset_gen(type_name, partition_name):
    args = update_args(EvalArguments())
    set_seed(args.seed)
    model, tokenizer, accelerator = setup()
    dataset = load_local_dataset(type_name, partition_name)["test"]

    references, generations, fillings = [], [], []
    # for sample in tqdm(range(0, 2), desc="Generating"):
    for sample in tqdm(range(0, dataset.shape[0]), desc="Generating"):
        doc = dataset[sample]
        code = doc["code"]

        from mosestokenizer import MosesDetokenizer

        docstring = " ".join(doc["docstring_tokens"]).replace("\n", "")
        if docstring[0] == "r":
            docstring = docstring[1:]
        with MosesDetokenizer("en") as detokenize:
            docstring = detokenize(docstring.strip().split())

        reference = docstring

        code = doc["code"]
        # python code includes the docstring
        text = doc["docstring"]
        # TODO:HC:prompt edit

        prompt = code[: code.index(text)]
        input_ids = tokenizer(prompt, return_tensors="pt", padding=False)["input_ids"]
        input_ids = input_ids.to(accelerator.device)
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=128,
            # stop_token=tokenizer.eos_token_id,
            # log_probs=False,
        )
        generated = tokenizer.batch_decode(
            generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]

        if sample % 100:
            print(f"{Fore.YELLOW}prompt = {Fore.RESET}{prompt}")
            print(f"{Fore.BLUE}[{sample}]Generated code: {Fore.RESET}{generated}")

        references.append(reference)
        generations.append(generated)

    save_dir = f'ref_and_gen/{TEST_MODEL.split("/")[-1]}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(f"{save_dir}/{type_name}_{partition_name}_gen.json", "w") as f:
        json.dump(generations, f, indent=4)
        f.close()
    with open(f"{save_dir}/{type_name}_{partition_name}_ref.json", "w") as f:
        json.dump(references, f, indent=4)
        f.close()


if __name__ == "__main__":
    # test_code_llama_7b()
    single_dataset_gen("semantic", "IHR")
