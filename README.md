# PoorCodeSumEval

This repository for our ASE 2024 paper "How Effective Do Code Language Models Understand Poor-Readability Code?" includes benchmark suite, results, methods for acquiring and preparing materials, and source code of our automatic scoring tool. We hope this artifact can motivate and help future research on code summarization.


## What's inside this repository

1. Script to construct perturbed datasets from source data.
2. Automatic inference scripts. Models include: `CodeBERT`, `CodeT5`, `Codellama`. Programming languages ​​include: `Go`, `Java`, `Python`. Data types include: `source data` and `perturbation generated data`.
3. Script for automatic scoring, scoring targets include: `CodeBERT`, `CodeT5`, `CodeLlama`, `GPT-4o` 's inference results. Evaluation indicators include: `BLEUScore`, `BERTScore`, `P-value`

## Get Started


Experiments are conducted using Python 3.9.7 on a Ubuntu 22.01.1 server.

To install all required packages, navigate to the root directory of this project and run:

```bash
git clone https://github.com/ythere-y/PoorCodeSumEval.git
cd PoorCodeSumEval
pip install -r requirements.txt
```

### DataPreparation

Get CodeXGlue dataset from https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text


Get TL-CodeSum from https://github.com/xing-hu/TL-CodeSum

Get DeppCom from https://github.com/xing-hu/EMSE-DeepCom

## Construct Dataset


`process_data/RobustCodeSum.py` process Python Code.

`process_data/RobustCodeSumGo.py` process Go Code.

`process_data/RobustCodeSumJava.py` process Java Code.

### Example

Use `Python` & `CodeXGlue` as an example to construct the IOE perturbation dataset.

#### Code Edit: `process_data/CodeXGlue.py`

```python
    DATASET_PATH = "path_to_code_x/code_x_glue_ct_code_to_text"
```

in main function

```python
if __name__ == "__main__":
    robust = PythonRobustCodeSum()
    robust.gen_IOE()
```

#### Run code

```bash
python process_data/RobustCodeSum.py
```

#### Result

The result dataset will saved into `local_data/single/semantic/IOE/python/CSN`

## Inference

Inference with CodeBERT or CodeT5 or CodeLlama-7b, in `tasks`

### Example

Use `Go` & `CodeLlama` & `FNE` as an example to conduct inference.

#### Code Edit: `tasks/single_llama_task.py`

Edit the main function like this to set the language and the dataset type.

```python
if __name__ == "__main__":
    lang_name = "go"
    limit = 2000
    single_dataset_gen(
        partition_name="single",
        type_name="semantic",
        mode_name="FNE",
        task_name="work",
        lang_name=lang_name,
        limit=limit,
    )
```

#### Run Code

```bash
python tasks/single_llama_task.py
```

#### Result

The result includes the inference results of CodeLlama-7b on the Go dataset with FNE perturbation and the reference summaries.

The result will be saved into path `ref_and_gen/codellama-7b/single/semantic/FNE/go/work_gen_[0-2000].json` and `ref_and_gen/codellama-7b/single/semantic/FNE/go/work_ref_[0-2000].json`

## Calculating the score
TODO: add p-value calculation
In `scores`, BLEU and BERT scores are calculated.

## Appendix

Some explanations of common questions and experiments on P-value can be found in `appendix.pdf`.
