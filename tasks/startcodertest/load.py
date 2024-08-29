from datasets import load_dataset

dataset = load_dataset("code_x_glue_ct_code_to_text", "python", trust_remote_code=True)
