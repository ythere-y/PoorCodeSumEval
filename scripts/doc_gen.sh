#!bash
cd ..
export HF_ENDPOINT=https://hf-mirror.com
accelerate launch  main.py \
  --model "bigcode/santacoder" \
  --max_length_generation 512 \
  --tasks codexglue_code_to_text-python-left \
  --n_samples 1 \
  --batch_size 1 \
  --trust_remote_code \
  --save_generations \
  --save_generations_path "/data2/huchao/11.novels/bigcode-evaluation-harness/ref_and_gen/generations.json" \
  --save_references \
  --save_references_path "/data2/huchao/11.novels/bigcode-evaluation-harness/ref_and_gen/references.json"
  # --check_references