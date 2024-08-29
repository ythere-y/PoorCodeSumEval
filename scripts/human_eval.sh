#!bash
cd ..

accelerate launch main.py \
  --model "bigcode/santacoder" \
  --max_length_generation 512 \
  --tasks humanevalsynthesize-python \
  --temperature 0.2 \
  --n_samples 20 \
  --batch_size 10 \
  --allow_code_execution \
  --prompt "instruct" \
  --save_generations


