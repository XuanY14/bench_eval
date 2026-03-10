#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

cd /mnt/DataFlow/wyz/bench_eval

echo "🧪 Running evaluation for fine-tuned model..."
python run_eval.py \
  --generator_model /mnt/DataFlow/wyz/Math_sft_output/qwen3_8b/full_solution/math_cot/checkpoint-1500 \
  --evaluator_model /mnt/DataFlow/models/Qwen3-8B \
  --dataset aime2024 gsm8k aime2025 math500 \
  --generate_template_path /mnt/DataFlow/wyz/bench_eval/templates/general/generate_template.txt \
  --judge_template_path /mnt/DataFlow/wyz/bench_eval/templates/general/judge_template.txt \
  --tensor_parallel_size 8 \
  --output_dir /mnt/DataFlow/wyz/bench_eval/output/Qwen3-8B-math_cot_ckp1500_3

python run_eval.py \
  --generator_model /mnt/DataFlow/wyz/Math_sft_output/qwen3_8b/full_solution/math_cot/checkpoint-1500 \
  --evaluator_model /mnt/DataFlow/models/Qwen3-8B \
  --dataset aime2024 gsm8k aime2025 math500 \
  --generate_template_path /mnt/DataFlow/wyz/bench_eval/templates/math/generate_template_math.txt \
  --judge_template_path /mnt/DataFlow/wyz/bench_eval/templates/math/judge_template_math.txt \
  --tensor_parallel_size 8 \
  --output_dir /mnt/DataFlow/wyz/bench_eval/output/Qwen3-8B-math_cot_ckp1500_math_template_2

python run_eval.py \
  --generator_model /mnt/DataFlow/wyz/Math_sft_output/qwen3_8b/full_solution/math_cot/checkpoint-1500 \
  --evaluator_model /mnt/DataFlow/models/Qwen3-8B \
  --dataset aime2024 gsm8k aime2025 math500 \
  --generate_template_path /mnt/DataFlow/wyz/bench_eval/templates/math/generate_template_math.txt \
  --judge_template_path /mnt/DataFlow/wyz/bench_eval/templates/math/judge_template_math.txt \
  --tensor_parallel_size 8 \
  --output_dir /mnt/DataFlow/wyz/bench_eval/output/Qwen3-8B-math_cot_ckp1500_math_template_3

# chmod +x run_train_and_eval.sh
# ./run_train_and_eval.sh