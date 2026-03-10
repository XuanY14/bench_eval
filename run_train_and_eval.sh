#!/bin/bash
python run_eval.py \
  --generator_model  \
  --evaluator_model  \
  --dataset aime2024 gsm8k aime2025 math500 \
  --generate_template_path bench_eval/templates/math/generate_template_math.txt \
  --judge_template_path bench_eval/templates/math/judge_template_math.txt \
  --tensor_parallel_size 8 \
  --output_dir bench_eval/output/Qwen3-8B-math_cot_ckp1500_math_template_3
