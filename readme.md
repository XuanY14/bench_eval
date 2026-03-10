# Bench Eval

A lightweight LLM evaluation framework featuring batch inference across diverse benchmarks and LLM-based judging. It supports both local models and API-based evaluators.

## Quick Start

### 1. Configure Datasets

Edit the `configs/datasets.yaml` file to add or modify your dataset configurations.

**Example (Hugging Face Datasets):**

```yaml
datasets:
  gsm8k:
    type: hf
    path: gsm8k
    config: main
    split: test
    problem_key: question
    label_key: answer
    label_processor: "lambda x: x.split('####')[-1].strip()"

  math:
    type: hf
    path: lighteval/MATH
    split: test
    problem_key: problem
    label_key: solution
    label_processor: "lambda x: re.findall(r'\\\\boxed\\{([^}]*)\\}', x)[-1] if re.findall(r'\\\\boxed\\{([^}]*)\\}', x) else None"
```

**Example (Local Files):**

```yaml
  my_custom_dataset:
    type: local
    path: data/my_dataset.jsonl  # Or data/my_dataset.parquet
    problem_key: problem
    label_key: answer
    label_processor: "str"
```

### 2. Prepare Prompt Templates (Optional)

- **Generation Template (`generate_template.txt`)**: Defines the prompt structure sent to the generation model.
  - **Variables**: `{problem}`
  - **Example**:
    ```text
    Solve the following math problem step by step.

    Problem: {problem}

    Solution:
    ```
- **Evaluation Template (`judge_template.txt`)**: Defines the prompt structure sent to the evaluation model.
  - **Variables**: `{problem}`, `{label}`, `{prediction}`
  - **Example**:
    ```text
    I will show you a math problem, a reference answer, and a predicted answer. Your task is to determine if the predicted answer is correct.

    [Math Problem]
    {problem}

    [Reference Answer]
    {label}

    [Predicted Answer]
    {prediction}

    Are these two answers mathematically equivalent? Respond only "Yes" or "No".
    ```

### 3. Run Evaluation

```bash
python run_eval.py \
  --generator_model /path/to/your/generator/model \
  --evaluator_model /path/to/your/evaluator/model \
  --dataset gsm8k math \
  --tensor_parallel_size 8 \
  --output_dir /path/to/output/directory \
  --generate_template_path templates/generate_template.txt \
  --judge_template_path templates/judge_template.txt
```

**Argument Descriptions:**

- `--generator_model`: Path to the model used for generating answers (HuggingFace Hub ID or local path).
- `--evaluator_model`: Path to the model used for evaluating answer correctness (HuggingFace Hub ID or local path).
- `--dataset`: One or more dataset names to evaluate (names must match definitions in `configs/datasets.yaml`). Example: `gsm8k math`.
- `--dataset_config`: (Optional) Path to the dataset configuration file. Defaults to `configs/datasets.yaml`.
- `--output_dir`: (Optional) Directory to save results. Defaults to `outputs`.
- `--tensor_parallel_size`: (Optional) Number of GPUs to use. Defaults to 1.
- `--max_gen_tokens`: (Optional) Max output tokens for the generator model. Defaults to 1024.
- `--max_eval_tokens`: (Optional) Max output tokens for the evaluator model. Defaults to 10.
- `--generate_template_path`: (Optional) Path to the generation prompt template. Defaults to `templates/generate_template.txt`.
- `--judge_template_path`: (Optional) Path to the evaluation prompt template. Defaults to `templates/judge_template.txt`.
- `--debug_eval`: (Optional) Manually set `DEBUG_EVAL = True` in `run_eval.py` to enable detailed debug output during the evaluation phase.

## Results

After evaluation, results are saved in the directory specified by `--output_dir`, structured as follows:

```
/path/to/output/directory/
└── <generator_model_name>_vs_<evaluator_model_name>/
    ├── gsm8k/
    │   ├── generated.jsonl  # Results from Step 1 (generation)
    │   └── results.jsonl    # Final evaluation results
    ├── math/
    │   ├── generated.jsonl
    │   └── results.jsonl
    └── ...
```