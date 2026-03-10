\# Bench Eval



A framework for evaluating the mathematical problem-solving capabilities of Large Language Models (LLMs) using two separate models: a Generator and an Evaluator.



\## Features



\- \*\*Dual-Model Evaluation\*\*: Uses one model (Generator) to produce answers and another (Evaluator) to assess the correctness of those answers.

\- \*\*Multi-Dataset Support\*\*: Compatible with Hugging Face Datasets or local files (JSONL, Parquet, Arrow, etc.).

\- \*\*Configurable Prompt Templates\*\*: Supports custom templates for generation and evaluation prompts, facilitating adaptation to different domains.

\- \*\*Optional "Thinking" Mode Disabling\*\*: When supported by vLLM, can disable the model's internal reasoning steps during generation and evaluation, prompting direct output.

\- \*\*Result Logging\*\*: Saves detailed per-sample results and prints overall accuracy for each dataset.

\- \*\*Progress Tracking\*\*: Utilizes `tqdm` to display progress during generation and evaluation phases.



\## Installation



```bash

\# It's recommended to create a virtual environment (e.g., conda or venv)

conda create -n matheval python=3.10

conda activate matheval



\# Install dependencies

pip install vllm transformers datasets pyyaml tqdm

```



\## Project Structure



```

math\_eval\_framework/

├── run\_eval.py                 # Main evaluation script

├── utils.py                    # Utility functions (loading data, extracting answers)

├── configs/

│   └── datasets.yaml           # Dataset configuration file

├── templates/

│   ├── generate\_template.txt   # Generation model prompt template (default)

│   └── judge\_template.txt      # Evaluator model prompt template (default)

├── data/                       # (Optional) Local dataset files (.jsonl, .parquet, .arrow, etc.)

└── README.md

```



\## Quick Start



\### 1. Configure Datasets



Edit the `configs/datasets.yaml` file to add or modify your dataset configurations.



\*\*Example (Hugging Face Datasets):\*\*



```yaml

datasets:

&nbsp; gsm8k:

&nbsp;   type: hf

&nbsp;   path: gsm8k

&nbsp;   config: main

&nbsp;   split: test

&nbsp;   problem\_key: question

&nbsp;   label\_key: answer

&nbsp;   label\_processor: "lambda x: x.split('####')\[-1].strip()"



&nbsp; math:

&nbsp;   type: hf

&nbsp;   path: lighteval/MATH

&nbsp;   split: test

&nbsp;   problem\_key: problem

&nbsp;   label\_key: solution

&nbsp;   label\_processor: "lambda x: re.findall(r'\\\\\\\\boxed\\\\{(\[^}]\*)\\\\}', x)\[-1] if re.findall(r'\\\\\\\\boxed\\\\{(\[^}]\*)\\\\}', x) else None"

```



\*\*Example (Local Files):\*\*



```yaml

&nbsp; my\_custom\_dataset:

&nbsp;   type: local

&nbsp;   path: data/my\_dataset.jsonl  # Or data/my\_dataset.parquet

&nbsp;   problem\_key: problem

&nbsp;   label\_key: answer

&nbsp;   label\_processor: "str"

```



\### 2. Prepare Prompt Templates (Optional)



\- \*\*Generation Template (`generate\_template.txt`)\*\*: Defines the prompt structure sent to the generation model.

&nbsp; - \*\*Variables\*\*: `{problem}`

&nbsp; - \*\*Example\*\*:

&nbsp;   ```text

&nbsp;   Solve the following math problem step by step.



&nbsp;   Problem: {problem}



&nbsp;   Solution:

&nbsp;   ```

\- \*\*Evaluation Template (`judge\_template.txt`)\*\*: Defines the prompt structure sent to the evaluation model.

&nbsp; - \*\*Variables\*\*: `{problem}`, `{label}`, `{prediction}`

&nbsp; - \*\*Example\*\*:

&nbsp;   ```text

&nbsp;   I will show you a math problem, a reference answer, and a predicted answer. Your task is to determine if the predicted answer is correct.



&nbsp;   \[Math Problem]

&nbsp;   {problem}



&nbsp;   \[Reference Answer]

&nbsp;   {label}



&nbsp;   \[Predicted Answer]

&nbsp;   {prediction}



&nbsp;   Are these two answers mathematically equivalent? Respond only "Yes" or "No".

&nbsp;   ```



\### 3. Run Evaluation



```bash

python run\_eval.py \\

&nbsp; --generator\_model /path/to/your/generator/model \\

&nbsp; --evaluator\_model /path/to/your/evaluator/model \\

&nbsp; --dataset gsm8k math \\

&nbsp; --tensor\_parallel\_size 8 \\

&nbsp; --output\_dir /path/to/output/directory \\

&nbsp; --generate\_template\_path templates/generate\_template.txt \\

&nbsp; --judge\_template\_path templates/judge\_template.txt

```



\*\*Argument Descriptions:\*\*



\- `--generator\_model`: Path to the model used for generating answers (HuggingFace Hub ID or local path).

\- `--evaluator\_model`: Path to the model used for evaluating answer correctness (HuggingFace Hub ID or local path).

\- `--dataset`: One or more dataset names to evaluate (names must match definitions in `configs/datasets.yaml`). Example: `gsm8k math`.

\- `--dataset\_config`: (Optional) Path to the dataset configuration file. Defaults to `configs/datasets.yaml`.

\- `--output\_dir`: (Optional) Directory to save results. Defaults to `outputs`.

\- `--tensor\_parallel\_size`: (Optional) Number of GPUs to use. Defaults to 1.

\- `--max\_gen\_tokens`: (Optional) Max output tokens for the generator model. Defaults to 1024.

\- `--max\_eval\_tokens`: (Optional) Max output tokens for the evaluator model. Defaults to 10.

\- `--generate\_template\_path`: (Optional) Path to the generation prompt template. Defaults to `templates/generate\_template.txt`.

\- `--judge\_template\_path`: (Optional) Path to the evaluation prompt template. Defaults to `templates/judge\_template.txt`.

\- `--debug\_eval`: (Optional) Manually set `DEBUG\_EVAL = True` in `run\_eval.py` to enable detailed debug output during the evaluation phase.



\## Results



After evaluation, results are saved in the directory specified by `--output\_dir`, structured as follows:



```

/path/to/output/directory/

└── <generator\_model\_name>\_vs\_<evaluator\_model\_name>/

&nbsp;   ├── gsm8k/

&nbsp;   │   ├── generated.jsonl  # Results from Step 1 (generation)

&nbsp;   │   └── results.jsonl    # Final evaluation results

&nbsp;   ├── math/

&nbsp;   │   ├── generated.jsonl

&nbsp;   │   └── results.jsonl

&nbsp;   └── ...

```



