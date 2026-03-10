# run_eval.py
import argparse
import os
import json
import re
from tqdm import tqdm 
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils import load_dataset_config, load_benchmark, extract_answer

def apply_chat_template(tokenizer, messages, enable_thinking=True):
    """Apply chat template with enable_thinking option."""
    return tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

def main():
    parser = argparse.ArgumentParser(description="Evaluate a generator model using an evaluator LLM.")
    # Generator Model
    parser.add_argument("--generator_model", type=str, required=True, help="Path to the generator model")
    # Evaluator Model
    parser.add_argument("--evaluator_model", type=str, required=True, help="Path to the evaluator model")
    # Data
    parser.add_argument("--dataset", type=str, nargs='+', required=True, help="Name(s) of the dataset(s) (e.g., gsm8k, math, omnimath olympiadbench)")
    parser.add_argument("--dataset_config", type=str, default="configs/datasets.yaml", help="Path to dataset config YAML")
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save results")
    # vLLM args
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--max_gen_tokens", type=int, default=4096, help="Max tokens for generator")
    parser.add_argument("--max_eval_tokens", type=int, default=32, help="Max tokens for evaluator")
    # Template args
    parser.add_argument("--generate_template_path", type=str, default="templates/generate_template.txt", help="Path to the generation prompt template")
    parser.add_argument("--judge_template_path", type=str, default="templates/judge_template.txt", help="Path to the judge prompt template")
    args = parser.parse_args()

    # Load dataset config
    all_datasets = load_dataset_config(args.dataset_config)

    gen_model_name = os.path.basename(args.generator_model.rstrip("/"))
    eval_model_name = os.path.basename(args.evaluator_model.rstrip("/"))

    all_results = {}
    # Iterate over each specified dataset
    for dataset_name in args.dataset:
        if dataset_name not in all_datasets:
            print(f"❌ Dataset '{dataset_name}' not found in {args.dataset_config}. Skipping...")
            continue

        print(f"\n--- Processing Dataset: {dataset_name} ---")
        data = load_benchmark(dataset_name, all_datasets[dataset_name])
        print(f"✅ Loaded {len(data)} samples from dataset: {dataset_name}")

        # Prepare output directory for this specific dataset
        output_base = os.path.join(args.output_dir, f"{gen_model_name}_judge_by_{eval_model_name}", dataset_name)
        os.makedirs(output_base, exist_ok=True)

        # =============================
        # Step 1: Generate responses using apply_chat_template with enable_thinking=False
        # =============================
        print("\n🔍 Generating responses with Generator Model (thinking disabled)...")
        gen_tokenizer = AutoTokenizer.from_pretrained(args.generator_model, trust_remote_code=True)
        gen_llm = LLM(
            model=args.generator_model,
            tokenizer=args.generator_model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=0.95,
            max_num_seqs=32,
            trust_remote_code=True,
        )
        gen_sampling = SamplingParams(temperature=0.0, max_tokens=args.max_gen_tokens, seed=42)
        # think调温度
        # gen_sampling = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768, seed=42)
        gen_template_path = args.generate_template_path
        with open(gen_template_path, encoding="utf-8") as f:
            gen_template = f.read().strip()

        # Prepare prompts using apply_chat_template with enable_thinking=False
        prompts_gen = []
        for item in data:
            filled_prompt = gen_template.format(problem=item["problem"])
            messages = [{"role": "user", "content": filled_prompt}]
            prompt = apply_chat_template(gen_tokenizer, messages, enable_thinking=False) 
            prompts_gen.append(prompt)

        # Add tqdm progress bar for generation
        gen_outputs = gen_llm.generate(prompts_gen, sampling_params=gen_sampling, use_tqdm=True) # vLLM handles this
        generated_responses = [out.outputs[0].text for out in gen_outputs]

        # Save generation results
        gen_file = os.path.join(output_base, "generated.jsonl")
        with open(gen_file, "w", encoding="utf-8") as f:
            for item, resp in tqdm(zip(data, generated_responses), total=len(data), desc="Saving Gen Results"):
                item = item.copy()
                item["model_response"] = resp
                item["extracted_prediction"] = extract_answer(resp)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ Generated responses saved to: {gen_file}")

        print("\n🧹 Cleaning up Generator Model memory...")
        del gen_llm  
        import gc; gc.collect()
        print("✅ Generator Model memory marked for release.")

        # =============================
        # Step 2: Evaluate with Evaluator Model using apply_chat_template with enable_thinking=False
        # =============================
        print("\n⚖️ Evaluating answers with Evaluator Model (thinking disabled)...")
        
        print("📚 Reloading generated responses for evaluation...")
        generated_data_path = os.path.join(output_base, "generated.jsonl")
        if not os.path.exists(generated_data_path):
            raise FileNotFoundError(f"Generated data file not found at {generated_data_path}. Cannot proceed with evaluation.")

        generated_data = []
        with open(generated_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                generated_data.append(json.loads(line.strip()))

        eval_tokenizer = AutoTokenizer.from_pretrained(args.evaluator_model, trust_remote_code=True)
        eval_llm = LLM(
            model=args.evaluator_model,
            tokenizer=args.evaluator_model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=0.95,
            max_num_seqs=32,
            trust_remote_code=True,
        )
        eval_sampling = SamplingParams(temperature=0.0, max_tokens=args.max_eval_tokens, seed=42)

        # Load judge template
        template_path = args.judge_template_path
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Judge template not found at {template_path}")
        with open(template_path, encoding="utf-8") as f:
            judge_template = f.read().strip()

        # Prepare prompts for evaluation using apply_chat_template with enable_thinking=False
        prompts_eval = []
        for item in generated_data:
            pred = item["model_response"]
            prompt = judge_template.format(
                problem=item["problem"],
                label=item["label"],
                prediction=pred
            )
            messages = [{"role": "user", "content": prompt}]
            prompt_formatted = apply_chat_template(eval_tokenizer, messages, enable_thinking=False) # <--- Disable thinking for evaluation
            prompts_eval.append(prompt_formatted)

        # Add tqdm progress bar for evaluation
        eval_outputs = eval_llm.generate(prompts_eval, sampling_params=eval_sampling, use_tqdm=True) # vLLM handles this
        eval_results = []

        for i, (item, out) in enumerate(zip(generated_data, eval_outputs)):
            text = out.outputs[0].text.strip().lower()
            is_correct = "yes" in text
            eval_results.append(is_correct)

        # =============================
        # Step 3: Save final results & compute accuracy
        # =============================
        final_file = os.path.join(output_base, "results.jsonl")
        matches = []

        # Open the file once and write all results inside the context manager
        with open(final_file, "w", encoding="utf-8") as f:
            for item, is_correct in tqdm(zip(generated_data, eval_results), total=len(generated_data), desc="Writing Final Results"):
                item = item.copy() # generated_data 中的 item 已经包含了 model_response 和 extracted_prediction
                item["evaluator_judgment"] = is_correct
                item["match"] = is_correct
                matches.append(is_correct)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        accuracy = sum(matches) / len(matches) * 100 if matches else 0.0
        print(f"\n✅ Final Accuracy on {dataset_name}: {accuracy:.2f}%")
        print(f"📁 Full results saved to: {final_file}")
        all_results[dataset_name] = {
            "accuracy": accuracy,
            "total_samples": len(matches),
            "correct_samples": sum(matches)
        }

        print("\n🧹 Cleaning up Evaluator Model memory...")
        del eval_llm  
        import gc; gc.collect()
        print("✅ Evaluator Model memory marked for release.")
        
    print("\n" + "="*50)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*50)
    for ds_name, result in all_results.items():
        print(f"{ds_name}: Accuracy = {result['accuracy']:.2f}% ({result['correct_samples']}/{result['total_samples']})")
    print("="*50)

if __name__ == "__main__":
    main()