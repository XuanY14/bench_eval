# run_eval.py
import argparse
import os
import json
import re
from tqdm import tqdm 
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils import load_dataset_config, load_benchmark, extract_answer
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def apply_chat_template(tokenizer, messages, enable_thinking=False):
    """Apply chat template with enable_thinking option."""
    return tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

def create_openai_client(api_key, base_url="http://172.96.160.199:3000/v1"):
    return OpenAI(
        base_url=base_url,
        api_key=api_key
    )

def call_gpt5mini_api(client, messages, max_tokens=32, temperature=0.0):
    try:
        completion = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        print(f"❌ API Error: {e}")
        return ""

def parse_judgment(text):
    """Parse the judgment from API response"""
    t = text.strip().lower()
    if "yes" in t:
        return True
    elif "no" in t:
        return False
    else:
        print(f"❓ Ambiguous: '{text}' → False")
        return False

def process_evaluation_item(client, item, judge_template, max_tokens=100000, temperature=0.0, delay=0.05):
    """Process a single evaluation item using API"""
    # Format the prompt using judge template
    prompt = judge_template.format(
        problem=item["problem"],
        label=item["label"],
        prediction=item["model_response"]
    )
    messages = [{"role": "user", "content": prompt}]
    
    # Add request delay to prevent server overload
    time.sleep(delay)
    
    # Call API
    raw_resp = call_gpt5mini_api(client, messages, max_tokens, temperature)
    is_correct = parse_judgment(raw_resp)
    
    # Update item with evaluation results
    out_item = item.copy()
    out_item["evaluator_judgment_raw"] = raw_resp
    out_item["evaluator_judgment"] = is_correct
    out_item["match"] = is_correct
    
    return out_item, is_correct

def evaluate_with_api(generated_data, judge_template, api_key, base_url="http://172.96.160.199:3000/v1", concurrent_threads=100, max_tokens=100000, temperature=0.0, delay=0.05):
    """Evaluate using GPT-5-Mini API with concurrency"""
    print(f"\n⚖️ Evaluating answers with GPT-5-Mini API...")
    print(f"🚀 Using {concurrent_threads} concurrent threads for API calls...")
    
    # Create a single client instance
    client = create_openai_client(api_key, base_url)
    
    results = []
    matches = []
    
    # Use ThreadPoolExecutor for concurrent API calls
    with ThreadPoolExecutor(max_workers=concurrent_threads) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_evaluation_item, client, item, judge_template, max_tokens, temperature, delay): i 
            for i, item in enumerate(generated_data)
        }
        
        # Use tqdm to show progress
        with tqdm(total=len(generated_data), desc="API Evaluation Progress") as pbar:
            for future in as_completed(future_to_index):
                try:
                    result_item, is_correct = future.result()
                    idx = future_to_index[future]
                    results.append((idx, result_item, is_correct))
                    matches.append(is_correct)
                except Exception as e:
                    print(f"❌ Error processing item: {e}")
                
                pbar.update(1)

    # Sort results by original index to maintain order
    results.sort(key=lambda x: x[0])
    sorted_items = [item for _, item, _ in results]

    return sorted_items, matches

def main():
    parser = argparse.ArgumentParser(description="Evaluate a generator model using an evaluator LLM.")
    # Generator Model
    parser.add_argument("--generator_model", type=str, required=True, help="Path to the generator model")
    # Evaluator API (GPT-5-Mini) - This will be ignored since we're using API
    parser.add_argument("--evaluator_model", type=str, default="gpt-5-mini", help="Evaluator will use GPT-5-Mini API")
    # Data
    parser.add_argument("--dataset", type=str, nargs='+', required=True, help="Name(s) of the dataset(s) (e.g., gsm8k, math, omnimath olympiadbench)")
    parser.add_argument("--dataset_config", type=str, default="configs/datasets.yaml", help="Path to dataset config YAML")
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save results")
    # vLLM args for generator only
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for generator")
    parser.add_argument("--max_gen_tokens", type=int, default=4096, help="Max tokens for generator")
    parser.add_argument("--max_eval_tokens", type=int, default=100000, help="Max tokens for evaluator API")
    # Template args
    parser.add_argument("--generate_template_path", type=str, default="templates/generate_template.txt", help="Path to the generation prompt template")
    parser.add_argument("--judge_template_path", type=str, default="templates/judge_template.txt", help="Path to the judge prompt template")
    # API settings
    parser.add_argument("--api_key", type=str, required=True, help="API Key for GPT-5-Mini API")
    parser.add_argument("--api_base_url", type=str, default="http://172.96.160.199:3000/v1", help="Base URL for GPT-5-Mini API")
    parser.add_argument("--concurrent_threads", type=int, default=100, help="Number of concurrent threads for API calls")
    parser.add_argument("--request_delay", type=float, default=0.05, help="Delay between API requests (seconds)")
    args = parser.parse_args()

    # Load dataset config
    all_datasets = load_dataset_config(args.dataset_config)

    gen_model_name = os.path.basename(args.generator_model.rstrip("/"))
    eval_model_name = "gpt5mini_api"  # Fixed name for API evaluator

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

        # Read generation template
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

        # Generate responses
        gen_outputs = gen_llm.generate(prompts_gen, sampling_params=gen_sampling, use_tqdm=True)
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
        # Step 2: Evaluate with GPT-5-Mini API
        # =============================
        print("📚 Reloading generated responses for evaluation...")
        generated_data_path = os.path.join(output_base, "generated.jsonl")
        if not os.path.exists(generated_data_path):
            raise FileNotFoundError(f"Generated data file not found at {generated_data_path}. Cannot proceed with evaluation.")

        generated_data = []
        with open(generated_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                generated_data.append(json.loads(line.strip()))

        # Load judge template
        template_path = args.judge_template_path
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Judge template not found at {template_path}")
        with open(template_path, encoding="utf-8") as f:
            judge_template = f.read().strip()

        # Evaluate using API with concurrency
        evaluated_data, matches = evaluate_with_api(
            generated_data, 
            judge_template,
            api_key=args.api_key,
            base_url=args.api_base_url,
            concurrent_threads=args.concurrent_threads,
            max_tokens=args.max_eval_tokens,
            temperature=0.0,
            delay=args.request_delay
        )

        # =============================
        # Step 3: Save final results & compute accuracy
        # =============================
        final_file = os.path.join(output_base, "results.jsonl")
        
        # Write all evaluated results to the final file
        with open(final_file, "w", encoding="utf-8") as f:
            for item in tqdm(evaluated_data, total=len(evaluated_data), desc="Writing Final Results"):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        accuracy = sum(matches) / len(matches) * 100 if matches else 0.0
        print(f"\n✅ Final Accuracy on {dataset_name}: {accuracy:.2f}%")
        print(f"📁 Full results saved to: {final_file}")
        
        all_results[dataset_name] = {
            "accuracy": accuracy,
            "total_samples": len(matches),
            "correct_samples": sum(matches),
            "results_file": final_file  # Include path to results file
        }

        # Save individual dataset summary
        summary_file = os.path.join(output_base, "summary.json")
        dataset_summary = {
            "dataset": dataset_name,
            "model": gen_model_name,
            "evaluator": eval_model_name,
            "accuracy": accuracy,
            "total_samples": len(matches),
            "correct_samples": sum(matches),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "concurrent_threads": args.concurrent_threads,
                "request_delay": args.request_delay,
                "max_eval_tokens": args.max_eval_tokens,
                "api_base_url": args.api_base_url
            }
        }
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(dataset_summary, f, ensure_ascii=False, indent=2)
        print(f"📋 Dataset summary saved to: {summary_file}")
        
    # =============================
    # Step 4: Save overall summary across all datasets
    # =============================
    overall_summary_dir = os.path.join(args.output_dir, f"{gen_model_name}_judge_by_{eval_model_name}")
    os.makedirs(overall_summary_dir, exist_ok=True)
    
    overall_summary_file = os.path.join(overall_summary_dir, "overall_summary.json")
    
    overall_summary = {
        "evaluation_config": {
            "generator_model": args.generator_model,
            "evaluator_model": eval_model_name,
            "datasets": args.dataset,
            "output_dir": args.output_dir,
            "api_base_url": args.api_base_url,
            "concurrent_threads": args.concurrent_threads,
            "request_delay": args.request_delay,
            "max_gen_tokens": args.max_gen_tokens,
            "max_eval_tokens": args.max_eval_tokens,
            "generate_template": args.generate_template_path,
            "judge_template": args.judge_template_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "results_by_dataset": all_results,
        "overall_statistics": {
            "total_datasets": len(all_results),
            "average_accuracy": sum(result['accuracy'] for result in all_results.values()) / len(all_results) if all_results else 0.0
        }
    }
    
    with open(overall_summary_file, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    print(f"\n📋 Overall summary saved to: {overall_summary_file}")
    
    print("\n" + "="*50)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*50)
    for ds_name, result in all_results.items():
        print(f"{ds_name}: Accuracy = {result['accuracy']:.2f}% ({result['correct_samples']}/{result['total_samples']})")
    print(f"\n📈 Average Accuracy: {overall_summary['overall_statistics']['average_accuracy']:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()