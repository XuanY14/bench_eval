# analyze_results.py
import os
import json
import re

# 👇 在这里设置你要统计的关键词
TARGET_WORDS = ["check", "wait", "maybe", "but", "Solution:"]

def calculate_accuracy_and_keyword_ratios(results_file_path, target_words):
    total = 0
    correct = 0
    word_present_counts = {word: 0 for word in target_words}
    total_response_length = 0  # 新增：累计所有 response 的字符长度

    # 普通词用单词边界
    normal_words = [w for w in target_words if w != "Solution:"]
    patterns = {
        word: re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        for word in normal_words
    }

    with open(results_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            total += 1
            if item.get("match", False):
                correct += 1

            response = item.get("model_response", "")
            if not isinstance(response, str):
                response = ""

            # 累加响应长度（按字符）
            total_response_length += len(response)

            # 处理普通词
            for word in normal_words:
                if patterns[word].search(response):
                    word_present_counts[word] += 1

            # 特殊处理 "Solution:"
            if "Solution:" in response:
                word_present_counts["Solution:"] += 1

    avg_length = total_response_length / total if total > 0 else 0.0
    return correct, total, word_present_counts, avg_length  # 返回平均长度


def main():
    output_dir = "/mnt/DataFlow/wyz/bench_eval/output/Qwen3-8B-merged-all-10-qa-math-phy-chem-bio-eng-less-lr-all/merged_all_10_qa_math_phy_chem_bio_eng_less_lr_judge_by_gpt5mini_api"
    
    print(f"🔍 Analyzing results in: {output_dir}")
    print("=" * 120)

    for dataset_name in sorted(os.listdir(output_dir)):
        dataset_path = os.path.join(output_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        results_file = os.path.join(dataset_path, "results.jsonl")
        if os.path.exists(results_file):
            correct, total, present_counts, avg_len = calculate_accuracy_and_keyword_ratios(results_file, TARGET_WORDS)
            accuracy = (correct / total) * 100 if total > 0 else 0.0

            # 计算每个词的“包含比例”
            ratios_str = ", ".join(
                f"{word}={present_counts[word]/total*100:>5.1f}%" 
                for word in TARGET_WORDS
            )
            
            # 格式化输出：加入 AvgRespLen
            print(f"{dataset_name:<30} : Acc={accuracy:>5.2f}% ({correct}/{total}) | "
                  f"AvgRespLen={avg_len:>6.1f} | Keyword Ratios: {ratios_str}")
        else:
            print(f"{dataset_name:<30} : ❌ No results.jsonl found")

    print("=" * 120)


if __name__ == "__main__":
    main()