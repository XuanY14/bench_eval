# utils.py
import re
import json
import yaml
from pathlib import Path
from datasets import load_dataset
import ast

def extract_answer(text: str):
    if not isinstance(text, str):
        return None
    boxed = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed:
        return boxed[-1].strip()
    numbers = re.findall(r'-?\d+\.?\d*(?:/\d+)?', text)
    if numbers:
        return numbers[-1].strip()
    return None

def load_dataset_config(config_path: str = "configs/datasets.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)["datasets"]

def load_benchmark(name: str, config: dict):
    if config["type"] == "hf":
        hf_config = config.get("config")
        if hf_config:
            ds = load_dataset(config["path"], hf_config, split=config["split"])
        else:
            ds = load_dataset(config["path"], split=config["split"])
    elif config["type"] == "local":
        file_path = Path(config["path"])
        file_extension = file_path.suffix.lower()
        split_name = "train"
        if file_extension in [".jsonl", ".json"]:
            loader = "json"
        elif file_extension == ".parquet":
            loader = "parquet"
        elif file_extension in [".arrow", ".feather"]:
            loader = "arrow" 
        elif file_extension == ".csv":
            loader = "csv"
        elif file_extension == ".txt":
            loader = "text"
        else:
            raise ValueError(f"Unsupported file type for local dataset: {file_extension}. Supported: .json, .jsonl, .parquet, .arrow, .feather, .csv, .txt")
        
        ds = load_dataset(loader, data_files=str(file_path), split=split_name)
    else:
        raise ValueError("type must be 'hf' or 'local'")

    problem_key = config["problem_key"]
    label_key = config["label_key"]
    label_proc_str = config.get("label_processor", "str")

    try:
        label_processor = eval(label_proc_str)
    except:
        raise ValueError(f"Invalid label_processor: {label_proc_str}")

    data = []
    for item in ds:
        problem = item[problem_key]
        raw_label = item[label_key]
        try:
            label = label_processor(raw_label)
        except Exception as e:
            print(f"⚠️ Failed to process label: {raw_label}, error: {e}")
            label = None
        if label is not None:
            data.append({"problem": problem, "label": str(label)})

    repeat_times = config.get("repeat_times", 1) 
    if repeat_times > 1:
        print(f"🔄 Repeating dataset {name} {repeat_times} times (original size: {len(data)}, new size: {len(data) * repeat_times})")
        repeated_data = []
        for _ in range(repeat_times):
            repeated_data.extend(data) 
        data = repeated_data

    return data