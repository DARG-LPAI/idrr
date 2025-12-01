import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from IDRR_data import IDRRDataFrames
from utils.utils import read_file, re_search

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

SYSTEM_PROMPT = f"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{{}}."
ANSWER_MAP = {
    "Comparison": "A",
    "Contingency": "B",
    "Expansion": "C",
    "Temporal": "D"
}
# CKPT_PATH = "/data/whsun/pretrained_models/Qwen/Qwen2.5-1.5B-Instruct"
# CKPT_PATH = "expt/rl_cold_start/qwen3-0.6B/epo1/merged-qwen3-0.6B"
# CKPT_PATH = "/data/whsun/pretrained_models/Qwen/Qwen3-0.6B"
CKPT_PATH = "/data/whsun/idrr/expt/arg2def/pdtb2/llama3/epo5/merged"
TEST_DATA_PATH = "data/rl_cold_start/pdtb2/top/sft_rl_test.jsonl"

def read_parquet(file_path):
    import pandas as pd
    df = pd.read_parquet(file_path)
    return df.to_dict(orient='records')

# -------------------------------
# 新增：通用数据加载（alpaca）
# -------------------------------
def load_alpaca_data(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if p.suffix.lower() == ".jsonl":
        items = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    elif p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                return data["data"]
            if isinstance(data, list):
                return data
            raise ValueError("Unsupported JSON structure for alpaca data.")
    else:
        # 回退到项目内的 read_file（若其能处理该格式）
        return list(read_file(path))

# -------------------------------
# 新增：构建 alpaca prompts
# -------------------------------
def build_alpaca_prompts(
    data_path: str,
    ckpt_path: str,
    add_system_prompt: bool = False
) -> Tuple[List[str], List[Dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    items = load_alpaca_data(data_path)
    prompts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for i, item in enumerate(items):
        instruction = item.get("instruction", "")
        input_ = item.get("input", "")
        # 兼容某些数据集使用 "output" 作为参考答案
        output_ = item.get("output", None)

        messages = []
        if add_system_prompt:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": f"{instruction}\n{input_}"})

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)
        metas.append({
            "idx": i,
            "instruction": instruction,
            "input": input_,
            "output": output_
        })
    return prompts, metas

# -------------------------------
# 兼容：原 verl 格式（保留）
# -------------------------------
def build_prompts(data_format, data_path=TEST_DATA_PATH, ckpt_path=CKPT_PATH):
    prompts = []
    labels = []
    if data_format == 'verl':  # List[dict_keys(['data_source', 'prompt', 'reward_model'])]
        for item in read_parquet(data_path):
            prompts.append(item['prompt'])
            labels.append(item['reward_model']['ground_truth'])
    elif data_format == 'alpaca':  # 与旧接口兼容（返回 labels=None）
        prompts, metas = build_alpaca_prompts(data_path, ckpt_path)
        return prompts, None, metas
    return prompts, labels

# -------------------------------
# 新增：推理与抽取
# -------------------------------
def extract_prediction_from_text(text: str) -> Optional[str]:
    # 优先提取 \boxed{}；若失败尝试 box
    if 'box' in text:
        return re_search(text, type="boxed")
    elif "Relation: " in text:
        return text.split('Relation: ')[0]
    return None

def generate_with_vllm(
    model_path: str,
    prompts: List[str],
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 2048
) -> List[Dict[str, Any]]:
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p)
    llm = LLM(model=model_path)
    outputs = llm.generate(prompts, sampling_params=sampling_params, max_model_len=8192)

    results: List[Dict[str, Any]] = []
    for i, output in enumerate(outputs):
        text = output.outputs[0].text if output.outputs else ""
        results.append({
            "idx": i,
            "prompt": output.prompt,
            "output_text": text
        })
    return results

# -------------------------------
# 新增：保存结果
# -------------------------------
def save_jsonl(rows: List[Dict[str, Any]], out_path: str) -> None:
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def default_out_path(data_path: str, model_path: str) -> str:
    d = Path(data_path)
    m = Path(model_path)
    base = d.stem  # test (from test.json / test.jsonl)
    model_name = m.name.replace("/", "_")
    return str(d.with_name(f"{base}.{model_name}.vllm.pred.jsonl"))

# -------------------------------
# 主入口（仅保存推理结果，不做评估）
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline inference (save-only).")
    parser.add_argument("--data-format", type=str, default="alpaca", choices=["alpaca", "verl"], help="Data format to load.")
    parser.add_argument("--data-path", type=str, default="/data/whsun/idrr/data/arg2def/pdtb2/aplaca/test.json", help="Path to test dataset.")
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH, help="Model checkpoint path.")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--add-system-prompt", action="store_true", help="Whether to prepend SYSTEM_PROMPT in chat messages.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2048)

    args = parser.parse_args()

    if args.data_format == "alpaca":
        prompts, metas = build_alpaca_prompts(
            data_path=args.data_path,
            ckpt_path=args.ckpt,
            add_system_prompt=args.add_system_prompt
        )
        results = generate_with_vllm(
            model_path=args.ckpt,
            prompts=prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        # 合并 meta 与抽取到的 pred
        merged_rows = []
        for r in results:
            idx = r["idx"]
            meta = metas[idx] if idx < len(metas) else {}
            pred = extract_prediction_from_text(r["output_text"])
            merged_rows.append({
                "idx": idx,
                "prompt": r["prompt"],
                "output_text": r["output_text"],
                "pred": pred,
                "meta": meta
            })
        out_path = args.out or default_out_path(args.data_path, args.ckpt)
        save_jsonl(merged_rows, out_path)
        print(f"Saved {len(merged_rows)} predictions to: {out_path}")
    else:
        # 保留对 VERL 的兼容：同样只做推理+保存
        prompts, labels = build_prompts("verl", data_path=args.data_path, ckpt_path=args.ckpt)
        results = generate_with_vllm(
            model_path=args.ckpt,
            prompts=prompts,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        rows = []
        for i, r in enumerate(results):
            pred = extract_prediction_from_text(r["output_text"])
            rows.append({
                "idx": i,
                "prompt": r["prompt"],
                "output_text": r["output_text"],
                "pred": pred,
                "meta": {"label": labels[i] if labels and i < len(labels) else None}
            })
        out_path = args.out or default_out_path(args.data_path, args.ckpt)
        save_jsonl(rows, out_path)
        print(f"Saved {len(rows)} predictions to: {out_path}")