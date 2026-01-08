import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from IDRR_data import IDRRDataFrames
from utils.utils import read_file, re_search, write_file

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

def read_parquet(file_path):
    import pandas as pd
    df = pd.read_parquet(file_path)
    return df.to_dict(orient='records')

# -------------------------------
# 新增：构建 alpaca prompts and labels
# -------------------------------
def build_alpaca_prompts_labels(
    data_path: str,
    ckpt_path: str,
    add_system_prompt: bool = False
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    items = read_file(data_path)
    prompts: List[str] = []
    labels: List[str] = []
    # metas: List[Dict[str, Any]] = []

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
        labels.append(extract_prediction_from_text(output_) if output_ else None)
    return prompts, labels

# -------------------------------
# 兼容：原 verl 格式（保留）
# -------------------------------
def build_prompts(data_format, data_path, ckpt_path):
    prompts = []
    labels = []
    if data_format == 'verl':  # List[dict_keys(['data_source', 'prompt', 'reward_model'])]
        for item in read_parquet(data_path):
            prompts.append(item['prompt'][0]['content'])
            labels.append(item['reward_model']['ground_truth'])
    elif data_format == 'alpaca':  # 与旧接口兼容（返回 labels=None）
        prompts, metas = build_alpaca_prompts_labels(data_path, ckpt_path)
        return prompts, None, metas
    return prompts, labels

# -------------------------------
# 新增：推理与抽取
# -------------------------------
def extract_prediction_from_text(text: str) -> Optional[str]:
    # 优先提取 \boxed{}；若失败尝试 box
    if 'box' in text:
        return re_search(text, type="box")
    elif "Relation: " in text:
        return text.split('Relation: ')[0]
    return None

def generate_with_vllm(
    model_path: str,
    prompts: List[str],
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    gpu_memory_utilization: float = 0.75,
    max_tokens: int = 1024,
    max_model_len: int = 4096,
    max_num_seqs: int = 128,
    ) -> List[Dict[str, Any]]:
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )
    results: List[Dict[str, Any]] = []
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        # dtype="half",
        # enforce_eager=True
    )
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
    )
    for j, output in enumerate(outputs):
        text = output.outputs[0].text if output.outputs else ""
        results.append({
            "idx": j,
            "prompt": output.prompt,
            "output_text": text
        })
    return results


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
    parser.add_argument("--ckpt", type=str, default=None, help="Model checkpoint path.")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--add-system-prompt", action="store_true", default=False, help="Whether to prepend SYSTEM_PROMPT in chat messages.")
    # New inference parameters
    parser.add_argument("--cutoff_len", type=int, default=1024, help="Maximum context length (used as max_model_len).")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--enable-thinking", action="store_true", default=True, help="Enable reasoning SYSTEM_PROMPT.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.71)

    args = parser.parse_args()

    # 直接使用命令行参数
    max_model_len = args.cutoff_len + args.max_new_tokens

    if args.data_format == "alpaca":
        prompts, labels = build_alpaca_prompts_labels(
            data_path=args.data_path,
            ckpt_path=args.ckpt,
            add_system_prompt=args.add_system_prompt
        )
        # prompts *= 10000
        out_path = args.out or default_out_path(args.data_path, args.ckpt)
        if os.path.exists(out_path):
            results = read_file(out_path)
            for i, row in results.items():
                results[i]['pred'] = extract_prediction_from_text(row["output_text"])
            write_file(data=results, path=out_path)
            print(f"Saved {len(results)} predictions to: {out_path}")
        else:
            results = generate_with_vllm(
                model_path=args.ckpt,
                prompts=prompts,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                max_model_len=max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization
            )
            # 合并 meta 与抽取到的 pred
            data_dict = {}
            for i, r in enumerate(results):
                data_dict[str(i+1)] = {
                    "prompt": r["prompt"],
                    "output_text": r["output_text"],
                }
            write_file(data=data_dict, path=out_path)
            print(f"Saved {len(data_dict)} predictions to: {out_path}")
    else:
        # 保留对 VERL 的兼容：同样只做推理+保存
        prompts, labels = build_prompts("verl", data_path=args.data_path, ckpt_path=args.ckpt)
        results = generate_with_vllm(
            model_path=args.ckpt,
            prompts=prompts,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            max_model_len=max_model_len,
        )
        rows = []
        for i, r in enumerate(results):
            rows.append({
                "idx": i,
                "prompt": r["prompt"],
                "output_text": r["output_text"],
                "meta": {"label": labels[i] if labels and i < len(labels) else None}
            })
        out_path = args.out or default_out_path(args.data_path, args.ckpt)
        write_file(data=rows, path=out_path)
        print(f"Saved {len(rows)} predictions to: {out_path}")