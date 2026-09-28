"""LoRA 微调后的推理脚本。

加载基座模型 + 训练好的 LoRA 适配器，对给定 prompt 生成回答。

运行：
    python inference.py --model ./output --prompt "请用一句话解释什么是 RAG"
"""
from __future__ import annotations

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 推理 Demo")
    parser.add_argument("--base_model", type=str, default="uer/gpt2-chinese-cluecorpussmall")
    parser.add_argument("--model", type=str, default="./output", help="LoRA 适配器目录")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"加载基座模型：{args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", device_map="auto")

    print(f"加载 LoRA 适配器：{args.model}")
    model = PeftModel.from_pretrained(base, args.model)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n=== 生成结果 ===")
    print(text)


if __name__ == "__main__":
    main()
