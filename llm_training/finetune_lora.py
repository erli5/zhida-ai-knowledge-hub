"""大模型 LoRA 微调脚本（演示完整 SFT 流水线）。

技术要点：
- 使用 HuggingFace Transformers + PEFT(LoRA) 做参数高效微调。
- 数据采用 SFT 标准格式：{instruction, input, output}。
- 只训练低秩适配矩阵，显存占用远小于全参微调，适合学生本地/单卡实验。

依赖（见 requirements.txt）：torch, transformers, peft, datasets, accelerate, yaml

运行：
    pip install -r requirements.txt
    python finetune_lora.py --config configs/lora_config.yaml
快速试跑（小数据、少步数）：
    python finetune_lora.py --config configs/lora_config.yaml --max_steps 20
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import List, Optional

import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class Config:
    model_name_or_path: str = "uer/gpt2-chinese-cluecorpussmall"
    data_path: str = "./data/sample.jsonl"
    output_dir: str = "./output"
    max_seq_length: int = 256
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 5
    save_strategy: str = "epoch"
    fp16: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["c_attn"])
    seed: int = 42
    max_steps: Optional[int] = None
    max_train_samples: Optional[int] = None


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(**raw)


def build_prompt(instruction: str, input_text: str, output: str = "") -> str:
    """构造训练/推理用的 prompt 模板。output 为空表示仅构造输入部分。"""
    if input_text:
        text = f"下面是一项任务。\n指令：{instruction}\n输入：{input_text}\n回答："
    else:
        text = f"下面是一项任务。\n指令：{instruction}\n回答："
    if output:
        text += output
    return text


def load_dataset(cfg: Config) -> Dataset:
    """读取 JSONL，构造 input_ids / attention_mask / labels。"""
    records = []
    with open(cfg.data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = build_prompt(obj.get("instruction", ""), obj.get("input", ""), obj.get("output", ""))
            records.append({"text": prompt})
            if cfg.max_train_samples and len(records) >= cfg.max_train_samples:
                break

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            max_length=cfg.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    ds = Dataset.from_list(records)
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 微调 Demo")
    parser.add_argument("--config", type=str, default="configs/lora_config.yaml")
    parser.add_argument("--max_steps", type=int, default=None, help="覆盖配置：最大训练步数（调试用）")
    parser.add_argument("--max_train_samples", type=int, default=None, help="仅使用前 N 条样本（调试用）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.max_train_samples is not None:
        cfg.max_train_samples = args.max_train_samples

    set_seed(cfg.seed)

    print(f"[1/4] 加载基座模型：{cfg.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
    )

    print("[2/4] 注入 LoRA 适配器")
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[3/4] 准备数据")
    dataset = load_dataset(cfg)
    print(f"      训练样本数：{len(dataset)}")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        fp16=cfg.fp16,
        report_to="none",
        max_steps=cfg.max_steps if cfg.max_steps else -1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("[4/4] 开始训练")
    trainer.train()

    print(f"保存 LoRA 适配器到：{cfg.output_dir}")
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("训练完成。")


if __name__ == "__main__":
    main()
