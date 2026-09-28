# 大模型训练模块（LoRA 微调）

本模块演示**大模型参数高效微调（LoRA）** 的完整流水线，对应目标岗位方向"大模型训练"。
它独立于 RAG 核心模块，用来体现你对"训练"而非"只会调 API"的理解。

## 为什么是 LoRA？

全参数微调需要更新模型全部权重，显存与算力门槛高。LoRA（Low-Rank Adaptation）的核心思想：

> 不直接修改原始权重矩阵 W，而是为其旁路增加两个低秩矩阵 A、B（W + BA），
> 训练时只更新 A、B，原始权重冻结。

好处：可训练参数量大幅减少、显存占用低、训练快；同一基座可挂载多个适配器；降低灾难性遗忘风险。

## 目录结构

```
llm_training/
├── finetune_lora.py        # 微调主脚本
├── inference.py            # 训练后推理
├── configs/lora_config.yaml# 训练超参与 LoRA 配置
├── data/sample.jsonl       # 示例 SFT 数据（instruction/input/output）
└── requirements.txt
```

## 数据格式（SFT）

每行一个 JSON，字段：`instruction`（指令）、`input`（可选输入）、`output`（期望回答）。

```json
{"instruction": "请用一句话解释什么是 RAG", "input": "", "output": "RAG 是检索增强生成..."}
```

## 运行

```bash
pip install -r requirements.txt
python finetune_lora.py --config configs/lora_config.yaml
# 快速试跑（少样本、少步数，验证流程）
python finetune_lora.py --config configs/lora_config.yaml --max_steps 20 --max_train_samples 5
# 推理
python inference.py --model ./output --prompt "请用一句话解释什么是 RAG"
```

## 面试可讲的点

- LoRA 为什么省显存、可插拔；`r` / `alpha` / `dropout` 各自作用。
- 训练流程：数据预处理 → Tokenize → LoraConfig → get_peft_model → Trainer → 评估 → 保存。
- 如何判断是否过拟合：对比 train / eval loss，看生成样例质量。
- 与 RAG 的边界：微调用于"注入知识/风格"，RAG 用于"动态检索最新事实"。
