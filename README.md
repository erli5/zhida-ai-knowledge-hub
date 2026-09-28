# 智答 · AI 知识库问答平台（ZhiDa AI Knowledge Hub）

> 一个面向 **大一学生暑假实习面试** 的综合性 Demo 项目。
> 以"文档智能问答"为主线，把 **AI 应用开发 / 大模型训练 / 云运维 / RAG 开发 / Python 后端开发** 五个目标岗位方向，串成一个结构规范、可独立运行、可直接上传 GitHub 的端到端项目。

---

## 一、项目定位与亮点

本项目不是"玩具脚本"，而是一个**能跑起来的小型产品原型**：用户上传文档 → 系统切片建索引 → 提问时检索相关片段 → 生成带出处引用的答案。

设计上特意做了**分层解耦**，让每个岗位方向都有"独立可讲"的模块：

| 目标岗位方向 | 对应模块 | 关键技术点 |
| --- | --- | --- |
| **RAG 开发** | `rag/` | 文本切片、向量化（TF-IDF / 可选句向量）、向量检索、Prompt 编排、答案合成 |
| **Python 后端开发** | `backend/` | FastAPI、Pydantic、依赖注入、REST 接口、pytest、Docker 化 |
| **AI 应用开发** | `ai_app/` | Streamlit 对话式界面、前后端联调、用户体验设计 |
| **大模型训练** | `llm_training/` | LoRA 微调（PEFT）、数据预处理、Trainer、评估、推理 |
| **云运维** | `deploy/` | Docker / docker-compose、Nginx 反向代理、GitHub Actions CI/CD、Prometheus 监控 |

**亮点：**
- ✅ **零依赖即可运行核心 RAG**：`rag/` 用纯 Python 实现了 TF-IDF 向量化与余弦检索，不下载任何大模型也能跑通"检索→回答"全流程（适合面试现场演示，不依赖网络/GPU）。
- ✅ **模块化、可替换**：嵌入模型、向量库、LLM 都做了"接口 + 可选实现"的设计，演示时可以说清楚"生产环境怎么升级"。
- ✅ **工程规范**：统一目录、requirements 分层、pytest 测试、Dockerfile、CI/CD、监控，体现工程素养。
- ✅ **文档齐全**：根 README + `docs/` 三篇说明 + 各模块 README，覆盖"怎么跑、怎么讲、考点在哪"。

---

## 二、技术栈总览

```
检索/语义   纯 Python TF-IDF（默认） · sentence-transformers（可选）
向量存储   NumPy / 内存稀疏向量（可替换为 FAISS / Milvus）
LLM       抽取式答案合成（默认，离线） · HuggingFace / OpenAI 兼容（可选）
后端      Python 3.10+ · FastAPI · Uvicorn · Pydantic · pytest
前端      Streamlit（对话式 Web UI）
训练      PyTorch · Transformers · PEFT(LoRA) · Datasets
运维      Docker · docker-compose · Nginx · GitHub Actions · Prometheus
```

---

## 三、目录结构

```
zhida-ai-knowledge-hub/
├── README.md                     # 项目总览（本文件）
├── LICENSE                       # MIT 许可证
├── .gitignore
├── requirements.txt              # 一键安装全部依赖（开发用）
│
├── docs/                         # 项目说明文档
│   ├── 项目说明.md               # 需求背景、功能清单、使用场景
│   ├── 架构设计.md               # 系统架构图与模块交互
│   └── 面试技术要点.md           # 五方向考点映射 + 可能被问到的问题
│
├── rag/                          # 【RAG 开发】核心检索增强生成模块（纯 Python 可运行）
│   ├── chunker.py                # 文本切片
│   ├── embedder.py               # 向量化（TF-IDF / 可选句向量）
│   ├── vector_store.py           # 向量存储与余弦检索
│   ├── retriever.py              # 检索器（建索引 + 查询）
│   ├── llm.py                    # 答案合成（抽取式 / 可选真实 LLM）
│   ├── pipeline.py               # RAG 端到端流水线
│   ├── example.py                # 可运行 Demo（零依赖）
│   ├── requirements.txt
│   └── data/sample_docs/         # 示例文档
│
├── backend/                      # 【Python 后端】FastAPI 服务
│   ├── app/
│   │   ├── main.py               # 应用入口
│   │   ├── config.py             # 配置
│   │   ├── core/path.py          # 路径/模块加载
│   │   ├── models/schemas.py     # Pydantic 模型
│   │   ├── api/routes.py         # 路由
│   │   └── services/rag_service.py  # 业务服务（集成 rag）
│   ├── tests/test_api.py         # 接口测试
│   ├── requirements.txt
│   └── Dockerfile
│
├── ai_app/                       # 【AI 应用开发】Streamlit 前端
│   ├── app.py                    # 对话式问答界面
│   ├── requirements.txt
│   └── README.md
│
├── llm_training/                 # 【大模型训练】LoRA 微调
│   ├── finetune_lora.py          # 微调脚本
│   ├── inference.py              # 推理脚本
│   ├── configs/lora_config.yaml  # LoRA 配置
│   ├── data/sample.jsonl         # 示例训练数据
│   ├── requirements.txt
│   └── README.md
│
└── deploy/                       # 【云运维】部署与监控
    ├── docker-compose.yml        # 一键编排（后端 + 前端 + nginx）
    ├── nginx/nginx.conf           # 反向代理
    ├── monitoring/prometheus.yml # 监控配置
    └── README.md
│
└── .github/workflows/            # 【云运维】CI/CD（必须位于仓库根）
    ├── ci.yml                    # CI：语法检查 + pytest
    └── cd.yml                    # CD：构建并推送镜像到 GHCR
```

---

## 四、快速开始

### 0. 环境准备
- Python 3.10+
- Git
- （可选）Docker Desktop

### 1. 运行 RAG 核心 Demo（零依赖，推荐先跑这个）
```bash
cd rag
python example.py
```
无需安装任何第三方包，即可看到"上传文档 → 提问 → 带出处回答"的完整效果。

### 2. 启动后端 API（FastAPI）
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# 打开 http://127.0.0.1:8000/docs 查看交互式接口文档
```

### 3. 启动 AI 应用界面（Streamlit）
```bash
cd ai_app
pip install -r requirements.txt
streamlit run app.py
```

### 4. 大模型 LoRA 微调（需 GPU/较大内存，演示用）
```bash
cd llm_training
pip install -r requirements.txt
python finetune_lora.py --config configs/lora_config.yaml
python inference.py --model ./output --prompt "你好，请介绍一下你自己"
```

### 5. 一键容器化部署（云运维）
```bash
cd deploy
docker compose up --build
# 访问 http://localhost 即可（nginx 反向代理前端与后端）
```

---

## 五、如何上传到 GitHub

```bash
git init
git add .
git commit -m "feat: 初始化智答 AI 知识库问答平台 Demo"
git branch -M main
git remote add origin https://github.com/<你的用户名>/zhida-ai-knowledge-hub.git
git push -u origin main
```

> 本项目已内置 `.gitignore`（忽略 `__pycache__`、`*.pyc`、虚拟环境、模型权重等大文件），
> 不会把无关文件提交上去，可直接 push。

---

## 六、面试怎么讲（速记）

- **RAG**：能讲清"切片粒度怎么选、向量化为什么先用 TF-IDF 兜底、检索召回怎么评估、如何避免幻觉（答案必须引用原文）"。
- **后端**：能讲清"FastAPI 异步模型、Pydantic 校验、依赖注入、接口如何限流/鉴权（可扩展点）"。
- **AI 应用**：能讲清"前端怎么和后端联调、加载态/错误处理/引用展示这些体验细节"。
- **大模型训练**：能讲清"LoRA 为什么省显存（只训低秩矩阵）、数据格式、过拟合怎么看、评估指标"。
- **云运维**：能讲清"容器化解决了什么问题、nginx 为什么放前面、CI 保证什么、监控看哪些指标"。

更完整的考点清单见 [`docs/面试技术要点.md`](docs/面试技术要点.md)。

---

## 七、许可证

[MIT](LICENSE) — 可自由学习、修改、二次发布，注明出处即可。
