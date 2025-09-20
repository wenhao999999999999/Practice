# GenAI Quickstart

## 项目简介
GenAI Quickstart 演示了一个完整的工厂质检智能体解决方案：利用自建 RAG 检索服务提供上下文，调用本地推理服务承载经过 LoRA 微调的 Qwen 模型，最后通过 LangGraph 编排的智能体生成并校验结构化的检验报告。整个流程完全离线可部署，便于在工厂内网快速验证与二次开发。

## 核心功能与实现
- **上下文检索（「rag/」）**：使用 Haystack 组件将 Markdown/PDF/TXT 文档切分、清洗并通过「BAAI/bge-m3」向量化，结果写入内存向量库后导出「store.jsonl」。FastAPI 服务「rag/qa_server.py」同时提供 BM25 与向量检索，并用「BAAI/bge-reranker-base」交叉编码器重排，最终返回最相关的 5 段上下文。
- **本地推理服务（「service/local_infer_server.py」）**：加载「Qwen/Qwen2.5-7B-Instruct」基座模型与可选 LoRA 适配器（默认路径「models/qwen2.5-7b-lora」），通过 4bit 量化结合「peft」减少显存占用，并暴露兼容 OpenAI Chat Completions 的 REST 接口，支持流式生成。
- **质检报告智能体（「agent/graph_main.py」）**：基于 LangGraph 的 StateGraph 搭建三段式流程：
  1. 节点「retrieve」调用「/ask」接口获取检索上下文；
  2. 节点「generate_report」向本地推理服务发起请求，提示词中约束字段、推理规则与输出格式；
  3. 节点「validate_json」用 JSON Schema 验证输出，如有缺项则调用「_infer_from_rules」进行规则兜底并重试校验。
  节点间状态通过 TypedDict 串联，同时提供两次 LLM 调用重试与全规则兜底的容灾策略。
- **训练配置（「train/llamafactory/」）**：利用 LLaMA Factory 的 QLoRA 流程在自定义数据集「my_sft.jsonl」上微调，配置文件「qlora_qwen2_5_7b.yaml」定义了局部微调目标层、训练超参数与导出目录，训练产出即为「models/qwen2.5-7b-lora」。

## 原理与流程
整体数据流如下：
    用户提问 → LangGraph.StateGraph
      ├─ retrieve() → RAG 检索服务（返回上下文）
      ├─ generate_report() → 本地 OpenAI 兼容推理服务 → LoRA 模型生成 JSON
      └─ validate_json() → JSON Schema 校验 + 规则兜底 → 最终报告
其中「_infer_from_rules」通过规则化解析温度、阈值与噪音信息，保证严重等级、建议与依据在缺失上下文或 LLM 输出异常时仍可生成可用结果。

## 快速开始
1. **准备 Python 环境**（建议 3.10+）并安装依赖：
        pip install langgraph jsonschema requests fastapi uvicorn[standard] farm-haystack[faiss]
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install transformers accelerate peft bitsandbytes llamafactory
2. **构建检索索引**（默认读取「data/docs」，输出到「rag/index_bge」）：
        python rag/build_index.py --docs_dir data/docs --index_dir rag/index_bge
3. **启动检索服务**：
        uvicorn rag.qa_server:app --host 0.0.0.0 --port 8020
4. **启动本地推理服务**（如需自定义模型路径可改写环境变量）：
        python service/local_infer_server.py --base Qwen/Qwen2.5-7B-Instruct --lora models/qwen2.5-7b-lora --port 9000
5. **运行质检智能体**：
        set OPENAI_API_KEY=sk-local
        set OPENAI_API_BASE=http://127.0.0.1:9000/v1
        python agent/graph_main.py
   输出的「report_json」已满足「data/schema/report.schema.json」中定义的结构与约束。

## 目录结构
    ├─ agent/                 # LangGraph 智能体实现
    │  └─ graph_main.py       # 检索/生成/校验主流程
    ├─ data/
    │  ├─ docs/               # 质检 SOP 等知识文档
    │  └─ schema/             # 输出 JSON Schema
    ├─ models/qwen2.5-7b-lora # LoRA 适配器产物
    ├─ rag/
    │  ├─ build_index.py      # 语料索引脚本
    │  └─ qa_server.py        # 检索 FastAPI 服务
    ├─ service/local_infer_server.py # 本地 OpenAI 兼容推理服务
    └─ train/llamafactory/    # QLoRA 训练配置与数据

## 数据与模型
- **知识库**：「data/docs/sop.md」保存厂内 SOP，索引后用于检索。
- **Schema**：「data/schema/report.schema.json」约束报告字段，确保落地系统可直接消费。
- **训练数据**：「train/llamafactory/datasets/my_sft.jsonl」以指令+输出形式提供结构化报告样例。
- **微调模型**：「models/qwen2.5-7b-lora」为 LoRA 权重，可在「service/local_infer_server.py」中热加载。

## 进阶使用建议
- 调整「_infer_from_rules」可快速扩展新的设备/故障规则，提升鲁棒性。
- 「rag/qa_server.py」的检索参数（如「top_k」）与重排模型可按语料量级调优。
- 通过增量扩充「data/docs/」与「my_sft.jsonl」能持续优化检索召回与生成质量。
- 若部署在 GPU 资源有限的环境，可将推理服务量化到 4bit 并只加载 LoRA，维持本地化运行能力。

## 技术架构设计
**Q4: 整体技术架构是如何设计的？如何将 RAG 检索增强、QLoRA 微调模型和 LangGraph Agent 编排结合起来？**  
**A4:** 系统采用分层式 agentic RAG 架构。检索服务「rag/qa_server.py」负责构建混合索引：同一查询会并行触发 BM25 与「BAAI/bge-m3」向量检索，通过互惠排名融合（RRF）合并候选，再交给「BAAI/bge-reranker-base」交叉编码器重排，得到高质量上下文。生成模块由「service/local_infer_server.py」暴露的 OpenAI 兼容接口承载，核心引擎是经 QLoRA 微调的「Qwen/Qwen2.5-7B-Instruct」（权重位于「models/qwen2.5-7b-lora」）。流程编排由「agent/graph_main.py」基于 LangGraph StateGraph 构建：「retrieve」→「generate_report」→「validate_json」三节点串联，若校验失败会回退到规则兜底逻辑「_infer_from_rules」，实现检索增强、生成和校验的闭环。

**Q5: 为什么检索部分要同时采用 BM25 和稠密向量检索，两者各自的作用是什么？**  
**A5:** BM25 擅长关键词精确匹配，可锁定包含设备编号、故障编码等关键字的片段；向量检索借助「BAAI/bge-m3」嵌入捕捉语义近似，弥补措辞差异造成的漏检。「rag/qa_server.py」中两路检索并行执行，随后由 RRF 融合和交叉编码器重排，以兼顾召回率与精确度，为生成模型提供既相关又全面的证据。

**Q6: 系统用了重排序（rerank）技术，请问是如何实现的，又为什么需要重排序？**  
**A6:** 初步检索后的候选片段通过 SentenceTransformersSimilarityRanker（加载「BAAI/bge-reranker-base」交叉编码器）重新评分排序，仅保留 Top5 供下游使用。重排序能够消除仅凭相似度粗筛产生的噪音，使真正与查询语义契合、信息密度高的片段排在前列，从而提升 LangGraph 智能体生成结果的准确性与稳定性。

**Q7: LangGraph 在这里具体扮演什么角色？能否举例说明 Agent 编排的流程？**  
**A7:** LangGraph 担任流程调度员。「agent/graph_main.py」定义的 StateGraph 以状态字典贯穿三个节点：
1. 「retrieve」调用「/ask」获取上下文；
2. 「generate_report」将上下文与用户问题注入提示，调用本地 LLM 生成 JSON 报告；
3. 「validate_json」用「data/schema/report.schema.json」做校验，不合格时触发「_infer_from_rules」兜底并返回错误标记。  
若最终标记出错，LangGraph 可根据需要扩展回退路径或直接终止流程，保障 Agent 在复杂场景下具备自检与容错能力。
