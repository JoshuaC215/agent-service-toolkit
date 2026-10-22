# AgentOps 售后政策 RAG

本模块检索 `knowledge/after_sales/` 中的演示政策。它提供证据，不审批或执行退款。

## 调用关系

`order-agent → search_after_sales_policy → PolicyRAG → Embedding + 本地 Chroma`

订单查询和工单工具继续使用 PostgreSQL，Chroma 不修改业务表或 LangGraph 检查点。

## 配置与运行

聊天模型与 Embedding 独立配置。默认采用本地 CPU 上的 `BAAI/bge-small-zh-v1.5`，
继续保留现有 DeepSeek 聊天配置。本地向量化不调用收费 Embedding API，
但 DeepSeek 生成回答仍按原账号规则计费。
首次需要联网下载依赖和模型；不需要训练，也不需要新的 API Key。

默认值已设置为本地模式。如需显式配置，可在项目 `.env` 中添加或更新以下非敏感项目：

```dotenv
POLICY_EMBEDDING_PROVIDER=local
POLICY_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
POLICY_LOCAL_CACHE_DIR=data/embedding_models
POLICY_MIN_SCORE=0.55
POLICY_TOP_K=3
```

首次安装和下载模型，在 Windows PowerShell 的项目根目录运行：

```powershell
Set-Location C:\Projects\agent-service-toolkit
uv sync --extra local-rag
uv run --extra local-rag python scripts/setup_policy_model.py
```

`uv sync` 会解析本地模型的可选依赖、更新 `uv.lock` 并安装 PyTorch 等运行库，
下载可能耗时。依赖锁更新完成后才能使用 Docker 的 `--frozen` 构建。

模型固定为官方仓库修订 `7999e1d3359715c523056ef9478215996d62a620`。
初始化脚本显式允许下载；正常检索只加载本地缓存，不会自动联网下载或切换收费服务。
缓存不存在时会返回检索不可用。查询添加中文检索指令，文档不添加指令，向量均归一化。
本地模型上限为 512 token，本地政策采用 400 字符切分、60 字符重叠，减少长段截断。
缓存目录不进入 Git 或 Docker 构建上下文。

云端接口仍保留：OpenAI 使用 `POLICY_EMBEDDING_PROVIDER=openai`、
`POLICY_EMBEDDING_MODEL=text-embedding-3-small` 和 `OPENAI_API_KEY`；
Google 使用 `POLICY_EMBEDDING_PROVIDER=google`、
`POLICY_EMBEDDING_MODEL=gemini-embedding-001` 和 `GOOGLE_API_KEY`。
这些只有显式选用时才会启用，不会复用 DeepSeek 密钥。

在 Windows PowerShell 中从项目根目录运行离线测试：

```powershell
Set-Location C:\Projects\agent-service-toolkit
uv run pytest tests/agents/test_policy_rag.py tests/agents/test_local_embeddings.py -q --tb=short
```

本地模型准备好后，可执行以下命令建立索引并检查真实中文检索。
默认本地模式不向 Embedding 云服务发送政策或问题，不产生 Embedding API 调用费用。

```powershell
uv run --extra local-rag python -c "import sys; sys.path.insert(0, 'src'); from agents.policy_tools import get_policy_rag; rag = get_policy_rag(); print('chunks:', rag.build_index()); print(rag.search('质量问题退款需要什么材料？'))"
```

首次检索也会自动建立索引；后续查询复用文档向量，只对问题生成向量。
默认索引目录 `data/policy_chroma/` 已被 Git 忽略。
Docker 镜像包含政策文档，默认索引由独立的 `policy_chroma_data` 卷保存。
首次在宿主机下载好模型后，Compose 以只读方式挂载 `data/embedding_models/`。
直接本机运行 API 服务时使用 `uv run --extra local-rag python src/run_service.py`。
修改 Embedding 配置后需重启服务，修改镜像中的政策需重新构建镜像。

## 版本与错误处理

文件内容、文件名、Embedding 服务与模型、切分参数共同生成索引版本。
新增、修改或删除政策后，下次检索会构建或复用对应版本，旧版本不会被检索。
构建失败时不使用旧版本兜底，下次请求重试；部分写入通过确定性块 ID 幂等覆盖。
同一进程的索引构建与检索由锁保护。历史版本暂保留在磁盘，第一版不自动清理。
多个服务进程或多台机器的共享写入未做专项验证；第一版使用单进程本地 Chroma。

每个结果包含 `source`、`title`、`chunk_id`、`content`、`score`。
分数为余弦相似度，不是退款资格或审批通过率。当前阈值 0.55 根据六个政策问题和
四个无关问题的本地 BGE 检索分数初步校准：相关问题的最高分约 0.57–0.74，
无关问题最高分约 0.24–0.42。它只适用于当前演示语料的初步验证，不是通用阈值或正式评测结果。
退款材料问题还需保留约 0.62 的申请材料片段，不能仅返回排名更高的人工审批片段。
低于阈值时返回 `no_evidence`，无文档、配置或服务错误返回 `unavailable`。
Agent 被要求标注演示政策、引用来源，并在证据缺失或冲突时建议人工核实。
系统提示不能替代退款审批和权限控制；当前没有新增退款执行工具。

## 验收范围

离线测试使用真实 Chroma 和确定性主题向量，验证加载、来源、检索过滤、持久化复用、
文档更新与删除、模型变更、失败重试、工具错误和已有工具保留。
本地适配器另外测试 CPU、固定修订、默认禁用下载、查询指令、归一化以及不需要云端密钥。
测试向量仅用于测试，不作为生产回退方案。离线通过不证明真实 Embedding 的中文语义效果。
真实向量服务、模型是否遵循引用指令，以及订单查询和工单协作的端到端验收另行进行。

模型下载完成后，可显式运行真实本地模型的检索回归测试：

```powershell
uv run --extra local-rag pytest tests/integration/test_policy_rag_local.py --run-local-rag -q --tb=short
```

这些测试不调用聊天模型或 Embedding API，验证六个政策问题的来源和关键条款，
以及四个无关问题返回空证据。默认全量测试跳过它们，不要求其他开发者预先下载模型。

## DeepSeek 端到端验收

本地模型准备好、现有 `DEEPSEEK_API_KEY` 可用后，在项目根目录运行：

```powershell
uv run --extra local-rag python scripts/verify_policy_agent.py
```

脚本使用真实 DeepSeek API（按原账号规则计费），复用现有订单 Agent 的模型节点、
提示词和工具节点。测试会话使用内存检查点，不读写 PostgreSQL 检查点。
在每次执行工具前暂停：只允许 `search_after_sales_policy`，其他工具请求直接阻断，
因此本次政策咨询不会创建工单或查询业务数据库。最多执行三轮政策工具调用，
整个 Agent 调用默认超时 90 秒，可用 `--timeout` 调整。

输出包括模型工具请求、实际政策证据、DeepSeek 最终回答和结构检查。
检查要求调用检索、匹配工具结果、引用退款政策文件名和标题、标注演示政策、
提及申请材料及人工审批。所有结构检查通过不代表回答的每句话均已获得政策支持，
仍须人工逐条比对显示的证据。不要把仅包含工具调用的中间消息当作最终回答。

退出码 0 表示结构检查通过，1 表示至少一项不符合预期，2 表示配置或运行错误。
脚本不会输出密钥。网络连接失败不等于 Agent 功能验证失败，需在能访问 DeepSeek 的终端重试。
