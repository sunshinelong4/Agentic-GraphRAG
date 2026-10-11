# Agentic-GraphRAG

> **鉴权与并发升级**：默认需要登录；管理员维护共享知识库，普通用户查询。
> 首次启动前在 `backend/` 运行 `python -m app.security admin` 创建管理员。
> 新增登录页面、可撤销会话、RBAC、用户限流、重请求并发控制、Agent 请求隔离与压测脚本。
> 当前部署使用 **1 个 API worker**。详见 [启动、权限和并发说明](docs/auth_concurrency.md)。

面向垂直领域文档（医疗 / 法律 / 金融）的 **图谱增强智能问答系统**。

在向量 RAG 的基础上补齐了**后端知识图谱**与**多跳子图检索**，并用 **LangGraph** 把
"检索 → 生成"的单趟流水线改造成带**工具调用决策**与**引用校验重试**的 Agent。

> 原 README 停留在 `LangExtractApp` 阶段（只描述文本抽取，未提 RAG/GraphRAG），
> 本文档已按当前代码重写。

---

## 1. 它解决什么问题

普通的向量 RAG 把文档切块、按语义相似度召回。它在**单个块就能回答**的问题上很好用，
但有两类问题它**原理上**答不好：

| 问题类型 | 例子 | 为什么切块检索做不到 |
|---|---|---|
| **多跳关系** | "张建国服用的药是哪家公司研发的？" | "张建国"在一个块、"诺华"在另一个块，**没有任何一个块同时包含两者**，相似度召回无法把跨块事实连起来 |
| **聚合统计** | "文档中一共涉及哪几家机构？" | 机构分散在各块，没有单块包含完整清单，模型看不到全貌 |

这两类问题正是 GraphRAG 的用武之地。本项目把知识图谱落到后端、支持多跳遍历与社区摘要，
并用评测**量化**了增量（见第 5 节，含诚实的不利结论）。

---

## 2. 架构

```
                        ┌──────────────── 前端 (React 18 + TS + Vite) ────────────────┐
                        │  4 步流程: 文档输入 → 信息抽取 → 知识图谱 → 智能问答          │
                        │  溯源高亮: 依据 char_interval 把答案里的实体锚回原文          │
                        └────────────────────────────┬───────────────────────────────┘
                                                     │ HTTP / SSE
                        ┌────────────────────────────┴───────────────────────────────┐
                        │                    FastAPI 后端                             │
                        │                                                            │
   PDF ──MinerU──► Markdown ──LangExtract+DeepSeek──► 结构化抽取(含 char_interval)     │
                        │                                     │                      │
                        │                       ┌─────────────┴─────────────┐        │
                        │                       ▼                           ▼        │
                        │              ┌─────────────────┐        ┌─────────────────┐ │
                        │              │  向量库 (Chroma) │        │ 知识图谱(内存+JSON)│ │
                        │              │  语义相似召回      │        │ 实体/关系/多跳/社区 │ │
                        │              └────────┬────────┘        └────────┬────────┘ │
                        │                       │                          │          │
                        │        ┌──────────────┴──────────────────────────┘          │
                        │        ▼                                                   │
                        │  ┌──────────────────────────────────────────────────────┐  │
                        │  │  LangGraph Agent                                     │  │
                        │  │  plan ─► tools ─► generate ─► verify ─┬─► END        │  │
                        │  │   ▲                                   │              │  │
                        │  │   └────────── rewrite ◄───────────────┘              │  │
                        │  │  工具: vector_search / graph_search /                │  │
                        │  │        graph_global_search / kb_meta                 │  │
                        │  └──────────────────────────────────────────────────────┘  │
                        └────────────────────────────────────────────────────────────┘
```

### Agent 的三个要素

1. **自主决策**：`plan` 节点决定调用哪些检索工具。明显的问题走零成本启发式，
   含糊的问题交 LLM 复核；LLM 失败自动回退启发式并记录在 trace 里。
2. **ReAct 式多轮工具调用**：`plan → tools → assess → plan` 循环。
   第 0 轮只看问题决策；第 N>0 轮把**已获得的证据摘要**交给模型，让它判断还缺什么。
   两条循环职责不同，不要混为一谈：
   - `assess → plan`：**补检索**（解决"检索没拿到东西"）
   - `verify → rewrite`：**改问题**（解决"检索到了但答案编造了"）
3. **可观测 + 可计量**：`trace` 记录每轮决策方式、调用了哪些工具、校验结果；
   `usage` 记录 LLM 调用次数、token 与**按节点归因**的耗时，配置单价后自动算费用。

**规划失败降级链**（保证永远有工具可执行）：
非法工具名 → 兜底组合；兜底组合已用过 → 未用过的其它工具；全用过 → 重复兜底组合。
另外：单轮内所有工具全部返回空时，会自动换未用过的工具再查一轮。

---

## 3. 快速开始

### 3.1 环境变量

```bash
cd backend
cp .env.example .env      # 本项目已提供；原 README 提到的该文件此前缺失
# 编辑 .env 填入密钥
```

| 变量 | 用途 | 是否必需 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 信息抽取 / 关系抽取 / Agent 问答 | 是 |
| `DEEPSEEK_BASE_URL` | 默认 `https://api.deepseek.com` | 否 |
| `DEFAULT_MODEL` | 如 `deepseek-chat` | 是 |
| `DASHSCOPE_API_KEY` | 阿里云百炼 Embedding（向量检索） | 向量功能必需 |
| `EMBEDDING_MODEL` | 如 `text-embedding-v4` | 向量功能必需 |
| `MINERU_API_KEY` | PDF 解析（MinerU） | PDF 上传必需 |
| `GRAPH_PERSIST_PATH` | 图谱持久化路径 | 否 |
| `GRAPH_HOPS` / `GRAPH_MAX_NODES` | 图检索跳数与子图预算 | 否 |
| `AGENT_MAX_ATTEMPTS` / `AGENT_GROUNDED_THRESHOLD` | 反思轮数上限与引用校验阈值 | 否 |

> ⚠️ **必须从 `backend/` 目录启动后端**。早期版本的 `config.py` 用
> `load_dotenv("../.env")`（相对当前工作目录解析），从项目根启动会导致所有密钥读到空值。
> 现已改为按 `BACKEND_DIR` 拼绝对路径，但历史文档与习惯仍建议在 `backend/` 下启动。

### 3.2 启动

```bash
# 后端
cd backend
uvicorn app.main:app --reload --port 8000
# 打开 http://localhost:8000/docs 查看全部接口

# 前端（另开终端）
cd frontend
npm run dev
```

### 3.3 测试与评测

```bash
cd backend

# 单元 + 集成测试（380 个用例，全部离线，无需 API Key）
python -m pytest tests/ -q

# 检索评测：纯离线，零 API 成本，秒级返回
python -m evaluation.run_eval

# 追加真实向量检索（需 DASHSCOPE_API_KEY，费用极低）
python -m evaluation.run_eval --with-vector

# 追加 Agent 端到端评测（需 DEEPSEEK_API_KEY）
python -m evaluation.run_eval --live

# 真实链路冒烟验证（1 段文本 + 2 个问题，成本可控）
python -m evaluation.live_smoke
```

---

## 4. 主要接口

### 图谱接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/rag/graph/build` | 从抽取结果构建图谱（实体 + LLM 关系抽取 + 落盘） |
| GET | `/api/rag/graph/stats` | 规模、可溯源率、社区数、模块度 Q |
| GET | `/api/rag/graph/communities` | 社区划分与成员清单 |
| POST | `/api/rag/graph/query` | 图谱局部（多跳）/ 全局（社区）检索 |
| GET | `/api/rag/graph/entities/{name}` | 实体详情、N 跳邻居、关联关系（含证据与字符区间） |
| POST | `/api/rag/graph/reset` | 清空图谱 |

### Agent 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/rag/agent/qa` | 完整 Agent 流程，返回答案 / 来源 / 引用校验 / trace |
| POST | `/api/rag/agent/qa/stream` | SSE 流式版本 |
| GET | `/api/rag/agent/route-preview` | 只跑路由逻辑（不调用模型），便于调试 |

**SSE 事件协议**（与既有 `/api/rag/qa/stream` 向后兼容）：

```
event: sources       检索来源（含 char_interval，可直接用于前端高亮）
event: graph         图检索结果（关系链 / 社区摘要 / 命中实体）
event: chunk         答案增量
event: verification  引用校验（可接地比例、未接地项）
event: retry         触发反思重试（附改写后的查询）
event: done          结束（尝试轮数 + 最终校验结果）
event: error         异常
```

---

## 5. 评测结果

### 5.1 检索评测（`python -m evaluation.run_eval`，纯离线、零成本、可复现）

14 个问题，22 个语料块，切块检索取 top-3：

```
图谱: 实体 23 | 关系 26 | 社区 7 | 模块度 Q=0.3861 | 可溯源率 1.0
必须跨块才能凑齐证据的问题: 10 / 14

配置                     总召回   detail  multihop   global   跨块召回   跨块全对   延迟ms
hybrid_bm25_graph        0.982    1.000     1.000    0.950    0.975     9/10      0.2
graph_auto               0.821    0.750     0.900    0.800    0.850     6/10      0.2
chunk_bm25               0.738    1.000     0.567    0.700    0.633     4/10      0.1
graph_local              0.679    0.750     0.900    0.400    0.650     6/10      0.1
graph_global             0.571    0.000     0.800    0.800    0.800     5/10      0.2
```

### 5.2 追加真实向量检索与 Agent（`--with-vector --live`，需 API Key）

同一批问题，接入真实 Chroma + DashScope Embedding，以及完整 Agent：

```
配置                     总召回   detail  multihop   global   跨块召回   跨块全对   延迟ms
hybrid_vector_graph      0.946    1.000     0.900    0.950    0.925     8/10     130.2
agent_end_to_end         0.804    0.500     0.900    0.950    0.925     8/10    9033.2
vector_chroma            0.732    1.000     0.500    0.750    0.625     3/10     210.2

Agent 端到端指标
  引用可验证率（溯源质量）  : 1.000     ← 答案里的数字/实体全部能在证据中找到
  引用校验通过率           : 1.000
  反思重试：平均轮数 / 触发题数 : 1.07 / 1

  多轮工具调用
    平均工具轮数             : 1.5
    触发多轮补检索的题数       : 4 / 14
    单题最多工具轮数          : 3

  LLM 成本
    总调用次数 / 单题均值      : 44 / 3.14
    单题最多调用次数           : 6
    总 token / 单题均值        : 59,666 / 4,261
    token 按节点归因：
      assess         调用  6 次  输入   3,787  输出   2,298
      generate       调用 15 次  输入  24,996  输出  16,118
      plan_followup  调用  7 次  输入   4,213  输出   2,639
      route          调用 15 次  输入   3,382  输出   1,444
```

> **成本追踪的一个说明**：单价刻意**不写死在代码里**（各家模型单价会变，写死迟早变成
> 错误信息）。未配置时只报 token 数，`cost` 返回 `null` 并标记 `cost_configured=false`；
> 设了 `LLM_PRICE_INPUT_PER_1M` / `LLM_PRICE_OUTPUT_PER_1M` 才计算费用。
> 宁可少一个数字，也不要一个假数字。

> ⚠️ **离线指标是确定性的，`--live` 指标不是。**
> 前 5 个配置是纯本地检索，两次运行逐位相同（有回归测试锁定）。
> `agent_end_to_end` 与 `vector_chroma` 依赖真实模型，多次运行会有波动
> （实测 `agent_end_to_end` 总召回在 0.750–0.875 之间，`multihop` 稳定在 0.900）。
> 引用可验证率两次运行都是 1.000 —— 这是引用校验的价值：
> **即使答案措辞每次不同，"有没有编造"这个判断始终确定。**
>
> 评测产物分文件留存：`evaluation/results/offline.json` 与 `evaluation/results/live.json`。

### 5.3 可写进简历的结论

| 结论 | 数据 |
|---|---|
| 多跳问题：图检索 vs 切块检索 | **0.900 vs 0.567**（相对提升约 59%） |
| 多跳问题：图检索 vs **真实向量检索** | **0.900 vs 0.500**（相对提升 80%） |
| 跨块问题"证据全部召回" | 图 **6/10** vs 切块 4/10 vs 向量 3/10 |
| 混合配置综合最优 → 两路**互补而非替代** | 0.982（BM25+图）/ 0.946（向量+图） |
| 引用可验证率（垂直领域防编造） | **1.000**，14 题零未接地项 |
| 图检索延迟 | **0.2ms**（内存图）；真实向量检索 210ms —— 差约 1000 倍 |
| **多轮工具调用调优：成本 −46%、延迟 −44%** | 见下方 5.4 |

### 5.4 一次靠实测数据完成的性能调优（多轮改造的真实过程）

ReAct 式多轮工具调用上线后，实测发现它**过度触发**：14 题里 11 题都补检索、
平均 2.29 轮，但答案召回相比单轮版**没有提升** —— 判官天然倾向于"再查一点更保险"。

于是加了一道**零成本前置门控**：只在证据"薄"（< 400 字符）或聚合型问题
（"涉及哪些/一共有几个"这类需要完整清单的问题）时才请判官。

| 指标 | 初版（无门控） | 加门控后 | 变化 |
|---|---|---|---|
| 平均工具轮数 | 2.29 | **1.5** | −34% |
| 触发多轮补检索的题数 | 11 / 14 | **4 / 14** | −64% |
| 单题 LLM 调用 | 4.57 | **3.14** | −31% |
| 单题 token | 7,839 | **4,261** | −46% |
| 单题延迟 | 16,150 ms | **9,033 ms** | −44% |
| 判官调用次数 | 18 | **6** | −67% |
| 多跳召回 / 聚合召回 | 0.900 / 0.950 | **0.900 / 0.950** | 不变 |
| 引用可验证率 | 1.000 | **1.000** | 不变 |

**在关键指标完全不变的前提下，成本与延迟各降约一半。**
这正是为什么 Agent 系统必须带计量：没有 token/延迟归因，这个过度触发问题
只能靠"感觉慢"来发现，而"感觉"没法支撑任何优化决策。

### 5.5 必须如实说明的不利结论

- **细节/数值类问题上切块检索更好**（1.000 vs 0.750）。`38%` 这类数值不是图谱实体，
  图检索天生不该在这里取胜。把它包装成"全面碾压"会在面试中被一句话问穿：
  *"那你们的图检索对什么样的查询反而更差？"*
- **真实的向量检索在多跳题上甚至略逊于 BM25**（0.500 vs 0.567）。
  这说明"Embedding 语义相似"并不能替代"关系路径"：
  跨块事实在向量空间里并不因为语义相关而靠得更近。
- 图检索延迟优势来自内存图。换成 Neo4j 后延迟会显著上升，届时必须重新评测。

指标口径见 `evaluation/run_eval.py` 头部注释。其中 `single_chunk_ceiling`
给出"任何切块方案的理论上限"，用来判断一道题是否**原理上**需要跨块推理。

---

## 6. 已知局限

1. **关系抽取不完全可复现**。实测同一段文本、`temperature=0`，两次调用的三元组
   数量与内容都可能不同（一次 4 条、一次 3 条），并可能漏掉关键边。
   好消息是检索对图谱不完整有鲁棒性（查询中出现的实体都会被设为种子做多跳扩展），
   但**抽取完整性与一致性才是当前的主要瓶颈**，而非图算法。
   改进方向：多轮抽取取并集、或对同一文档做多次抽取后按共识过滤（Microsoft GraphRAG 的
   gleaning 思路）。
2. **关系抽取是逐段独立进行的**，没有跨段共指消解。文档里"该药""该患者"能否被正确
   绑定到具体实体，取决于单段文本内的信息是否充分。
3. **社区发现是 Louvain 的局部移动阶段**（未做图聚合）。中小规模图上效果接近完整
   Louvain，超大图应换 Leiden（`networkx` / `graspologic`）。
4. **图谱存储是 JSON 文件 + 进程内单例**，适合单机中小规模。百万级边需换 Neo4j。
5. **前端的关系图仍是启发式生成**（`frontend/src/components/steps/KnowledgeGraph.tsx`
   的 `generateRelationshipsFromItems`），尚未改用后端图谱接口。
   这是下一步最值得做的衔接：后端已有 `/api/rag/graph/*`，前端切换过去即可。

---

## 7. 代码结构

```
backend/
├── app/
│   ├── main.py                      # FastAPI 入口（已挂载图谱与 Agent 路由）
│   ├── config.py                    # 配置（.env 按绝对路径加载）
│   ├── api/
│   │   ├── routes.py                # 抽取 / 场景 / 缓存
│   │   ├── rag_routes.py            # 原有向量 RAG 接口（保持兼容）
│   │   └── graph_routes.py          # 新增：图谱接口 + Agent 接口
│   ├── core/
│   │   └── extractor.py             # LangExtract 结构化抽取（产出 char_interval）
│   ├── services/
│   │   ├── pdf_parser.py            # MinerU PDF → Markdown
│   │   ├── vector_store_chroma.py   # Chroma 向量库（含溯源 metadata）
│   │   ├── vector_store.py          # Qdrant 备选实现
│   │   ├── qa_agent.py              # 原有单趟 RAG 问答（保持兼容）
│   │   ├── graph_store.py           # ★ 知识图谱：实体/关系/多跳/社区/持久化
│   │   ├── relation_extractor.py    # ★ LLM 三元组抽取（四道防幻觉约束）
│   │   ├── graph_retriever.py       # ★ 局部多跳 + 全局社区检索 → LLM 上下文
│   │   ├── agent_qa.py              # ★ LangGraph Agent（双循环 + 降级链）
│   │   └── llm_usage.py             # ★ LLM 用量与成本追踪（按节点归因）
│   ├── scenarios/                   # 7 个业务场景插件（注册表模式）
│   └── models/schemas.py            # Pydantic 模型（含图谱与 Agent 模型）
├── evaluation/
│   ├── fixtures.py                  # ★ 固定评测语料 + 人工标注图谱 + 问题集
│   ├── run_eval.py                  # ★ 评测框架（离线可跑 / --live 接真实 API）
│   └── live_smoke.py                # ★ 真实 API 端到端冒烟验证
└── tests/                           # 380 个用例，全部离线
    ├── conftest.py                  # sys.path + 工作区临时目录 fixture
    ├── test_graph_store.py          # 图谱存储（含多跳可达性核心断言）
    ├── test_relation_extractor.py   # 关系抽取（FakeLLM 验证四道约束）
    ├── test_graph_retriever.py      # 图谱检索（局部/全局/分流）
    ├── test_agent_qa.py             # Agent 基础闭环（反思重试、路由降级、SSE）
    ├── test_agent_multiround.py     # ★ 多轮工具调用、判官门控、成本接线
    ├── test_llm_usage.py            # ★ 用量与成本口径
    ├── test_api_routes.py           # FastAPI 集成 + SSE 协议
    └── test_evaluation.py           # 评测数据完整性 + 结论回归
```

★ = 为兑现"GraphRAG + Agent"新增或重写的模块。

**文档**：
- `docs/graphrag_agent.md` —— 技术深度说明 + 面试讲稿（含所有取舍与踩坑记录）
- `docs/backend_agent_reading_guide.md` —— **后端/Agent 视角的代码阅读路线**（明确排除前端）
- `docs/resume_agent_role.md` —— **Agent 岗专用简历素材**与面试问答
- `docs/scenarios_demo.md` —— 四个业务场景的示例文本

---

## 8. 技术栈

**后端**：FastAPI · Pydantic · LangExtract · LangChain / **LangGraph** · DeepSeek API ·
ChromaDB / Qdrant · MinerU · 纯标准库实现的图存储与 Louvain 社区发现

**前端**：React 18 · TypeScript · Vite · TailwindCSS · motion · 手写 canvas 力导向图

**工程**：pytest（380 用例）· 可复现评测框架 · SSE 流式协议 · 三处降级设计
（LLM 路由失败退启发式 / 关系抽取失败退共现边 / 向量库缺失仍可用图检索）
