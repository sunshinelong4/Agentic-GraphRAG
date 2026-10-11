# 后端 / Agent 视角代码阅读路线

> 本文档只服务于 **Agent / RAG 应用开发岗**。
> **前端 12,503 行代码与本路线无关**，其中 4,639 行是 shadcn 组件库脚手架。
> 按本路线走，需要真正读懂的后端代码约 **4,600 行**（含新增模块），跳过约 **12,000 行**前端。

---

## 0. 先建立规模感

| 部分 | 行数 | 对你是否重要 |
|---|---|---|
| `backend/app/`（Python 后端） | 4,020 | ✅ 全部重要 |
| `backend/evaluation/`（评测） | ~1,100 | ✅ 全部重要 |
| `backend/tests/`（测试） | ~4,300 | ✅ 建议读 |
| `frontend/src/components/ui/`（shadcn 脚手架） | 4,639 | ❌ 跳过 |
| `frontend/src/` 其余业务代码 | 7,864 | ⚠️ 只读 2 个文件（见第 5 节） |

---

## 1. 第一遍：建立"数据从哪来到哪去"（半天）

**目标**：能白板画出这条链路。

| 顺序 | 文件 | 只回答一个问题 |
|---|---|---|
| 1 | `backend/initialize_pdf_parsing.ipynb` | 整条流水线长什么样？（25 个 cell，线性可跑，是小白的入口） |
| 2 | `backend/app/main.py` | 服务暴露了哪些路由？各自挂在哪？ |
| 3 | `backend/app/api/routes.py` | 抽取接口的请求/响应长什么样？ |
| 4 | `backend/app/core/extractor.py` | LangExtract 怎么调？`char_interval` 从哪来？ |
| 5 | `backend/app/scenarios/base.py` + `medical.py` | 场景插件是怎么注册和扩展的？ |

**自测**：不看代码写出 `char_interval` 的产生、存储、消费三个环节分别在哪个文件。

---

## 2. 第二遍：RAG 主链路（1 天）

**目标**：讲清"一次问答里，检索到的文本是怎么变成答案的"。

| 顺序 | 文件 | 只回答一个问题 |
|---|---|---|
| 6 | `backend/app/services/pdf_parser.py` | 第三方异步 API（上传→轮询→下载）怎么集成？为什么不能同步等？ |
| 7 | `backend/app/services/vector_store_chroma.py` | Chunk 怎么切、怎么存？`score = 1 - distance/2` 怎么来的？ |
| 8 | `backend/app/services/qa_agent.py` | 原有的单趟 RAG：retrieve → format → prompt → generate。**注意这就是"没有 Agent 的基线"** |
| 9 | `backend/app/api/rag_routes.py` | SSE 事件协议怎么设计的？为什么先发 sources 再发 chunk？ |

**自测**：
- 为什么 Chroma 的 metadata 只能放 `str/int/float/bool`？代价是什么？
- `qa_agent.py` 里的 `_is_kb_meta_question` 为什么要单独走一条分支？

---

## 3. 第三遍：图谱与检索质量（1–2 天，核心）

**目标**：能解释"为什么切块检索答不出多跳问题"，并动手验证。

| 顺序 | 文件 | 只回答一个问题 |
|---|---|---|
| 10 | `backend/app/services/graph_store.py` | 实体怎么归并？多跳 BFS 怎么带预算？Louvain 的模块度增益公式怎么用？ |
| 11 | `backend/app/services/relation_extractor.py` | 四道防幻觉约束分别拦什么？降级为什么只在"没产出"时触发？ |
| 12 | `backend/app/services/graph_retriever.py` | 关系链怎么渲染成 LLM 能读的文本？弱证据为什么要标注？ |
| 13 | `backend/evaluation/fixtures.py` | 评测语料为什么要"事实原子化 + 类名指代 + 加干扰块"？ |
| 14 | `backend/evaluation/run_eval.py` | `single_chunk_ceiling` 的意义是什么？为什么检索指标能离线算？ |

**必做的动手验证**（这比读代码有用得多）：

```bash
cd backend
python -m evaluation.run_eval            # 离线，秒级
python -m pytest tests/test_graph_store.py -v   # 看多跳可达性用例
```

**自测**：给出一个具体问题，说明为什么真实向量检索（0.500）在多跳题上比关键词 BM25（0.567）还低。

---

## 4. 第四遍：Agent 工程（2 天，简历重点）

**目标**：能讲清两个循环的分工、三层降级、以及成本归因。

| 顺序 | 文件 / 方法 | 只回答一个问题 |
|---|---|---|
| 15 | `agent_qa.py` 的 `AgentState` | 为什么 `route` / `used_tools` / `current_tools` 要分成三个字段？ |
| 16 | `_plan_node` + `_initial_plan` + `_followup_plan` | 单轮规划与多轮续查规划的区别是什么？ |
| 17 | `_assess_node` + `_needs_assessment` | 为什么判官要先过一道廉价门控？（→ 看 README 5.4 的实测数据） |
| 18 | `_after_assess` vs `_after_verify` | 两个循环各解决什么问题？ |
| 19 | `_make_tools` / `_execute_tools` | 工具名白名单与参数白名单分别防什么？为什么流式路径不能复用 `ToolNode`？ |
| 20 | `verify_citations` | 为什么不用另一个 LLM 当裁判？可计算性带来了什么好处？ |
| 21 | `llm_usage.py` | 为什么要按节点归因？为什么价格不写死在代码里？ |
| 22 | `app/api/graph_routes.py` | SSE 事件序列的语义？`assess` / `usage` 事件为什么存在？ |

**自测**（面试高频）：
- Agent 每问一次要调用几次 LLM？分别是什么？花多少 token？
- 如果检索全部为空，Agent 会怎么做？
- 引用校验不通过会怎样？元信息问题为什么不重试？

---

## 5. 前端你只需要读 2 个文件（各 10 分钟）

| 文件 | 读它的**非前端**理由 |
|---|---|
| `frontend/src/services/api.ts` | 这是**接口契约文档**。TS 类型是后端 Pydantic 的影子。你要能跟人约定协议变更 |
| `frontend/src/components/steps/KnowledgeGraph.tsx` 里的 `generateRelationshipsFromItems()` | ⚠️ 这是**反面教材**：`if 人物 and 机构 → "属于"` 的笛卡尔积硬编码，造出的全是假边。面试讲"我为什么重做关系抽取"时，这是第一手素材 |

**明确跳过**：`components/ui/`（48 文件）、`GraphVisualization.tsx`（手写 canvas 力模拟）、
`Landing.tsx` / `TeachingMode.tsx` / `Header.tsx` / `ProgressBar.tsx` / `StepContainer.tsx` /
`Tooltip.tsx` / `EntityCard.tsx` / `EvidenceHighlight*.tsx`、`data/mockDocuments.ts`、
`styles/*.css`、以及全部 Tailwind / Radix / motion / recharts 依赖。

**你只需要"会跑"，不需要"会写"**：

```bash
cd frontend && npm run dev    # 打开页面走一遍 4 步流程，用于演示
```

---

## 6. 时间预算参考

| 阶段 | 内容 | 时间 |
|---|---|---|
| 第一遍 | 数据流全景 | 0.5 天 |
| 第二遍 | RAG 主链路 | 1 天 |
| 第三遍 | 图谱与检索质量 | 1.5 天 |
| 第四遍 | Agent 工程 | 2 天 |
| 动手 | 跑评测、改一个参数看指标变化 | 0.5 天 |
| **合计** | | **约 5.5 天** |

如果只有 2 天：直接做**第三遍的第 10–14 项 + 第四遍的 15–20 项**，
其余靠 `README.md` 补齐。这条路径覆盖了面试 90% 的提问面。

---

## 7. 每读一个文件的产出物

不要只是"读过"。每读完一个核心文件，给自己留三句话：

1. **它解决什么问题**（不用术语，用业务语言）
2. **它的关键取舍是什么**（选了什么、放弃了什么、代价是什么）
3. **如果让我改，我会改哪里**（哪怕只是"这个函数太长了"）

这三句话就是面试时的回答模板。本项目里可以这么练手的目标（都带真实取舍）：

| 文件 | 关键取舍 |
|---|---|
| `graph_store.py` | entity_id 用规范化名而非 UUID：牺牲规范性换可读性 |
| `relation_extractor.py` | 四道约束 vs 召回率；共现降级 vs 图谱纯净度 |
| `graph_retriever.py` | 子图节点预算：牺牲召回完整度换上下文不爆 |
| `agent_qa.py` | 多轮补检索 vs 延迟成本（实测 −46% 数字见 README 5.4） |
| `llm_usage.py` | 估算 token vs 精确；不写死价格 vs 报告里少一个数字 |
| `run_eval.py` | 关键词命中 vs 语义等价判定：选确定的那个 |
