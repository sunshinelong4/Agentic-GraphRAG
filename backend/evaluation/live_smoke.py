"""真实 API 冒烟验证

目的：离线测试再全，也不能证明**真实链路**能跑通（模型返回的格式、被指代解析、
网络与额度问题都只在真调用时暴露）。本脚本用最小的 token 成本做一次端到端验证。

用法（在 backend/ 目录下执行，需要 .env 里已配置 DEEPSEEK_API_KEY）：

    python -m evaluation.live_smoke

验证内容
--------
1. 关系抽取：真实 DeepSeek 能否按约束产出合法三元组（含实体锚定与证据校验）
2. 图谱落图：抽取结果能否建成图并支持多跳
3. Agent 端到端：规划 → 工具 → 生成 → 引用校验 是否闭环
4. 成本观测：打印每次调用的耗时与 token 用量（若模型返回 usage）

成本控制：只用一段约 400 字的文本、2 个问题、最多 2 轮重试，
单次运行通常在 1 万 token 以内。
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# 先导入 config：它负责用**绝对路径**加载 .env（见 config.py 里的说明）。
# 其它模块里散落的 os.getenv 都依赖这次加载先发生。
from app.config import get_settings
from app.services.agent_qa import GraphRAGAgent
from app.services.graph_store import KnowledgeGraphStore
from app.services.relation_extractor import RelationExtractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

settings = get_settings()

# 刻意选一段"关系跨句、含指代"的文本：如果模型不解析指代就抽不出关键关系
SAMPLE_TEXT = (
    "患者张建国，男，62岁，因反复胸闷、气短3个月入院。"
    "心脏彩超显示左室射血分数38%，诊断为扩张型心肌病。"
    "给予沙库巴曲缬沙坦钠片50mg po bid抗心衰治疗。"
    "该药属于ARNI类药物，由诺华公司研发。"
    "用药期间需监测血压与肾功能。"
)

EXTRACTIONS: List[Dict[str, Any]] = [
    {"extraction_class": "患者", "extraction_text": "张建国",
     "attributes": {"性别": "男", "年龄": "62"}},
    {"extraction_class": "诊断", "extraction_text": "扩张型心肌病", "attributes": {}},
    {"extraction_class": "药物", "extraction_text": "沙库巴曲缬沙坦钠片",
     "attributes": {"剂量": "50mg po bid"}},
    {"extraction_class": "药物类别", "extraction_text": "ARNI", "attributes": {}},
    {"extraction_class": "机构", "extraction_text": "诺华", "attributes": {}},
]

QUESTIONS = [
    "张建国服用的沙库巴曲缬沙坦钠片是哪家公司研发的？",
    "文档中一共涉及哪些机构？",
]


def _attach_positions() -> List[Dict[str, Any]]:
    """给抽取项补上字符区间（真实链路里由 LangExtract 提供）。"""
    enriched = []
    for item in EXTRACTIONS:
        start = SAMPLE_TEXT.find(item["extraction_text"])
        enriched.append(
            {
                **item,
                "char_interval": (
                    {"start_pos": start, "end_pos": start + len(item["extraction_text"])}
                    if start != -1
                    else None
                ),
            }
        )
    return enriched


def check_key() -> str:
    """确认 API Key 可用。走 Settings 而不是 os.getenv，保证 .env 一定被加载过。"""
    key = settings.deepseek_api_key or ""
    if not key:
        print("!! 未配置 DEEPSEEK_API_KEY（确保 backend/.env 存在且含该字段）")
        raise SystemExit(2)
    print(f"API Key 已配置: {key[:8]}...  模型: {settings.default_model}")
    return key


def main() -> int:
    check_key()

    print("\n" + "=" * 72)
    print("步骤 1/3：真实关系抽取（验证四道约束在真实模型上是否成立）")
    print("=" * 72)

    extractor = RelationExtractor(
        model=settings.default_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )

    started = time.perf_counter()
    triples = extractor.extract(
        text=SAMPLE_TEXT,
        known_entities=[item["extraction_text"] for item in EXTRACTIONS],
        max_relations=20,
    )
    elapsed = time.perf_counter() - started

    print(f"\n耗时 {elapsed:.1f}s，抽取到 {len(triples)} 条关系：")
    for triple in triples:
        evidence_ok = triple.evidence in SAMPLE_TEXT if triple.evidence else False
        mark = "OK " if evidence_ok else "证据非原文子串"
        print(f"  • {triple.head} --[{triple.relation}]--> {triple.tail}  ({mark}, conf={triple.confidence})")
    print(f"\n抽取过程统计: {extractor.last_stats}")

    if not triples:
        print("\n!! 未抽取到任何关系。真实模型可能改动了输出格式，需要检查提示词。")
        return 1

    validated = sum(1 for t in triples if t.evidence and t.evidence in SAMPLE_TEXT)
    print(f"\n证据可验证比例: {validated}/{len(triples)}")

    # 说明一个实测发现：即便 temperature=0，两次调用抽取到的三元组也可能不同
    # （第一次抽到 "张建国-患有->扩张型心肌病"，第二次抽到 "张建国-使用->沙库巴曲缬沙坦钠片"）。
    # 这正是**离线评测必须用人工标注 fixture** 的原因 —— 用 LLM 现抽的图做基准，
    # 每次评测的数据都不一样，数字就无法复现。
    print(
        "\n注意：关系抽取在真实模型上并非完全可复现（同一文本两次调用可能给出\n"
        "      不同但都合理的三元组）。因此评测用的是人工标注的固定图谱，\n"
        "      而不是现场抽出来的图。"
    )

    print("\n" + "=" * 72)
    print("步骤 2/3：图谱落图与多跳可达性")
    print("=" * 72)

    store = KnowledgeGraphStore()
    # 复用步骤 1 已经抽好的三元组，不再重复调用模型 ——
    # 既省 token，也保证报告的前后两步描述的是同一份数据。
    entity_stats = store.add_extractions("smoke-1", "冒烟测试病历.pdf", _attach_positions())
    relation_stats = store.add_triples(
        triples, doc_id="smoke-1", doc_title="冒烟测试病历.pdf", auto_create_entities=False
    )
    print(f"\n实体落图: {entity_stats}")
    print(f"关系落图: {relation_stats}")
    print(f"图谱统计: {store.stats()}")

    path = store.shortest_path("张建国", "诺华", max_hops=4)
    if path:
        print(f"\n多跳路径（{path['hops']} 跳）: {path['text']}")
        print("→ 张建国与诺华从未出现在同一句话里，这条路径只能由图给出")
    else:
        print("\n!! 未找到 张建国 → 诺华 的路径，说明指代关系没被抽出。")
        print("   这是真实模型上最常见的失败模式（'该药'未被解析成具体药名）。")
        print(f"   当前图中的关系: {[(r.source_id, r.relation_type, r.target_id) for r in store.relations.values()]}")

    print("\n" + "=" * 72)
    print("步骤 3/3：Agent 端到端（规划 → 工具 → 生成 → 引用校验）")
    print("=" * 72)

    agent = GraphRAGAgent(
        graph_store=store,
        vector_store=None,  # 冒烟测试不连向量库，只验证图链路
        model=settings.default_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        use_llm_router=True,
        max_attempts=2,
    )

    failures = 0
    for index, question in enumerate(QUESTIONS, 1):
        print(f"\n--- 问题 {index}: {question} ---")
        started = time.perf_counter()
        try:
            answer = agent.answer(question)
        except Exception as exc:
            print(f"!! Agent 执行失败: {exc}")
            failures += 1
            continue
        elapsed = time.perf_counter() - started

        print(f"回答: {answer['answer'][:200]}")
        print(f"使用的工具: {answer['route']}")
        print(
            f"引用校验: 可接地比例={answer['verification'].get('grounded_ratio')}, "
            f"通过={answer['verification'].get('passed')}, "
            f"未接地={answer['verification'].get('ungrounded_numbers', []) + answer['verification'].get('ungrounded_entities', [])}"
        )
        print(f"尝试轮数: {answer['attempts']} | 工具轮数: {answer['tool_rounds']} | 耗时 {elapsed:.1f}s")
        plan = next((e for e in answer["trace"] if e["node"] == "plan"), {})
        print(f"规划方式: {plan.get('decided_by')}（llm=模型决策, heuristic=启发式回退, assessor=采纳判官建议）")

        usage = answer.get("usage") or {}
        print(
            f"LLM 用量: 调用 {usage.get('llm_calls')} 次 | "
            f"token {usage.get('total_tokens')} | "
            f"耗时 {usage.get('llm_latency_ms')}ms"
        )
        for node, bucket in (usage.get("by_node") or {}).items():
            print(
                f"    {node:<14} 调用 {bucket['calls']} 次  "
                f"输入 {bucket['prompt_tokens']}  输出 {bucket['completion_tokens']}"
            )
        if usage.get("cost_configured"):
            print(f"    费用估算: {usage.get('cost')}")
        else:
            print("    费用估算: 未配置单价（设 LLM_PRICE_INPUT_PER_1M / LLM_PRICE_OUTPUT_PER_1M 后自动计算）")

    print("\n" + "=" * 72)
    if failures:
        print(f"冒烟验证未完全通过：{failures} 个问题执行失败")
        return 1
    print("冒烟验证通过：真实链路（关系抽取 → 落图 → Agent 问答 → 引用校验）可用")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
