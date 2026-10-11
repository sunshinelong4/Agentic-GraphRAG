"""评测用的固定语料与知识图谱

为什么把评测数据**用代码写死**而不是丢一个巨大的 JSON：
1. **可复现**：`python -m evaluation.run_eval` 在任何机器上得到同样结果
2. **可审查**：数据人工编写，Git diff 里看得清改了什么
3. **不依赖被测对象**：评测不应该用 LLM 建图再拿这张图去证明"LLM 关系抽得好"，
   那是循环论证。这里的实体与关系全部人工标注。

语料设计的三个关键决策（都是第一版评测跑出不合理结果后返工出来的）
--------------------------------------------------------------------
**决策 1：事实原子化。**
第一版把"沙库巴曲缬沙坦钠片由诺华研发，属于 ARNI"写在同一块里，
结果一个块就装下了全部答案，多跳题退化成单块题，基线拿到满分。
现在一个块只放一个事实。

**决策 2：说明书用类名/指代，不重复药名。**
真实的药品说明书经常写"该 ARNI 类药物由…研制"，而不是反复写全名。
这让"药名 → 类名 → 研制方"必须走**两跳**：查询"沙库巴曲缬沙坦钠片是哪家公司研制的"
在词面上匹配不到 ck-11（那里只有"该 ARNI 类药物"），只有图上
`沙库巴曲缬沙坦钠片 --属于--> ARNI --研制--> 诺华` 这条路径能连起来。

**决策 3：加入干扰块，让 top_k 真正具有选择性。**
第一版只有 8 个块却取 top-5，几乎等于全取，任何检索器都能"蒙中"。
现在 20 个块取 top-3，检索器必须真的排序。

附带一个诚实的说明：跨块问题并不总是能难住词面检索 —— 只要查询里的词
恰好也出现在远端块里，BM25 就会"碰巧搭上桥"。所以评测同时报告
`ceiling`（单块上限）与 top_k，让读者自己判断哪些题是靠运气过的，
而不是靠一个好看的总分掩盖问题。
"""

from __future__ import annotations

from typing import Any, Dict, List

# =============================================================================
# 语料：20 个块（12 个信息块 + 8 个干扰块）
# =============================================================================

CORPUS: List[Dict[str, Any]] = [
    # ---------------- 病历摘要（doc-medical） ----------------
    {
        "chunk_id": "ck-1",
        "doc_id": "doc-medical",
        "doc_title": "住院病历摘要.pdf",
        "content": "患者信息：姓名张建国，男，62岁，已婚，退休教师。医保类型：北京市职工医保。",
    },
    {
        "chunk_id": "ck-2",
        "doc_id": "doc-medical",
        "doc_title": "住院病历摘要.pdf",
        "content": (
            "主诉：反复胸闷、气短3个月，加重伴双下肢水肿1周。"
            "既往有高血压病史15年，2型糖尿病史8年。"
        ),
    },
    {
        "chunk_id": "ck-3",
        "doc_id": "doc-medical",
        "doc_title": "住院病历摘要.pdf",
        "content": (
            "辅助检查：心脏彩超显示左房内径48mm，左室射血分数（LVEF）38%，"
            "二尖瓣中度反流，诊断为扩张型心肌病。BNP：1856pg/ml。"
        ),
    },
    {
        "chunk_id": "ck-4",
        "doc_id": "doc-medical",
        "doc_title": "住院病历摘要.pdf",
        "content": "长期用药：苯磺酸氨氯地平片5mg qd；二甲双胍0.5g bid。",
    },
    {
        "chunk_id": "ck-5",
        "doc_id": "doc-medical",
        "doc_title": "住院病历摘要.pdf",
        "content": "住院治疗方案（前三种）：呋塞米注射液40mg iv bid；螺内酯片20mg po qd；琥珀酸美托洛尔缓释片23.75mg po qd。",
    },
    {
        "chunk_id": "ck-6",
        "doc_id": "doc-medical",
        "doc_title": "住院病历摘要.pdf",
        "content": "住院治疗方案（后三种）：沙库巴曲缬沙坦钠片50mg po bid；达格列净片10mg po qd；华法林钠片2.5mg po qd。",
    },
    {
        "chunk_id": "ck-7",
        "doc_id": "doc-medical",
        "doc_title": "住院病历摘要.pdf",
        "content": "诊断：1. 慢性心力衰竭急性加重（NYHA IV级）；2. 扩张型心肌病；3. 心房颤动。",
    },
    {
        "chunk_id": "ck-8",
        "doc_id": "doc-medical",
        "doc_title": "住院病历摘要.pdf",
        "content": "诊断续：4. 高血压病3级（极高危）；5. 2型糖尿病；6. 轻度贫血。",
    },
    {
        "chunk_id": "ck-9",
        "doc_id": "doc-medical",
        "doc_title": "住院病历摘要.pdf",
        "content": "出院情况：症状明显改善，LVEF复查提升至42%，BNP降至580pg/ml。",
    },
    {
        "chunk_id": "ck-10",
        "doc_id": "doc-medical",
        "doc_title": "住院病历摘要.pdf",
        "content": "经治医生：心内科王晓明主任医师；住院医师：李小红。",
    },

    # ---------------- 药品说明书（doc-pharma） ----------------
    # 注意：这里刻意用类名指代，不重复具体药名 —— 见文件头"决策 2"
    {
        "chunk_id": "ck-11",
        "doc_id": "doc-pharma",
        "doc_title": "药品说明书.pdf",
        "content": "该ARNI类药物出自诺华公司。",
    },
    {
        "chunk_id": "ck-12",
        "doc_id": "doc-pharma",
        "doc_title": "药品说明书.pdf",
        "content": "该ARNI类药物的常见不良反应包括低血压、高钾血症和肾功能恶化。",
    },
    {
        "chunk_id": "ck-13",
        "doc_id": "doc-pharma",
        "doc_title": "药品说明书.pdf",
        "content": "该SGLT2抑制剂出自阿斯利康公司。",
    },
    {
        "chunk_id": "ck-14",
        "doc_id": "doc-pharma",
        "doc_title": "药品说明书.pdf",
        "content": "该SGLT2抑制剂除降糖外，还被证实具有心血管和肾脏保护作用。",
    },

    # ---------------- 干扰块 ----------------
    {
        "chunk_id": "dk-1",
        "doc_id": "doc-distractor",
        "doc_title": "其他病历.pdf",
        "content": "患者李女士，女，45岁，因反复咳嗽、咳痰2周入院，诊断为社区获得性肺炎。",
    },
    {
        "chunk_id": "dk-2",
        "doc_id": "doc-distractor",
        "doc_title": "其他病历.pdf",
        "content": "李女士治疗方案：阿莫西林克拉维酸钾0.6g iv q8h；氨溴索30mg iv bid。",
    },
    {
        "chunk_id": "dk-3",
        "doc_id": "doc-distractor",
        "doc_title": "其他病历.pdf",
        "content": "李女士出院时体温正常，复查胸片提示炎症吸收良好。",
    },
    {
        "chunk_id": "dk-4",
        "doc_id": "doc-guideline",
        "doc_title": "心衰诊疗指南.pdf",
        # 刻意不列出类名：否则这一个块会"意外答对"类名聚合题，
        # 把本该跨块的问题变成单块问题（第一版评测就踩了这个坑）
        "content": "慢性心力衰竭的药物治疗以四联疗法为基础，兼顾神经内分泌抑制与代谢改善。",
    },
    {
        "chunk_id": "dk-5",
        "doc_id": "doc-guideline",
        "doc_title": "心衰诊疗指南.pdf",
        "content": "所有心衰患者均应定期监测肾功能与电解质，起始用药后1-2周复查。",
    },
    {
        "chunk_id": "dk-6",
        "doc_id": "doc-guideline",
        "doc_title": "心衰诊疗指南.pdf",
        "content": "对于合并房颤的患者，需评估抗凝治疗指征，常用华法林或新型口服抗凝药。",
    },
    {
        "chunk_id": "dk-7",
        "doc_id": "doc-guideline",
        "doc_title": "心衰诊疗指南.pdf",
        "content": "生活方式干预包括限盐、限水、戒烟限酒及适度运动康复。",
    },
    {
        "chunk_id": "dk-8",
        "doc_id": "doc-distractor",
        "doc_title": "其他病历.pdf",
        "content": "患者王先生，男，58岁，因血糖控制不佳就诊，糖化血红蛋白8.9%。",
    },
]


# =============================================================================
# 知识图谱（人工标注）
# 每个三元组的证据都必须能在 CORPUS 里找到 —— 有测试校验这一点。
# =============================================================================

# (实体名, 类型, 产出该实体的块)
ENTITIES: List[Dict[str, Any]] = [
    # 患者与症状
    {"name": "张建国", "type": "患者", "chunk_id": "ck-1", "attributes": {"性别": "男", "年龄": "62"}},
    {"name": "双下肢水肿", "type": "症状", "chunk_id": "ck-2", "attributes": {}},

    # 诊断
    {"name": "扩张型心肌病", "type": "诊断", "chunk_id": "ck-3", "attributes": {}},
    {"name": "慢性心力衰竭急性加重", "type": "诊断", "chunk_id": "ck-7", "attributes": {"分级": "NYHA IV级"}},
    {"name": "心房颤动", "type": "诊断", "chunk_id": "ck-7", "attributes": {}},
    {"name": "高血压病3级", "type": "诊断", "chunk_id": "ck-8", "attributes": {"危险分层": "极高危"}},
    {"name": "2型糖尿病", "type": "诊断", "chunk_id": "ck-8", "attributes": {}},
    {"name": "轻度贫血", "type": "诊断", "chunk_id": "ck-8", "attributes": {}},

    # 长期用药
    {"name": "苯磺酸氨氯地平片", "type": "药物", "chunk_id": "ck-4", "attributes": {"剂量": "5mg qd"}},
    {"name": "二甲双胍", "type": "药物", "chunk_id": "ck-4", "attributes": {"剂量": "0.5g bid"}},

    # 住院用药
    {"name": "呋塞米注射液", "type": "药物", "chunk_id": "ck-5", "attributes": {"剂量": "40mg iv bid"}},
    {"name": "螺内酯片", "type": "药物", "chunk_id": "ck-5", "attributes": {"剂量": "20mg po qd"}},
    {"name": "琥珀酸美托洛尔缓释片", "type": "药物", "chunk_id": "ck-5", "attributes": {"剂量": "23.75mg po qd"}},
    {"name": "沙库巴曲缬沙坦钠片", "type": "药物", "chunk_id": "ck-6", "attributes": {"剂量": "50mg po bid"}},
    {"name": "达格列净片", "type": "药物", "chunk_id": "ck-6", "attributes": {"剂量": "10mg po qd"}},
    {"name": "华法林钠片", "type": "药物", "chunk_id": "ck-6", "attributes": {"剂量": "2.5mg po qd"}},

    # 药物类别（说明书里的指代实体）
    {"name": "ARNI", "type": "药物类别", "chunk_id": "ck-11", "attributes": {}},
    {"name": "SGLT2抑制剂", "type": "药物类别", "chunk_id": "ck-13", "attributes": {}},

    # 机构
    {"name": "诺华", "type": "机构", "chunk_id": "ck-11", "attributes": {}},
    {"name": "阿斯利康", "type": "机构", "chunk_id": "ck-13", "attributes": {}},

    # 不良反应
    {"name": "低血压", "type": "不良反应", "chunk_id": "ck-12", "attributes": {}},
    {"name": "高钾血症", "type": "不良反应", "chunk_id": "ck-12", "attributes": {}},
    {"name": "肾功能恶化", "type": "不良反应", "chunk_id": "ck-12", "attributes": {}},
]

# (头实体, 关系, 尾实体, 证据所在块)
TRIPLES: List[Dict[str, str]] = [
    # 患者 -> 疾病
    {"head": "张建国", "relation": "患有", "tail": "扩张型心肌病", "chunk_id": "ck-3"},
    {"head": "张建国", "relation": "患有", "tail": "慢性心力衰竭急性加重", "chunk_id": "ck-7"},
    {"head": "张建国", "relation": "患有", "tail": "心房颤动", "chunk_id": "ck-7"},
    {"head": "张建国", "relation": "患有", "tail": "高血压病3级", "chunk_id": "ck-8"},
    {"head": "张建国", "relation": "患有", "tail": "2型糖尿病", "chunk_id": "ck-8"},
    {"head": "张建国", "relation": "患有", "tail": "轻度贫血", "chunk_id": "ck-8"},
    {"head": "张建国", "relation": "表现为", "tail": "双下肢水肿", "chunk_id": "ck-2"},

    # 患者 -> 药物
    {"head": "张建国", "relation": "服用", "tail": "苯磺酸氨氯地平片", "chunk_id": "ck-4"},
    {"head": "张建国", "relation": "服用", "tail": "二甲双胍", "chunk_id": "ck-4"},
    {"head": "张建国", "relation": "服用", "tail": "呋塞米注射液", "chunk_id": "ck-5"},
    {"head": "张建国", "relation": "服用", "tail": "螺内酯片", "chunk_id": "ck-5"},
    {"head": "张建国", "relation": "服用", "tail": "琥珀酸美托洛尔缓释片", "chunk_id": "ck-5"},
    {"head": "张建国", "relation": "服用", "tail": "沙库巴曲缬沙坦钠片", "chunk_id": "ck-6"},
    {"head": "张建国", "relation": "服用", "tail": "达格列净片", "chunk_id": "ck-6"},
    {"head": "张建国", "relation": "服用", "tail": "华法林钠片", "chunk_id": "ck-6"},

    # 疾病 -> 药物（治疗关系）
    {"head": "扩张型心肌病", "relation": "治疗", "tail": "沙库巴曲缬沙坦钠片", "chunk_id": "ck-6"},
    {"head": "2型糖尿病", "relation": "治疗", "tail": "二甲双胍", "chunk_id": "ck-4"},
    {"head": "高血压病3级", "relation": "治疗", "tail": "苯磺酸氨氯地平片", "chunk_id": "ck-4"},
    {"head": "心房颤动", "relation": "治疗", "tail": "华法林钠片", "chunk_id": "ck-6"},

    # 药物 -> 类别（这两条是跨"药名 ↔ 类名"的关键桥）
    {"head": "沙库巴曲缬沙坦钠片", "relation": "属于", "tail": "ARNI", "chunk_id": "ck-6"},
    {"head": "达格列净片", "relation": "属于", "tail": "SGLT2抑制剂", "chunk_id": "ck-6"},

    # 类别 -> 机构（说明书里只有类名，没有药名 —— 见"决策 2"）
    {"head": "ARNI", "relation": "研制", "tail": "诺华", "chunk_id": "ck-11"},
    {"head": "SGLT2抑制剂", "relation": "研制", "tail": "阿斯利康", "chunk_id": "ck-13"},

    # 类别 -> 不良反应
    {"head": "ARNI", "relation": "导致", "tail": "低血压", "chunk_id": "ck-12"},
    {"head": "ARNI", "relation": "导致", "tail": "高钾血症", "chunk_id": "ck-12"},
    {"head": "ARNI", "relation": "导致", "tail": "肾功能恶化", "chunk_id": "ck-12"},
]


# =============================================================================
# 评测问题集
#
# `expected_terms` 是判分依据：检索到的证据里必须出现这些词串。
# 多跳/聚合题的 terms 刻意**跨块**，让 `single_chunk_ceiling < 1`。
# =============================================================================

QUESTIONS: List[Dict[str, Any]] = [
    # ---------------- 细节型：单块可答（切块检索的主场） ----------------
    {
        "id": "q01",
        "question": "张建国的左室射血分数是多少？",
        "type": "detail",
        "expected_terms": ["38%"],
        "note": "仅 ck-3 含答案",
    },
    {
        "id": "q02",
        "question": "沙库巴曲缬沙坦钠片的剂量是多少？",
        "type": "detail",
        "expected_terms": ["50mg po bid"],
        "note": "仅 ck-6 含答案",
    },
    {
        "id": "q03",
        "question": "患者既往高血压病史有多久？",
        "type": "detail",
        "expected_terms": ["15年"],
        "note": "仅 ck-2 含答案",
    },
    {
        "id": "q04",
        "question": "出院时射血分数复查结果如何？",
        "type": "detail",
        "expected_terms": ["42%"],
        "note": "仅 ck-9 含答案",
    },

    # ---------------- 多跳型：答案跨块 ----------------
    {
        "id": "q05",
        "question": "张建国服用的沙库巴曲缬沙坦钠片是出自哪家公司？",
        "type": "multihop",
        "expected_terms": ["沙库巴曲缬沙坦钠片", "诺华"],
        "note": "药名在 ck-6，研制方在 ck-11（且 ck-11 只写'该ARNI类药物'，不写药名）",
    },
    {
        "id": "q06",
        "question": "张建国服用的达格列净片属于哪类药物，由哪家公司研制？",
        "type": "multihop",
        "expected_terms": ["达格列净片", "SGLT2抑制剂", "阿斯利康"],
        "note": "药名在 ck-6，类名与机构在 ck-13。三项跨两块 -> ceiling 2/3",
    },
    {
        "id": "q07",
        "question": "治疗张建国扩张型心肌病的药物有哪些不良反应？",
        "type": "multihop",
        "expected_terms": ["沙库巴曲缬沙坦钠片", "低血压"],
        "note": "药在 ck-6，不良反应在 ck-12；路径需经 ARNI 桥接",
    },
    {
        "id": "q08",
        "question": "张建国和诺华公司之间是什么关系？",
        "type": "multihop",
        "expected_terms": ["沙库巴曲缬沙坦钠片", "诺华"],
        "note": "答案本身就是关系路径：张建国-服用->药-属于->ARNI-研制->诺华",
    },
    {
        "id": "q09",
        "question": "张建国所患的2型糖尿病用了什么药治疗？",
        "type": "multihop",
        "expected_terms": ["2型糖尿病", "二甲双胍"],
        "note": "诊断在 ck-8，用药在 ck-4",
    },

    # ---------------- 聚合型：需要汇总多个块 ----------------
    {
        "id": "q10",
        "question": "文档中一共提到了哪些药物类别？",
        "type": "global",
        "expected_terms": ["ARNI", "SGLT2抑制剂"],
        "note": "两个类名分别在 ck-11 与 ck-13，无单块含全部",
    },
    {
        "id": "q11",
        "question": "文档中一共涉及哪几家制药公司？",
        "type": "global",
        "expected_terms": ["诺华", "阿斯利康"],
        "note": "两家机构分别在 ck-11 与 ck-13",
    },
    {
        "id": "q12",
        "question": "张建国一共被诊断出哪些疾病？",
        "type": "global",
        "expected_terms": ["扩张型心肌病", "心房颤动", "高血压病3级", "2型糖尿病"],
        "note": "诊断分布在 ck-3、ck-7、ck-8 三块",
    },
    {
        "id": "q13",
        "question": "病历中一共记录了几种住院治疗药物？",
        "type": "global",
        "expected_terms": ["呋塞米注射液", "螺内酯片", "沙库巴曲缬沙坦钠片", "达格列净片"],
        "note": "住院药分布在 ck-5 与 ck-6 两块",
    },
    {
        "id": "q14",
        "question": "张建国一共服用了几种药物？",
        "type": "global",
        "expected_terms": ["二甲双胍", "苯磺酸氨氯地平片", "华法林钠片", "呋塞米注射液"],
        "note": "长期药在 ck-4、住院药在 ck-5/ck-6",
    },
]


def corpus_by_chunk_id() -> Dict[str, Dict[str, Any]]:
    """返回 {chunk_id: chunk}，方便按块定位证据。"""
    return {chunk["chunk_id"]: chunk for chunk in CORPUS}
