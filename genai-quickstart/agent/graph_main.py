# agent/graph_main.py
import os
import re
import json
import requests
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from jsonschema import validate, ValidationError

# ========= 环境与常量 =========
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "sk-local")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:9000/v1")
REPORT_SCHEMA_PATH = "data/schema/report.schema.json"

# LangGraph 的 State：用 TypedDict，invoke 返回 dict，方便直接 json.dumps
class State(TypedDict, total=False):
    query: str
    contexts: List[str]
    report_json: Dict[str, Any]
    error: str

# 我们期望 LLM 只输出这些键；用于最小修正与裁剪
REQUIRED_KEYS = ["设备", "部件", "问题描述", "缺陷类别", "位置", "严重等级", "建议", "依据"]


# ========= 规则推断（兜底补全） =========
def _infer_from_rules(query: str, contexts: Optional[List[str]] = None) -> Dict[str, Any]:
    """从用户问题和 SOP 文本里做最小可用的规则推断，返回可合并到 JSON 的字段"""
    contexts = contexts or []
    ctx_text = " ".join(contexts)

    # 设备/部件
    dev = ""
    m_dev = re.search(r"机台[ A-Za-z0-9]+", query)
    if m_dev:
        dev = m_dev.group(0).replace(" ", "")

    part = "主轴" if "主轴" in query else ""

    # 观测温度 & 阈值
    obs = None
    m_obs = re.search(r"(\d+)\s*℃", query)
    if m_obs:
        try:
            obs = int(m_obs.group(1))
        except Exception:
            pass

    thr = None
    m_thr = re.search(r"阈值[:：]?\s*(\d+)\s*℃", ctx_text)
    if m_thr:
        try:
            thr = int(m_thr.group(1))
        except Exception:
            pass
    # 常见 SOP 缺省阈值（若没从上下文里解析到）
    if thr is None:
        thr = 75

    # 噪音
    has_noise = "噪音" in query

    sev = 3
    suggestion = ""
    basis = ""
    defect = ""
    loc = part or ""

    if obs is not None and thr is not None:
        delta = obs - thr
        # 严重等级规则
        if delta > 15:
            sev = 5
        elif delta >= 5:
            sev = 4
        elif delta > 0:
            sev = 2
        else:
            sev = 1
        # 噪音加剧
        if has_noise and sev < 4:
            sev = 4

        if has_noise or sev >= 4:
            suggestion = "立即停机，检查润滑与冷却回路，联系维修并报修；复测并记录温度"
        elif sev == 2:
            suggestion = "降速观察并复测5分钟；检查冷却/润滑是否正常"
        else:
            suggestion = "按SOP正常巡检并记录"

        basis = f"SOP：阈值{thr}℃；"
        if has_noise:
            basis += "出现噪音同时过温→立即停机并报修"
        else:
            if 5 <= delta <= 15:
                basis += "超出 5~15℃ → 停机检查"
            elif delta > 15:
                basis += "超出 >15℃ → 严重异常"
            elif 0 < delta < 5:
                basis += "超出 ≤5℃ → 降速观察与复测"
            else:
                basis += "未超阈值"

        if delta > 0 and has_noise:
            defect = "过温/噪音异常"
        elif delta > 0:
            defect = "过温"
        elif has_noise:
            defect = "噪音异常"
        else:
            defect = ""
    else:
        # 没法计算温差：给个通用建议
        sev = 3 if has_noise else 2
        suggestion = "按SOP检查冷却与润滑回路，复测并记录；必要时停机"
        basis = "SOP通用处置：结合症状与阈值执行相应措施"
        defect = "噪音异常" if has_noise else ""

    return {
        "设备": dev,
        "部件": part,
        "缺陷类别": defect,
        "位置": loc,
        "严重等级": sev,
        "建议": suggestion,
        "依据": basis,
    }


# ========= 解析/修正/校验 =========
def _extract_json(text: str) -> Dict[str, Any]:
    """
    尽量从任意 LLM 输出里抽出首个合法 JSON 对象：
    - 去除```json 代码块
    - 先整体 parse
    - 再用括号配平截取第一个 {...}
    - 再用正则回退
    """
    if not isinstance(text, str):
        raise ValueError("LLM 返回内容不是字符串")

    # 去掉 ```json ... ``` 包裹
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)

    # 1) 直接尝试整体解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) 用括号配平找第一个完整 JSON 对象
    start, depth = None, 0
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    cand = text[start : i + 1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        start = None  # 继续找下一个候选

    # 3) 退一步：匹配所有最短 {...} 片段，逐个尝试
    for m in re.finditer(r"\{.*?\}", text, re.S):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue

    raise ValueError("无法从 LLM 输出中提取合法 JSON")


def _coerce_to_schema(d: Dict[str, Any], query: str, contexts: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    补齐字段、类型修正、去除多余字段，以满足 schema（最小修正策略）
    - 仅保留 REQUIRED_KEYS
    - “问题描述”缺失时用 query 兜底
    - “严重等级”确保为 1~5 的整数（默认 3）
    - 对于空字段，使用规则推断做兜底填充
    """
    if not isinstance(d, dict):
        d = {}
    out = {k: d.get(k, "") for k in REQUIRED_KEYS}

    # 问题描述兜底
    if not out.get("问题描述"):
        out["问题描述"] = query

    # 规则兜底
    inferred = _infer_from_rules(query, contexts)
    for k, v in inferred.items():
        if (k == "严重等级"):
            # 单独处理严重等级
            sev = out.get("严重等级")
            if not isinstance(sev, int):
                try:
                    sev = int(sev)
                except Exception:
                    sev = None
            if not isinstance(sev, int) or not (1 <= sev <= 5):
                out["严重等级"] = int(v)
        else:
            if not out.get(k):
                out[k] = v

    # 最终再校正严重等级
    sev = out.get("严重等级")
    if not isinstance(sev, int):
        try:
            sev = int(sev)
        except Exception:
            sev = 3
    if not (1 <= sev <= 5):
        sev = 3
    out["严重等级"] = sev

    return out


# ========= 节点实现 =========
def retrieve(s: State) -> State:
    try:
        r = requests.get(
            "http://127.0.0.1:8020/ask",
            params={"q": s["query"]},
            timeout=20,
        )
        r.raise_for_status()
        s["contexts"] = r.json().get("contexts", [])
    except Exception as e:
        s["error"] = f"retriever_error: {e}"
    return s


def generate_report(s: State) -> State:
    # 约束输出：仅这些键、严格 JSON、未知留空
    schema_hint = {
        "设备": "", "部件": "", "问题描述": "", "缺陷类别": "", "位置": "",
        "严重等级": 3, "建议": "", "依据": ""
    }
    prompt = (
        "你是工厂质检助手。请根据上下文严格生成 JSON 报告。\n"
        f"上下文: {s.get('contexts', [])}\n"
        f"问题: {s.get('query','')}\n\n"
        "请从【问题】中抽取设备（如“机台A”）和部件（如“主轴”）；\n"
        "若出现温度与阈值（SOP 提供阈值 75℃），请：\n"
        "1) 计算超温幅度；2) 依据 SOP 判断严重等级：超出 5~15℃ 设为 4；≤5℃ 设为 2；>15℃ 设为 5；\n"
        "3) 若出现“噪音”，建议应包含“立即停机并报修”。\n"
        "4) “建议”给出可执行操作（停机/检查润滑与冷却/复测记录/联系维修）。\n"
        "5) “依据”引用 SOP 关键条款（如阈值 75℃ 和相应处置规则）。\n\n"
        f"输出要求：仅输出一个 JSON 对象，键必须且仅为：{REQUIRED_KEYS}；未知信息用空字符串；“严重等级”为 1~5 的整数。\n"
        f"示例模板：{json.dumps(schema_hint, ensure_ascii=False)}"
    )
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "qwen2.5-7b-lora",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,          # 稍微有点随机性，避免过度保守
        "max_tokens": 512,
    }

    # 简单重试：LLM 解析失败 → 再尝试一次（更低温度/更短生成）
    tries = 2
    for attempt in range(tries):
        try:
            r = requests.post(
                f"{OPENAI_API_BASE}/chat/completions",
                headers=headers,
                json=payload if attempt == 0 else {**payload, "temperature": 0.01, "max_tokens": 400},
                timeout=90,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            parsed = _extract_json(content)
            s["report_json"] = _coerce_to_schema(parsed, s.get("query", ""), s.get("contexts", []))
            return s
        except Exception as e:
            last_err = e

    # 两次都失败：完全用规则兜底生成一个可过 schema 的报告
    fallback = {
        "设备": "",
        "部件": "",
        "问题描述": s.get("query",""),
        "缺陷类别": "",
        "位置": "",
        "严重等级": 3,
        "建议": "",
        "依据": ""
    }
    s["report_json"] = _coerce_to_schema(fallback, s.get("query",""), s.get("contexts", []))
    s["error"] = f"llm_error: {last_err}"
    return s


def validate_json(s: State) -> State:
    try:
        schema = json.load(open(REPORT_SCHEMA_PATH, "r", encoding="utf-8"))
        try:
            validate(instance=s.get("report_json", {}), schema=schema)
        except ValidationError:
            # 再做一次自动修正并复验
            s["report_json"] = _coerce_to_schema(s.get("report_json", {}), s.get("query", ""), s.get("contexts", []))
            validate(instance=s["report_json"], schema=schema)
    except ValidationError as e:
        s["error"] = f"schema_error: {e.message}"
    except Exception as e:
        s["error"] = f"schema_load_error: {e}"
    return s


def on_error(s: State) -> str:
    return END


# ========= 编排与运行 =========
if __name__ == "__main__":
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("generate_report", generate_report)
    g.add_node("validate_json", validate_json)
    g.add_edge("retrieve", "generate_report")
    g.add_edge("generate_report", "validate_json")
    g.set_entry_point("retrieve")
    g.add_conditional_edges("validate_json", lambda s: END if not s.get("error") else on_error(s))

    app = g.compile()

    init: State = {"query": "机台A主轴温度85℃并伴随噪音，按SOP如何处理？"}
    out: State = app.invoke(init)
    print(json.dumps(out, ensure_ascii=False, indent=2))
