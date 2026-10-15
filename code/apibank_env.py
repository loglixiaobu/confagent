#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API-Bank 加载器 + 确定性 mock 工具环境。

设计要点（为什么这样做）
--------------------------------------------------------------------
API-Bank 的 level-3 记录链里，每一步都完整保存了：
    (api_name, input) -> output     以及一个 exception 字段

于是可以把它当作一个「确定性重放环境」，**不必实现 73 个真实 API**：
    模型的实际调用与 gold 调用比对
      ├─ 匹配 -> 返回记录中的真实 output
      └─ 不匹配 -> 返回一个异常，并分类失败原因

【标签定义（本项目最关键的约定，答辩必被问）】
--------------------------------------------------------------------
自由文本参数（如 EmailReminder 的 content）不可能被逐字复现，若用精确字符串
匹配会产生大量**假阴性**，直接污染校准标签。因此采用**字段感知匹配**：

  结构化字段（时间/地点/收件人/ID/姓名等）  -> 严格匹配（时间做规范化）
  自由文本字段（长句、多词）               -> 实体覆盖率 / Jaccard 重叠

同时输出两套标签，用于**敏感性分析**：
  matched_gold         宽松口径（主标签）
  strict_matched_gold  严格口径（全字段精确匹配）

另外，模型重试时步骤位置会与 gold 错位，因此采用**按 api_name 贪心对齐**，
而不是简单的位置对齐。
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
from pathlib import Path
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# 类型映射：API-Bank 的参数类型 -> JSON Schema 类型
# ----------------------------------------------------------------------------
TYPE_MAP = {
    "str": "string", "string": "string",
    "int": "integer", "integer": "integer",
    "float": "number", "number": "number",
    "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array",
    "dict": "object", "object": "object",
}

# 自由文本判定阈值
FREE_TEXT_MIN_LEN = 25
FREE_TEXT_MIN_WORDS = 4

# 时间类字符串的规范化
_DT_PATTERNS = [
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2}):(\d{2})(?::(\d{2}))?$"),
     lambda m: "%04d-%02d-%02d %02d:%02d:%02d" % (
         int(m.group(1)), int(m.group(2)), int(m.group(3)),
         int(m.group(4)), int(m.group(5)), int(m.group(6) or 0))),
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"),
     lambda m: "%04d-%02d-%02d 00:00:00" % (int(m.group(1)), int(m.group(2)), int(m.group(3)))),
]


def parameter_schema(type_name: str) -> Dict[str, Any]:
    """Translate API-Bank types, including list(str), without silently losing items."""
    name = type_name.lower().strip()
    match = re.fullmatch(r"list(?:\((.+)\)|\[(.+)\])", name)
    if match:
        return {"type": "array", "items": parameter_schema(match[1] or match[2])}
    kind = TYPE_MAP.get(name)
    if kind is None:
        raise ValueError("Unsupported API parameter type: %s" % type_name)
    return {"type": kind, **({"items": {}} if kind == "array" else {})}


def api_schema_to_tool(api: Dict[str, Any]) -> Dict[str, Any]:
    """把 API-Bank 的 API 描述转成 OpenAI function-calling 的 tool schema。"""
    props: Dict[str, Any] = {}
    for pname, pmeta in (api.get("input_parameters") or {}).items():
        props[pname] = {
            **parameter_schema(str(pmeta.get("type", "str"))),
            "description": pmeta.get("description", ""),
        }
    return {
        "type": "function",
        "function": {
            "name": api.get("name"),
            "description": api.get("description", ""),
            "parameters": {"type": "object", "properties": props, "required": list(props.keys())},
        },
    }


# ----------------------------------------------------------------------------
# 值比较工具
# ----------------------------------------------------------------------------
def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)


def _norm_time(v: Any) -> Any:
    """把时间类字符串规范化，避免 '2021-01-01 10:00' 与 '2021-01-01 10:00:00' 被判不等。"""
    if not isinstance(v, str):
        return v
    s = v.strip()
    for pat, fmt in _DT_PATTERNS:
        m = pat.match(s)
        if m:
            return fmt(m)
    # 处理 "10:00 AM" 之类
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2}):(\d{2})\s*(AM|PM)$", s, re.I)
    if m:
        hh = int(m.group(4)) % 12
        if m.group(6).upper() == "PM":
            hh += 12
        return "%04d-%02d-%02d %02d:%02d:00" % (
            int(m.group(1)), int(m.group(2)), int(m.group(3)), hh, int(m.group(5)))
    return s


def _is_free_text(v: Any) -> bool:
    if not isinstance(v, str):
        return False
    s = v.strip()
    return len(s) >= FREE_TEXT_MIN_LEN and len(s.split()) >= FREE_TEXT_MIN_WORDS


def _salient_tokens(s: str) -> set:
    """抽取"有意义"的词：长度 > 3 或含数字。用于自由文本的实体覆盖率比较。"""
    toks = re.findall(r"[A-Za-z0-9]+", str(s).lower())
    return {t for t in toks if len(t) > 3 or any(c.isdigit() for c in t)}


def text_match(model_v: Any, gold_v: Any) -> bool:
    """自由文本字段的**宽松**判定：只检查"非空 + 主题相关"，不评判措辞。

    理由：像 EmailReminder 的 content 这类自然语言载荷，其具体措辞不是智能体
    能够"做对/做错"的决策——同一件事可以有无数种正确写法。若要求逐字或高重叠，
    会产生大量假阴性，直接污染校准标签。

    因此这里只做两条最低限度的检查：
      1) 模型给出的值非空且不是占位符；
      2) 与 gold 至少有 1 个有意义词（长度>3 或含数字）重叠，即"在说同一件事"。
    措辞质量不计入判定；严格口径由 strict_matched_gold 另行记录，供敏感性分析。
    """
    if model_v is None:
        return False
    s = str(model_v).strip()
    if len(s) < 3:
        return False
    a, b = _salient_tokens(s), _salient_tokens(gold_v)
    if not b:
        return True
    return len(a & b) >= 1


def structured_equal(a: Any, b: Any) -> bool:
    return _canon(_norm_time(a)) == _canon(_norm_time(b))


def numeric_equal(a: Any, b: Any, integer: bool = False) -> bool:
    """Normalize numeric *fields*, never arbitrary passwords/IDs or booleans."""
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    try:
        aa, bb = Decimal(str(a)), Decimal(str(b))
        return (aa.is_finite() and bb.is_finite() and aa == bb
                and (not integer or aa == aa.to_integral_value()))
    except (InvalidOperation, ValueError, TypeError):
        return False


def typed_equal(a: Any, b: Any, type_name: str = "str") -> bool:
    schema = parameter_schema(type_name)
    if schema["type"] in ("number", "integer"):
        return numeric_equal(a, b, schema["type"] == "integer")
    if schema["type"] == "array":
        if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
            return False
        item = re.fullmatch(r"list(?:\((.+)\)|\[(.+)\])", type_name.lower().strip())
        return all(typed_equal(x, y, item[1] or item[2]) if item
                   else structured_equal(x, y) for x, y in zip(a, b))
    return structured_equal(a, b)


def arithmetic_value(expression: Any) -> Decimal:
    """Evaluate a bounded arithmetic AST; no names, calls, powers, or eval."""
    if not isinstance(expression, str) or len(expression) > 256:
        raise ValueError("Not a bounded arithmetic expression")
    root = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(root)) > 80:
        raise ValueError("Expression too large")
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Decimal(str(node.value))
            if not value.is_finite() or abs(value) > Decimal('1e20'):
                raise ValueError("Constant out of bounds")
            return value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b
        raise ValueError("Unsupported arithmetic")
    result = visit(root)
    if not result.is_finite() or abs(result) > Decimal('1e30'):
        raise ValueError("Result out of bounds")
    return result


# ----------------------------------------------------------------------------
# 步骤结果
# ----------------------------------------------------------------------------
@dataclass
class StepResult:
    step_id: int
    api_name: str
    args: Dict[str, Any]
    observation: Any
    exception: Optional[str] = None

    matched_gold: bool = False            # 宽松口径（主标签）
    strict_matched_gold: bool = False     # 严格口径（敏感性分析用）
    matched_gold_step: Optional[int] = None
    gold_api_name: Optional[str] = None
    gold_args: Optional[Dict[str, Any]] = None
    failure_type: Optional[str] = None
    mismatch_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------
# Mock 环境
# ----------------------------------------------------------------------------
class APIBankMockEnv:
    """基于 API-Bank level-3 记录链的确定性重放环境。"""

    def __init__(self, task: Dict[str, Any],
                 distractor_apis: Optional[List[Dict[str, Any]]] = None) -> None:
        self.requirement: str = task.get("requirement", "")
        self.gold_response: str = task.get("response", "")
        self.gold_steps: List[Dict[str, Any]] = task.get("apis", [])
        self.task_id: str = self.requirement.strip()[:60]

        # 精确匹配表（原始参数）——保留作为快速路径
        self._exact: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for st in self.gold_steps:
            self._exact[(st.get("api_name"), _canon(st.get("input")))] = st.get("output") or {}

        self._consumed: List[bool] = [False] * len(self.gold_steps)

        # 工具注册表
        self.tool_registry: Dict[str, Dict[str, Any]] = {}
        for st in self.gold_steps:
            if st.get("api_name") == "ToolSearcher":
                out = (st.get("output") or {}).get("output")
                if isinstance(out, dict) and out.get("name"):
                    self.tool_registry[out["name"]] = out
        for api in distractor_apis or []:
            if api.get("name") and api["name"] not in self.tool_registry:
                self.tool_registry[api["name"]] = api

        self.history: List[StepResult] = []
        self._step_id = 0

    # -- 工具暴露 -----------------------------------------------------------
    def tools(self) -> List[Dict[str, Any]]:
        searcher = {
            "type": "function",
            "function": {
                "name": "ToolSearcher",
                "description": ("Search for a suitable API by keywords. Returns the API name, "
                                "description and its input parameters."),
                "parameters": {
                    "type": "object",
                    "properties": {"keywords": {"type": "string",
                                                "description": "Keywords describing the capability you need."}},
                    "required": ["keywords"],
                },
            },
        }
        return [searcher] + [api_schema_to_tool(a) for a in self.tool_registry.values()]

    # -- 执行一步 -----------------------------------------------------------
    def call(self, api_name: str, args: Dict[str, Any]) -> StepResult:
        self._step_id += 1
        sid = self._step_id

        # ToolSearcher：模拟 API 检索
        if api_name == "ToolSearcher":
            kws = str(args.get("keywords", ""))
            matched_api = self._search(kws)
            gi = self._align("ToolSearcher")
            res = StepResult(
                step_id=sid, api_name=api_name, args=args,
                observation=matched_api,
                exception=None if matched_api else "No API found for keywords: %s" % kws,
                matched_gold=gi is not None,
                strict_matched_gold=gi is not None,
                matched_gold_step=gi,
                gold_api_name="ToolSearcher" if gi is not None else None,
                gold_args=(self.gold_steps[gi].get("input") if gi is not None else None),
                failure_type=None if matched_api else "search_miss",
            )
            if gi is not None:
                self._consumed[gi] = True
            self.history.append(res)
            return res

        # 普通 API：先按 api_name 贪心对齐到未消费的 gold 步骤
        gi = self._align(api_name, args)
        if gi is None:
            ftype, msg, fields = self._diagnose_no_align(api_name, args)
            res = StepResult(
                step_id=sid, api_name=api_name, args=args, observation=None,
                exception=msg, matched_gold=False, strict_matched_gold=False,
                failure_type=ftype, mismatch_fields=fields,
            )
            self.history.append(res)
            return res

        gold = self.gold_steps[gi]
        gargs = gold.get("input") or {}
        relaxed_ok, strict_ok, fields = self._compare_args(args, gargs, api_name)

        if relaxed_ok:
            rec = gold.get("output") or {}
            res = StepResult(
                step_id=sid, api_name=api_name, args=args,
                observation=rec.get("output"), exception=rec.get("exception"),
                matched_gold=True, strict_matched_gold=strict_ok,
                matched_gold_step=gi, gold_api_name=api_name, gold_args=gargs,
            )
            self._consumed[gi] = True
        else:
            ftype = self._classify(fields, args, gargs)
            res = StepResult(
                step_id=sid, api_name=api_name, args=args, observation=None,
                exception="Call does not match the expected step. Problem field(s): %s" % (fields or "?"),
                matched_gold=False, strict_matched_gold=False,
                matched_gold_step=gi, gold_api_name=api_name, gold_args=gargs,
                failure_type=ftype, mismatch_fields=fields,
            )
            # 不消费该 gold 步骤，允许模型重试
        self.history.append(res)
        return res

    # -- 内部 ---------------------------------------------------------------
    def _align(self, api_name: str, args: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """按 api_name 贪心对齐到第一个未消费的 gold 步骤。"""
        candidates = [i for i, st in enumerate(self.gold_steps)
                      if not self._consumed[i] and st.get("api_name") == api_name]
        # Allow interchangeable calls in one repeated-API block, not jumps across dependencies.
        if candidates and args is not None:
            first = candidates[0]
            for i in candidates:
                intervening = [st.get("api_name") for j, st in enumerate(self.gold_steps[first:i])
                               if not self._consumed[first + j]]
                if any(n not in (api_name, "ToolSearcher") for n in intervening):
                    break
                if self._compare_args(args, self.gold_steps[i].get("input") or {}, api_name)[0]:
                    return i
        for i, st in enumerate(self.gold_steps):
            if not self._consumed[i] and st.get("api_name") == api_name:
                return i
        return None

    def _compare_args(self, args: Dict[str, Any], gargs: Dict[str, Any],
                      api_name: str = "") -> Tuple[bool, bool, List[str]]:
        """返回 (宽松是否通过, 严格是否通过, 不匹配字段列表)。"""
        missing = [k for k in gargs if k not in args]
        if missing:
            return False, False, ["missing:" + k for k in missing]

        bad_relaxed, bad_strict = [], []
        metadata = (self.tool_registry.get(api_name) or {}).get("input_parameters") or {}
        for k, gv in gargs.items():
            mv = args[k]
            if api_name == "Calculator" and k == "formula":
                try:
                    target = arithmetic_value(gv)
                    # One reference contains "30 / 3" for likes 70+80+90.
                    # When all actual like results are available, the explicit
                    # user request to total them supplies an independent oracle.
                    if 'total' in self.requirement.lower() and 'likes' in self.requirement.lower():
                        expected_ids = {str(s.get('input', {}).get('post_id')) for s in self.gold_steps
                                        if s.get('api_name') == 'LikeCount'}
                        observed = {str(s.args.get('post_id')): s.observation['like_count']
                                    for s in self.history if s.api_name == 'LikeCount' and s.matched_gold
                                    and isinstance(s.observation, dict) and 'like_count' in s.observation}
                        if expected_ids and set(observed) == expected_ids:
                            target = sum((Decimal(str(v)) for v in observed.values()), Decimal(0))
                    equal = arithmetic_value(mv) == target
                except (ValueError, SyntaxError, ArithmeticError):
                    equal = False
                if not equal:
                    bad_relaxed.append(k)
                if not structured_equal(mv, gv):
                    bad_strict.append(k)
            elif api_name == "PersonalInfoUpdate" and k == "address":
                if str(mv).strip().rstrip('.') != str(gv).strip().rstrip('.'):
                    bad_relaxed.append(k)
                if not structured_equal(mv, gv):
                    bad_strict.append(k)
            elif _is_free_text(gv):
                if not text_match(mv, gv):
                    bad_relaxed.append(k)
                if not structured_equal(mv, gv):
                    bad_strict.append(k)
            else:
                if not typed_equal(mv, gv, metadata.get(k, {}).get("type", "str")):
                    bad_relaxed.append(k)
                if not structured_equal(mv, gv):
                    bad_strict.append(k)
        # 额外参数不算错（模型多传不影响正确性）
        return (len(bad_relaxed) == 0, len(bad_strict) == 0, bad_relaxed)

    def _classify(self, fields: List[str], args: Dict[str, Any], gargs: Dict[str, Any]) -> str:
        if any(f.startswith("missing:") for f in fields):
            return "missing_arg"
        return "wrong_value"

    def _diagnose_no_align(self, api_name: str, args: Dict[str, Any]) -> Tuple[str, str, List[str]]:
        remaining = [self.gold_steps[i].get("api_name")
                     for i in range(len(self.gold_steps)) if not self._consumed[i]]
        if api_name in [st.get("api_name") for st in self.gold_steps]:
            return ("extra_call",
                    "This API has already been completed; it is not needed again.", [])
        return ("wrong_api",
                "Wrong API. Expected one of: %s" % (remaining[:4] or "none"), [])

    def _search(self, keywords: str) -> Optional[Dict[str, Any]]:
        kw = keywords.lower().strip()
        if not kw:
            return None
        for name, api in self.tool_registry.items():
            if name.lower() == kw:
                return api
        for name, api in self.tool_registry.items():
            if kw in name.lower():
                return api
        for name, api in self.tool_registry.items():
            if kw in str(api.get("description", "")).lower():
                return api
        for w in kw.replace("_", " ").split():
            if not w:
                continue
            for name, api in self.tool_registry.items():
                if w in name.lower() or w in str(api.get("description", "")).lower():
                    return api
        return None

    # -- 终局 ---------------------------------------------------------------
    def _required_gold_indices(self) -> List[int]:
        """真正需要完成的 gold 步骤：排除 ToolSearcher（它只是检索辅助，不是任务步骤）。"""
        return [i for i, st in enumerate(self.gold_steps)
                if st.get("api_name") != "ToolSearcher"]

    def success(self) -> bool:
        """成功 = 全部**实质** gold 步骤（不含 ToolSearcher）被匹配，且没有失败步骤。"""
        req = self._required_gold_indices()
        if not req:
            return False
        all_matched = all(self._consumed[i] for i in req)
        no_fail = not any(s.matched_gold is False for s in self.history)
        return bool(all_matched and no_fail and self.history)

    def summary(self) -> Dict[str, Any]:
        req = self._required_gold_indices()
        return {
            "task_id": self.task_id,
            "n_gold_steps": len(self.gold_steps),
            "n_required_steps": len(req),
            "n_actual_steps": len(self.history),
            "n_matched": sum(1 for s in self.history if s.matched_gold),
            "n_matched_strict": sum(1 for s in self.history if s.strict_matched_gold),
            "n_required_matched": sum(1 for i in req if self._consumed[i]),
            "success": self.success(),
            "failure_types": [s.failure_type for s in self.history if s.failure_type],
        }


# ----------------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------------
def load_level3(data_dir: str, split: str = "test", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    fname = "level-3.json" if split == "test" else "lv3-train.json"
    path = os.path.join(data_dir, "test-data" if split == "test" else "training-data", fname)
    with open(path, encoding="utf-8") as source:
        tasks = json.load(source)
    tasks = [t for t in tasks if isinstance(t, dict) and t.get("apis")]
    return tasks[:limit] if limit else tasks


def collect_distractor_apis(all_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pool: Dict[str, Dict[str, Any]] = {}
    for t in all_tasks:
        for st in t.get("apis", []):
            if st.get("api_name") == "ToolSearcher":
                out = (st.get("output") or {}).get("output")
                if isinstance(out, dict) and out.get("name"):
                    pool[out["name"]] = out
    return list(pool.values())


def dataset_stats(data_dir: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in sorted(glob.glob(os.path.join(data_dir, "*/**/*.json"), recursive=True)):
        try:
            d = json.load(open(p, encoding="utf-8"))
            out[os.path.relpath(p, data_dir)] = len(d) if hasattr(d, "__len__") else None
        except Exception as e:
            out[os.path.relpath(p, data_dir)] = "ERR: %s" % e
    l3 = load_level3(data_dir, "test")
    chains = [len(t["apis"]) for t in l3]
    out["_level3_chains"] = {
        "n_tasks": len(l3), "total_calls": sum(chains),
        "avg_chain_len": round(sum(chains) / len(chains), 2) if chains else 0,
        "max_chain_len": max(chains) if chains else 0,
        "unique_apis": len(collect_distractor_apis(l3)),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data/api-bank"))
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    print(json.dumps(dataset_stats(args.data_dir), ensure_ascii=False, indent=2))
    if not args.demo:
        return

    print()
    print("=" * 72)
    print("字段感知匹配自检")
    print("=" * 72)
    tasks = load_level3(args.data_dir, "test")
    task = tasks[0]
    env = APIBankMockEnv(task)

    # 用例1：gold 全对
    print("--- 用例1: 完全按 gold 调用 ---")
    for st in task["apis"]:
        r = env.call(st["api_name"], dict(st["input"]))
    print("   success=%s matched=%d/%d" % (env.success(), sum(1 for s in env.history if s.matched_gold), len(task["apis"])))

    # 用例2：content 换成不同措辞（应当仍然判对）
    print("--- 用例2: 自由文本换措辞（应判对，验证假阴性已消除） ---")
    env = APIBankMockEnv(task)
    for st in task["apis"]:
        a = dict(st["input"])
        if st["api_name"] == "EmailReminder":
            a["content"] = ("Hi, just a reminder that your meeting with the client is at "
                            "2021-01-01 10:00:00 in Room 1. Please be on time.")
        r = env.call(st["api_name"], a)
        print("   %-14s matched=%-5s strict=%-5s fields=%s" % (r.api_name, r.matched_gold, r.strict_matched_gold, r.mismatch_fields))
    print("   success=%s" % env.success())

    # 用例3：结构化字段改错（应判错）
    print("--- 用例3: 结构化字段改错（应判错） ---")
    env = APIBankMockEnv(task)
    for i, st in enumerate(task["apis"]):
        a = dict(st["input"])
        if i == 1 and "user_name" in a:
            a["user_name"] = "Mary"
        r = env.call(st["api_name"], a)
        print("   %-14s matched=%-5s failure=%-12s fields=%s" % (r.api_name, r.matched_gold, r.failure_type, r.mismatch_fields))
    print("   success=%s" % env.success())


if __name__ == "__main__":
    main()
