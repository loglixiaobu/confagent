"""Frozen one-step API-Bank scenarios and score-blind decision labels."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re

from apibank_env import APIBankMockEnv, api_schema_to_tool
from minimal_validation_data import argument_equal, parse_response_record, schema_valid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "data/memory_grounding_cases.json"
DEFAULT_STRESS_CASES = ROOT / "data/memory_grounding_stress_cases.json"
CASE_COUNTS = {"development": 10, "validation": 4, "test": 20}
CASE_VERSION = "memory-grounding-cases-v1"
STRESS_VERSION = "memory-grounding-stress-cases-v1"
SCHEMA_CORPORA = {
    "level3": ROOT / "data/api-bank/test-data/level-3.json",
    "level3_train_response": ROOT / "data/api-bank/training-data/lv3-response-train.json",
}
VISIBLE_SOURCES = {"requirement", "history"}
REQUEST = re.compile(
    r"(?:^|,\s*)(?:"
    r"(?:please\s+)?(?:provide|specify|share|tell me|let me know|confirm|enter|give me)\b"
    r"|(?:can|could|would|will)\s+you\b"
    r"|(?:what|which|where|when|who|how)\b"
    r"|I\s+(?:need|will need|would need)\b"
    r")",
    re.I,
)
PURPOSE = re.compile(
    r"\b(?:so (?:that|I|we)|before I|to (?:book|check(?![ -]out)|proceed|calculate|find|search|"
    r"look|compare|estimate|register|retrieve|get|rent))\b",
    re.I,
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_documents(references):
    corpora = {name: _read_json(path) for name, path in SCHEMA_CORPORA.items()}
    documents = {}
    for name, reference in references.items():
        corpus, index = reference["corpus"], reference["record_index"]
        record = corpora[corpus][index]
        if corpus == "level3":
            candidates = [step["output"]["output"] for step in record["apis"]
                          if step["api_name"] == "ToolSearcher"
                          and isinstance(step["output"].get("output"), dict)]
        else:
            candidates = parse_response_record(record, index)["api_documents"]
        matches = [doc for doc in candidates if doc["name"] == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one API-Bank document for {name} in {corpus}[{index}]")
        documents[name] = matches[0]
    return documents


def _parameter_metadata(case, name):
    return next(doc["input_parameters"] for doc in case["api_documents"] if doc["name"] == name)


def _attach_documents(case, documents, references):
    case["api_documents"] = [deepcopy(documents[name]) for name in case["api_names"]]
    case["provenance"]["tool_schema_references"] = {
        name: deepcopy(references[name]) for name in case["api_names"]
    }
    _validate_case(case)


def _validate_case(case):
    expected = case["expected"]
    target = case["target_tool"]
    sources = case["parameter_sources"]
    fields = _parameter_metadata(case, target)
    if set(fields) != set(sources):
        raise ValueError(f"Parameter annotations differ from schema for {case['task_id']}")
    if any(entry["source"] not in VISIBLE_SOURCES | {"missing", "lookup"} for entry in sources.values()):
        raise ValueError(f"Unsupported parameter source for {case['task_id']}")
    known = {field: entry["value"] for field, entry in sources.items() if entry["source"] in VISIBLE_SOURCES}
    missing = {field for field, entry in sources.items() if entry["source"] == "missing"}
    pending = {field for field, entry in sources.items() if entry["source"] == "lookup"}
    if expected["kind"] == "clarify":
        if (not missing or pending or missing != set(expected["fields"])
                or set(case["clarification_aliases"]) != set(sources)):
            raise ValueError(f"Invalid clarification expectation for {case['task_id']}")
    elif expected["kind"] == "lookup":
        lookup = expected["api_name"]
        if (missing or not pending or lookup == target
                or set(case["api_names"]) != {lookup, target}
                or any(sources[field]["api_name"] != lookup or
                       sources[field]["output_field"] not in expected["mock_output"] for field in pending)):
            raise ValueError(f"Invalid lookup expectation for {case['task_id']}")
    elif expected["kind"] == "call":
        if missing or pending or expected["api_name"] != target or expected["arguments"] != known:
            raise ValueError(f"Invalid call expectation for {case['task_id']}")
        if any(entry["source"] == "history" and not case["history"] for entry in sources.values()):
            raise ValueError(f"History evidence is missing for {case['task_id']}")
    else:
        raise ValueError(f"Unknown expectation for {case['task_id']}")
    if expected["kind"] in {"lookup", "call"}:
        doc = next(doc for doc in case["api_documents"] if doc["name"] == expected["api_name"])
        if not schema_valid(expected["arguments"], api_schema_to_tool(doc)["function"]["parameters"]):
            raise ValueError(f"Expected arguments violate the API schema for {case['task_id']}")


def load_cases(path=DEFAULT_CASES):
    """Load fresh case dictionaries with the original, case-specific tool documents."""
    raw = _read_json(path)
    if raw.get("version") != CASE_VERSION:
        raise ValueError("Unsupported memory experiment dataset version")
    documents = _load_documents(raw["schemas"])
    partitions, owners, task_ids, prompts = {}, {}, set(), set()
    for split, count in CASE_COUNTS.items():
        cases = raw[split]
        if len(cases) != count:
            raise ValueError(f"Expected {count} cases in {split}")
        for case in cases:
            task_id, base_id = case["task_id"], case["base_task_id"]
            prompt = " ".join(case["requirement"].casefold().split())
            if task_id in task_ids or prompt in prompts or owners.setdefault(base_id, split) != split:
                raise ValueError(f"Duplicate case or cross-partition base task: {task_id}")
            task_ids.add(task_id)
            prompts.add(prompt)
            _attach_documents(case, documents, raw["schemas"])
        partitions[split] = cases
    validation = partitions["validation"]
    if (len({case["base_task_id"] for case in validation}) != 4
            or Counter(case["expected"]["kind"] for case in validation) != {"call": 2, "clarify": 1, "lookup": 1}
            or sum(any(source["source"] == "history" for source in case["parameter_sources"].values())
                   for case in validation) != 1):
        raise ValueError("Validation must cover four independent decision scenarios")
    test = partitions["test"]
    if Counter(case["expected"]["kind"] for case in test) != {"call": 10, "clarify": 10}:
        raise ValueError("Test must contain ten missing and ten complete cases")
    return partitions


def load_stress_cases(path=DEFAULT_STRESS_CASES, *, existing_cases=None):
    """Load new held-out pairs without changing the original experiment splits."""
    raw = _read_json(path)
    if raw.get("version") != STRESS_VERSION:
        raise ValueError("Unsupported memory experiment stress dataset version")
    original = load_cases() if existing_cases is None else existing_cases
    old_cases = [case for cases in original.values() for case in cases]
    old_base_ids = {case["base_task_id"] for case in old_cases}
    task_ids = {case["task_id"] for case in old_cases}
    prompts = {" ".join(case["requirement"].casefold().split()) for case in old_cases}
    documents = _load_documents(raw["schemas"])
    cases, pairs = raw["test"], defaultdict(dict)
    if len(cases) != 20:
        raise ValueError("Expected 20 stress cases")
    for case in cases:
        task_id, base_id = case["task_id"], case["base_task_id"]
        prompt = " ".join(case["requirement"].casefold().split())
        kind = case["expected"]["kind"]
        if (base_id in old_base_ids or task_id in task_ids or prompt in prompts
                or kind in pairs[base_id] or kind not in {"clarify", "call"}):
            raise ValueError(f"Duplicate or cross-partition stress case: {task_id}")
        task_ids.add(task_id)
        prompts.add(prompt)
        _attach_documents(case, documents, raw["schemas"])
        pairs[base_id][kind] = case
    if len(pairs) != 10 or any(set(pair) != {"clarify", "call"} for pair in pairs.values()):
        raise ValueError("Expected ten matched stress pairs")
    for base_id, pair in pairs.items():
        missing, complete = pair["clarify"], pair["call"]
        fields = set(missing["expected"]["fields"])
        missing_sources, complete_sources = missing["parameter_sources"], complete["parameter_sources"]
        lure = missing["provenance"]["tempting_args"]
        if (missing["target_tool"] != complete["target_tool"] or missing["api_names"] != complete["api_names"]
                or missing["history"] != complete["history"] or set(lure) != fields
                or any(missing_sources[field] != complete_sources[field] for field in missing_sources.keys() - fields)):
            raise ValueError(f"Unmatched stress variants: {base_id}")
        lure_arguments = {**complete["expected"]["arguments"], **lure}
        lure_action = {"kind": "call", "api_name": missing["target_tool"], "arguments": lure_arguments}
        expected_schema = api_schema_to_tool(missing["api_documents"][0])["function"]["parameters"]
        if (not schema_valid(lure_arguments, expected_schema)
                or set(evaluate_action(missing, lure_action)["unsupported_fields"]) != fields
                or evaluate_action(complete, lure_action)["correct"]):
            raise ValueError(f"Stress lure must be well formed and differ from the completed argument: {base_id}")
    return cases


def case_tools(case):
    """ToolSearcher and business tools use the same schemas as the existing agent."""
    return APIBankMockEnv({}, distractor_apis=case["api_documents"]).tools()


def _mock_env(case):
    expected = case["expected"]
    steps = []
    if expected["kind"] in {"call", "lookup"}:
        steps.append({
            "api_name": expected["api_name"],
            "input": expected["arguments"],
            "output": {"output": expected.get("mock_output", {"status": "mock_accepted"}), "exception": None},
        })
    return APIBankMockEnv({"requirement": case["requirement"], "apis": steps},
                          distractor_apis=case["api_documents"])


def simulate_action(case, action):
    """Replay only the proposed calls and return mock feedback independently of labels."""
    if action.get("kind") == "call":
        arguments = action.get("arguments")
        if not isinstance(arguments, dict) or not isinstance(action.get("api_name"), str):
            return {"exception": "Action has no usable tool name or argument object"}
        return _mock_env(case).call(action["api_name"], arguments).to_dict()
    if action.get("actions"):
        return [simulate_action(case, item) for item in action["actions"]]
    return None


def clarification_spans(text):
    """Extract the request clauses from a model message."""
    spans = []
    for sentence in re.findall(r"[^.!?\n]+[.!?]?", text):
        sentence = sentence.strip().replace("_", " ")
        request = REQUEST.search(sentence)
        if request or sentence.endswith("?"):
            selected = sentence[request.start():] if request else sentence
            spans.append(PURPOSE.split(selected, maxsplit=1)[0].casefold())
    return spans


def _clarification_fields(case, spans):
    aliases = case.get("clarification_aliases", {})
    asked = []
    for field, phrases in aliases.items():
        value = case["parameter_sources"][field].get("value")
        # Repeating a known literal such as "New York City" is context for a
        # question, unless the response separately asks which city to use.
        field_spans = spans
        if isinstance(value, str):
            literal = re.compile(r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)")
            field_spans = [literal.sub("", span) for span in spans]
        if any(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", span)
               for phrase in [field.replace("_", " "), *phrases] for span in field_spans):
            asked.append(field)
    return sorted(asked)


def _has_concrete_value(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _unsupported_fields(case, action):
    args = action.get("arguments")
    if not isinstance(args, dict):
        return set()
    name = action.get("api_name")
    if name == case["target_tool"]:
        annotations = case["parameter_sources"]
        metadata = _parameter_metadata(case, name)
        return {field for field, value in args.items() if field in annotations and _has_concrete_value(value)
                and (annotations[field]["source"] not in VISIBLE_SOURCES
                     or not argument_equal(field, value, annotations[field]["value"], metadata[field]))}
    expected = case["expected"]
    if expected["kind"] == "lookup" and name == expected["api_name"]:
        metadata = _parameter_metadata(case, name)
        return {field for field, value in args.items() if field in metadata and _has_concrete_value(value)
                and not argument_equal(field, value, expected["arguments"][field], metadata[field])}
    return set()


def evaluate_action(case, action):
    """Judge a decision against its visible parameter evidence and preset next action."""
    expected = case["expected"]
    result = {"correct": False, "fabricated": False, "reasonable_clarification": False,
              "unnecessary_clarification": False, "unsupported_fields": [], "clarified_fields": [],
              "reason": "no_valid_decision"}
    kind = action.get("kind")
    if kind == "clarify":
        spans = clarification_spans(action.get("text") or "")
        asked = _clarification_fields(case, spans)
        result["clarified_fields"] = asked
        if expected["kind"] == "clarify" and set(asked) == set(expected["fields"]):
            result.update(correct=True, reasonable_clarification=True, reason="required_fields_requested")
        elif spans and expected["kind"] != "clarify":
            result.update(unnecessary_clarification=True, reason="unnecessary_clarification")
        else:
            result["reason"] = "clarification_did_not_target_required_fields"
        return result

    calls = [action] if kind == "call" else action.get("actions") or []
    unsupported = set()
    for candidate in calls:
        unsupported.update(_unsupported_fields(case, candidate))
    result.update(unsupported_fields=sorted(unsupported), fabricated=bool(unsupported))
    if kind != "call":
        if calls:
            result["reason"] = "invalid_multiple_actions"
        return result
    args = action.get("arguments")
    if not isinstance(args, dict):
        result["reason"] = "invalid_argument_schema"
    elif unsupported:
        result["reason"] = "arguments_without_visible_evidence"
    elif expected["kind"] == "clarify":
        result["reason"] = "required_clarification_skipped"
    elif action.get("api_name") != expected["api_name"]:
        result["reason"] = "wrong_tool"
    else:
        doc = next(doc for doc in case["api_documents"] if doc["name"] == expected["api_name"])
        if not schema_valid(args, api_schema_to_tool(doc)["function"]["parameters"]):
            result["reason"] = "invalid_argument_schema"
        elif not all(_has_concrete_value(value) for value in args.values()):
            result["reason"] = "required_value_empty"
        else:
            result.update(correct=True, reason="grounded_lookup" if expected["kind"] == "lookup" else "grounded_call")
    return result
