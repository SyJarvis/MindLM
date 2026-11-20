"""Strict loader for the version-one tool-calling evaluation corpus."""

import json
import re
from collections import Counter
from pathlib import Path

SUITE_QUOTAS = {"single_positive": 100, "negative": 50, "loop_closure": 100, "error_recovery": 50}
CATEGORIES = frozenset({"order_lookup", "order_cancel", "address_change", "email_user_lookup",
                        "calculation", "product", "reservation"})
TOP_FIELDS = frozenset({"id", "suite", "category", "query", "tools", "fixture", "expect", "provenance"})
ID_RE = re.compile(r"^EXA-EVAL-CASE-\d{4}$")
SCHEMA_TYPES = frozenset({"integer", "object", "number", "array", "boolean", "string"})
SCHEMA_FIELDS = frozenset({"type", "description", "enum", "properties", "required", "additionalProperties", "items"})


class ToolCaseValidationError(ValueError):
    pass


def _fail(path, message):
    raise ToolCaseValidationError(f"{path}: {message}")


def _object(value, fields, path):
    if not isinstance(value, dict):
        _fail(path, "expected object")
    if set(value) != set(fields):
        _fail(path, f"expected fields {sorted(fields)}, got {sorted(value)}")
    return value


def _nonempty_string(value, path):
    if not isinstance(value, str) or not value:
        _fail(path, "expected non-empty string")
    return value


def _resolve(document, dotted_path, path):
    value = document
    try:
        for part in dotted_path.split("."):
            value = value[int(part)] if isinstance(value, list) else value[part]
    except (IndexError, KeyError, TypeError, ValueError):
        _fail(path, f"unresolvable fixture path {dotted_path!r}")
    return value


def _validate_json_schema(schema, path):
    if not isinstance(schema, dict):
        _fail(path, "expected schema object")
    unknown = set(schema) - SCHEMA_FIELDS
    if unknown:
        _fail(path, f"unknown schema fields {sorted(unknown)}")
    schema_type = schema.get("type")
    if schema_type not in SCHEMA_TYPES:
        _fail(path, f"invalid schema type {schema_type!r}")
    if "description" in schema:
        _nonempty_string(schema["description"], f"{path}.description")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        _fail(f"{path}.enum", "expected non-empty list")
    if schema_type == "object":
        if not {"type", "properties", "required", "additionalProperties"} <= set(schema):
            _fail(path, "object schema is incomplete")
        properties, required = schema["properties"], schema["required"]
        if not isinstance(properties, dict):
            _fail(f"{path}.properties", "expected object")
        if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
            _fail(f"{path}.required", "expected string list")
        if len(required) != len(set(required)) or set(required) - set(properties):
            _fail(f"{path}.required", "contains duplicate or unknown fields")
        if schema["additionalProperties"] is not False:
            _fail(f"{path}.additionalProperties", "must be false")
        for name, child in properties.items():
            _nonempty_string(name, f"{path}.properties key")
            _validate_json_schema(child, f"{path}.properties.{name}")
    elif any(key in schema for key in ("properties", "required", "additionalProperties")):
        _fail(path, "non-object schema has object-only fields")
    if schema_type == "array":
        if "items" not in schema:
            _fail(path, "array schema has no items")
        _validate_json_schema(schema["items"], f"{path}.items")
    elif "items" in schema:
        _fail(path, "non-array schema has items")


def _validate_tools(tools, path):
    if not isinstance(tools, list) or not tools:
        _fail(path, "expected non-empty list")
    names = []
    for index, entry in enumerate(tools):
        item_path = f"{path}.{index}"
        entry = _object(entry, {"type", "function"}, item_path)
        if entry["type"] != "function":
            _fail(f"{item_path}.type", "must be 'function'")
        function = _object(entry["function"], {"name", "description", "parameters"}, f"{item_path}.function")
        names.append(_nonempty_string(function["name"], f"{item_path}.function.name"))
        _nonempty_string(function["description"], f"{item_path}.function.description")
        _validate_json_schema(function["parameters"], f"{item_path}.function.parameters")
    if len(names) != len(set(names)):
        _fail(path, "tool names must be unique")
    return set(names)


def _validate_final_answer(value, fixture, path):
    answer = _object(value, {"required", "oracle"}, path)
    if not isinstance(answer["required"], bool):
        _fail(f"{path}.required", "expected boolean")
    oracle = answer["oracle"]
    if oracle is None:
        return
    if isinstance(oracle, dict) and set(oracle) == {"required_any", "forbidden"}:
        for field in ("required_any", "forbidden"):
            phrases = oracle[field]
            if not isinstance(phrases, list) or not phrases or any(not isinstance(p, str) or not p for p in phrases):
                _fail(f"{path}.oracle.{field}", "expected non-empty string list")
        return
    oracle = _object(oracle, {"required_facts"}, f"{path}.oracle")
    facts = oracle["required_facts"]
    if not isinstance(facts, list) or not facts:
        _fail(f"{path}.oracle.required_facts", "expected non-empty list")
    for index, fact in enumerate(facts):
        fact_path = f"{path}.oracle.required_facts.{index}"
        fact = _object(fact, {"fixture_path", "match", "value"}, fact_path)
        fixture_path = _nonempty_string(fact["fixture_path"], f"{fact_path}.fixture_path")
        if fact["match"] != "exact_value":
            _fail(f"{fact_path}.match", "must be 'exact_value'")
        if _resolve(fixture, fixture_path, f"{fact_path}.fixture_path") != fact["value"]:
            _fail(fact_path, "oracle value is not grounded in fixture")


def _validate_case(record, index):
    path = f"record[{index}]"
    record = _object(record, TOP_FIELDS, path)
    case_id = _nonempty_string(record["id"], f"{path}.id")
    path = case_id
    if not ID_RE.fullmatch(case_id):
        _fail(f"{path}.id", "must match EXA-EVAL-CASE-NNNN")
    suite, category = record["suite"], record["category"]
    if not isinstance(suite, str) or suite not in SUITE_QUOTAS:
        _fail(f"{path}.suite", f"unknown suite {suite!r}")
    if not isinstance(category, str) or category not in CATEGORIES:
        _fail(f"{path}.category", f"unknown category {category!r}")
    _nonempty_string(record["query"], f"{path}.query")
    tool_names = _validate_tools(record["tools"], f"{path}.tools")
    provenance = _object(record["provenance"], {"generator", "language", "suite_index", "version", "seed"}, f"{path}.provenance")
    if (provenance["generator"] != "scripts/gen_tool_cases_v1.py" or provenance["version"] != 1
            or provenance["seed"] != 1337):
        _fail(f"{path}.provenance", "unexpected generator, version, or seed")
    if provenance["language"] not in {"en", "zh"}:
        _fail(f"{path}.provenance.language", "must be 'en' or 'zh'")
    if type(provenance["suite_index"]) is not int or not 0 <= provenance["suite_index"] < SUITE_QUOTAS[suite]:
        _fail(f"{path}.provenance.suite_index", "out of range")
    fixture_fields = {"fixture_id", "tool_results", "direct_answer"} if suite == "negative" else {"fixture_id", "tool_results"}
    fixture = _object(record["fixture"], fixture_fields, f"{path}.fixture")
    fixture_id = _nonempty_string(fixture["fixture_id"], f"{path}.fixture.fixture_id")
    if not re.fullmatch(r"EXA-EVAL-FIXTURE-\d{4}", fixture_id):
        _fail(f"{path}.fixture.fixture_id", "must match EXA-EVAL-FIXTURE-NNNN")
    results = fixture["tool_results"]
    if not isinstance(results, list):
        _fail(f"{path}.fixture.tool_results", "expected list")
    if suite == "negative":
        direct = _object(fixture["direct_answer"], {"tool_use_required"}, f"{path}.fixture.direct_answer")
        if direct["tool_use_required"] is not False:
            _fail(f"{path}.fixture.direct_answer.tool_use_required", "must be false")
    expect_fields = {"mode", "final_answer", "must_not_claim", "calls"}
    if suite == "error_recovery":
        expect_fields |= {"inject_error", "recovery"}
    expect = _object(record["expect"], expect_fields, f"{path}.expect")
    calls, claims = expect["calls"], expect["must_not_claim"]
    if not isinstance(calls, list):
        _fail(f"{path}.expect.calls", "expected list")
    if not isinstance(claims, list) or any(not isinstance(c, str) or not c for c in claims) or len(claims) != len(set(claims)):
        _fail(f"{path}.expect.must_not_claim", "expected unique string list")
    if len(calls) != len(results):
        _fail(path, "expected calls and fixture tool results have different lengths")
    for call_index, (call, result) in enumerate(zip(calls, results), 1):
        call_path = f"{path}.expect.calls.{call_index - 1}"
        call = _object(call, {"arguments", "name", "step"}, call_path)
        result = _object(result, {"result", "arguments", "name", "step"}, f"{path}.fixture.tool_results.{call_index - 1}")
        if call["step"] != call_index or result["step"] != call_index:
            _fail(call_path, "steps must be consecutive and one-based")
        name = _nonempty_string(call["name"], f"{call_path}.name")
        if name not in tool_names:
            _fail(f"{call_path}.name", f"unknown tool {name!r}")
        if result["name"] != name:
            _fail(call_path, "fixture result name does not match expected call")
        arguments = _object(call["arguments"], {"exact", "required"}, f"{call_path}.arguments")
        exact, required = arguments["exact"], arguments["required"]
        if not isinstance(exact, dict):
            _fail(f"{call_path}.arguments.exact", "expected object")
        if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
            _fail(f"{call_path}.arguments.required", "expected string list")
        if len(required) != len(set(required)) or set(required) != set(exact):
            _fail(f"{call_path}.arguments", "required fields must exactly match exact argument keys")
        if result["arguments"] != exact:
            _fail(call_path, "fixture arguments do not exactly match expected arguments")
        if not isinstance(result["result"], dict):
            _fail(f"{path}.fixture.tool_results.{call_index - 1}.result", "expected object")
    _validate_final_answer(expect["final_answer"], fixture, f"{path}.expect.final_answer")
    expected_modes = {"single_positive": "single_call", "negative": "no_tool", "loop_closure": "tool_loop", "error_recovery": "error_recovery"}
    if expect["mode"] != expected_modes[suite]:
        _fail(f"{path}.expect.mode", f"must be {expected_modes[suite]!r}")
    final = expect["final_answer"]
    if suite == "negative":
        oracle = final["oracle"]
        if calls or results or final["required"] is not True or not isinstance(oracle, dict) or set(oracle) != {"required_any", "forbidden"} or claims:
            _fail(path, "negative cases require no calls and an explicit direct-answer oracle")
    elif suite == "single_positive":
        if len(calls) != 1 or final != {"required": False, "oracle": None} or claims:
            _fail(path, "single_positive cases require one call and no final oracle")
    elif suite == "loop_closure":
        if not 1 <= len(calls) <= 3 or final["required"] is not True or final["oracle"] is None or claims:
            _fail(path, "loop_closure requires 1-3 calls and a grounded final oracle")
    else:
        if len(calls) != 2 or not claims:
            _fail(path, "error_recovery requires two calls and forbidden claims")
        injection = _object(expect["inject_error"], {"tool_name", "fixture_path", "error_type", "step"}, f"{path}.expect.inject_error")
        if injection["step"] != 1 or injection["tool_name"] != calls[0]["name"] or injection["error_type"] != "transient_timeout":
            _fail(f"{path}.expect.inject_error", "must target the first call with transient_timeout")
        injected = _resolve(fixture, injection["fixture_path"], f"{path}.expect.inject_error.fixture_path")
        if not isinstance(injected, dict) or injected.get("type") != injection["error_type"]:
            _fail(f"{path}.expect.inject_error", "fixture path does not identify the declared error")
        recovery = _object(expect["recovery"], {"then", "success_step", "behavior"}, f"{path}.expect.recovery")
        if recovery != {"behavior": "retry_same_call_once", "success_step": 2, "then": "answer_from_success_result"}:
            _fail(f"{path}.expect.recovery", "invalid recovery contract")
        if calls[0]["name"] != calls[1]["name"] or calls[0]["arguments"] != calls[1]["arguments"]:
            _fail(path, "error recovery must retry the exact same call once")
        if final["required"] is not True or final["oracle"] is None:
            _fail(path, "error recovery requires a final oracle")


def _selection(value, allowed, name):
    if value is None:
        return None
    selected = {value} if isinstance(value, str) else set(value)
    unknown = selected - set(allowed)
    if unknown:
        raise ValueError(f"unknown {name}: {sorted(unknown)}")
    return selected


def load_cases(path, suites=None, categories=None, limit=0):
    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                _fail(f"line {line_number}", "blank lines are not allowed")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                _fail(f"line {line_number}", f"invalid JSON: {exc.msg}")
    if len(records) != 300:
        _fail(str(path), f"expected 300 records, got {len(records)}")
    for index, record in enumerate(records):
        _validate_case(record, index)
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        _fail(str(path), "case ids must be unique")
    quotas = Counter(record["suite"] for record in records)
    if quotas != Counter(SUITE_QUOTAS):
        _fail(str(path), f"wrong suite quotas: {dict(quotas)}")
    for suite, quota in SUITE_QUOTAS.items():
        suite_records = [record for record in records if record["suite"] == suite]
        indexes = [record["provenance"]["suite_index"] for record in suite_records]
        if set(indexes) != set(range(quota)):
            _fail(str(path), f"{suite} suite_index values are not complete and unique")
        if {record["category"] for record in suite_records} != CATEGORIES:
            _fail(str(path), f"{suite} does not cover all categories")
    languages = Counter(record["provenance"]["language"] for record in records)
    if languages != Counter({"en": 150, "zh": 150}):
        _fail(str(path), f"wrong language quotas: {dict(languages)}")
    suite_filter = _selection(suites, set(SUITE_QUOTAS), "suites")
    category_filter = _selection(categories, CATEGORIES, "categories")
    selected = [record for record in records
                if (suite_filter is None or record["suite"] in suite_filter)
                and (category_filter is None or record["category"] in category_filter)]
    return selected[:limit] if limit else selected
