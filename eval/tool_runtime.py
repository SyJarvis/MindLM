"""Deterministic fixture runtime and scoring helpers for tool cases."""

import copy
import json
import re
import unicodedata
from collections.abc import Mapping
from types import MappingProxyType


def args_exact_match(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            args_exact_match(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            args_exact_match(left, right) for left, right in zip(actual, expected))
    return actual == expected


def args_required_match(actual, expected, required=None):
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    if required is None and set(expected) == {"exact", "required"}:
        required, expected = expected["required"], expected["exact"]
    if required is None:
        required = tuple(expected)
    if not isinstance(required, (list, tuple)) or any(not isinstance(key, str) for key in required):
        return False
    return all(key in actual and key in expected and args_exact_match(actual[key], expected[key])
               for key in required)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _resolve(document, path):
    value = document
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, (list, tuple)) else value[part]
    return value


class ToolCaseRuntime:
    """Execute one case against an isolated immutable fixture trace."""

    def __init__(self, case):
        isolated = copy.deepcopy(dict(case))
        self._results = tuple(isolated["fixture"]["tool_results"])
        self._injection = isolated["expect"].get("inject_error")
        self._position = 0

    @property
    def next_step(self):
        return self._position + 1

    def execute(self, name, args, step):
        if self._position >= len(self._results):
            raise RuntimeError("no fixture tool result remains")
        expected = self._results[self._position]
        if type(step) is not int or step != expected["step"] or step != self.next_step:
            raise ValueError(f"expected step {self.next_step}, got {step!r}")
        if name != expected["name"]:
            raise ValueError(f"expected tool {expected['name']!r}, got {name!r}")
        actual_args = dict(args) if isinstance(args, Mapping) else args
        if not args_exact_match(actual_args, expected["arguments"]):
            raise ValueError("tool arguments do not exactly match fixture arguments")
        result = expected["result"]
        if self._injection and step == self._injection["step"]:
            error = result.get("error") if isinstance(result, dict) else None
            if not isinstance(error, dict) or error.get("type") != self._injection["error_type"]:
                raise RuntimeError("injected error fixture does not match expect.inject_error")
        self._position += 1
        return _freeze(copy.deepcopy(result))


ToolRuntime = ToolCaseRuntime


def normalize_fact(value):
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(character for character in text if character.isalnum() or character in "_.@-")


def _fact_in_answer(answer, value):
    normalized = normalize_fact(answer)
    if value is True:
        return any(token in normalized for token in ("true", "success", "succeeded", "available", "有货", "成功", "是"))
    if value is False:
        return any(token in normalized for token in ("false", "failed", "unavailable", "缺货", "失败", "否"))
    return normalize_fact(value) in normalized


def final_oracle_match(answer, oracle, fixture):
    if oracle is None:
        return True
    if not isinstance(answer, str):
        return False
    try:
        if set(oracle) == {"required_any", "forbidden"}:
            normalized = normalize_fact(answer)
            required = [normalize_fact(phrase) for phrase in oracle["required_any"]]
            forbidden = [normalize_fact(phrase) for phrase in oracle["forbidden"]]
            return bool(required) and all(required) and any(
                phrase in normalized for phrase in required) and not any(
                    phrase in normalized for phrase in forbidden)
        return all(fact.get("match") == "exact_value"
                   and args_exact_match(_resolve(fixture, fact["fixture_path"]), fact["value"])
                   and _fact_in_answer(answer, fact["value"])
                   for fact in oracle["required_facts"])
    except (KeyError, IndexError, TypeError, ValueError):
        return False


_CLAIM_PATTERNS = {
    "success_before_retry": (r"success(?:ful)? before retry", r"重试前(?:已经)?成功"),
    "permanent_failure": (r"permanent(?:ly)? fail", "cannot be retried", "永久失败", "无法重试"),
    "invented_tool_result": ("invented tool result", "fabricated tool result", r"虚构(?:的)?工具结果"),
}


def must_not_claim_violations(answer, must_not_claim):
    normalized = unicodedata.normalize("NFKC", answer).casefold()
    violations = []
    for claim in must_not_claim:
        patterns = _CLAIM_PATTERNS.get(claim, (re.escape(claim.replace("_", " ")),))
        if any(re.search(pattern, normalized) for pattern in patterns):
            violations.append(claim)
    return violations


def must_not_claim_match(answer, must_not_claim):
    return not must_not_claim_violations(answer, must_not_claim)


def fabricated_success(answer, observed_results):
    normalized = unicodedata.normalize("NFKC", answer).casefold()
    negated = re.search(r"\b(?:not|never|no)\s+(?:successful|succeeded|a success)\b", normalized) or any(
        phrase in normalized for phrase in ("未成功", "没有成功", "不成功"))
    claims_success = _fact_in_answer(answer, True) and not negated
    grounded_success = any(isinstance(result, Mapping) and "error" not in result and (
        result.get("success") is True or result.get("status") in {"shipped", "confirmed", "cancelled"})
        for result in observed_results)
    return claims_success and not grounded_success
