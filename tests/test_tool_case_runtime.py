"""Offline fixture isolation, dispatch, and scoring regression checks."""

import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from eval.tool_case_loader import ToolCaseValidationError, load_cases
from eval.tool_runtime import (ToolCaseRuntime, args_exact_match, fabricated_success,
                               final_oracle_match, must_not_claim_violations)

CORPUS = Path(__file__).resolve().parent.parent / "eval" / "tool_cases_v1.jsonl"


def mutable_json(value):
    if isinstance(value, Mapping):
        return {key: mutable_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [mutable_json(child) for child in value]
    return value


class ToolCaseRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases(CORPUS)

    def test_restored_corpus_matches_archived_formal_v2(self):
        self.assertEqual(hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
                         "29cdad3fa42179ec30d3cbfeb89e755318a49bb30ed6212515dccd844cce5181")
        self.assertEqual(Counter(case["suite"] for case in self.cases),
                         {"single_positive": 100, "negative": 50, "loop_closure": 100, "error_recovery": 50})

    def test_runtime_rejects_wrong_dispatch_without_consuming_fixture(self):
        case = next(case for case in self.cases if case["expect"]["calls"])
        call = case["expect"]["calls"][0]
        runtime = ToolCaseRuntime(case)
        for name, args, step in [
            ("unknown_tool", call["arguments"]["exact"], 1),
            (call["name"], {"order_id": "wrong-order"}, 1),
            (call["name"], call["arguments"]["exact"], True),
            (call["name"], call["arguments"]["exact"], 2),
        ]:
            with self.subTest(name=name, arguments=args, step=step):
                with self.assertRaises(ValueError):
                    runtime.execute(name, args, step)
                self.assertEqual(runtime.next_step, 1)
        result = runtime.execute(call["name"], call["arguments"]["exact"], 1)
        with self.assertRaises(TypeError):
            result["success"] = False

    def test_runtime_isolated_from_case_mutations_and_other_runs(self):
        case = copy.deepcopy(next(case for case in self.cases if case["suite"] == "error_recovery"))
        call = copy.deepcopy(case["expect"]["calls"][0])
        first, second = ToolCaseRuntime(case), ToolCaseRuntime(case)
        case["fixture"]["tool_results"][0]["result"].clear()
        self.assertIn("error", first.execute(call["name"], call["arguments"]["exact"], 1))
        self.assertEqual(second.next_step, 1)
        self.assertIn("error", second.execute(call["name"], call["arguments"]["exact"], 1))

    def test_arguments_keep_json_types_and_extra_fields(self):
        self.assertFalse(args_exact_match({"x": True}, {"x": 1}))
        self.assertFalse(args_exact_match({"x": 1.0}, {"x": 1}))
        self.assertFalse(args_exact_match({"x": 1, "extra": None}, {"x": 1}))
        self.assertFalse(args_exact_match({"items": ["b", "a"]}, {"items": ["a", "b"]}))

    def test_all_fixture_calls_and_final_oracles_replay(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                runtime = ToolCaseRuntime(case)
                for call, fixture in zip(case["expect"]["calls"], case["fixture"]["tool_results"]):
                    result = runtime.execute(call["name"], call["arguments"]["exact"], call["step"])
                    self.assertEqual(mutable_json(result), fixture["result"])
                with self.assertRaises(RuntimeError):
                    runtime.execute("extra_call", {}, runtime.next_step)
                oracle = case["expect"]["final_answer"]["oracle"]
                if oracle is not None:
                    answer = (oracle["required_any"][0] if "required_any" in oracle else
                              "; ".join(str(fact["value"]) for fact in oracle["required_facts"]))
                    self.assertTrue(final_oracle_match(answer, oracle, case["fixture"]))
                    self.assertFalse(final_oracle_match("invented unrelated answer", oracle, case["fixture"]))

    def test_unobserved_success_and_forbidden_recovery_claims(self):
        error = {"error": {"type": "transient_timeout", "retryable": True}}
        self.assertTrue(fabricated_success("The operation succeeded before retry.", [error]))
        self.assertFalse(fabricated_success("The operation never succeeded.", [error]))
        self.assertFalse(fabricated_success("The operation succeeded.", [error, {"success": True}]))
        self.assertEqual(must_not_claim_violations("Permanent failure; cannot be retried.",
                                                 ["permanent_failure"]), ["permanent_failure"])

    def test_loader_validates_unselected_cases_before_filtering(self):
        records = copy.deepcopy(self.cases)
        case = next(case for case in records if case["suite"] == "single_positive")
        case["fixture"]["tool_results"][0]["arguments"] = {"fabricated": "value"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            with self.assertRaisesRegex(ToolCaseValidationError, "fixture arguments"):
                load_cases(path, suites=["negative"], limit=1)


if __name__ == "__main__":
    unittest.main()
