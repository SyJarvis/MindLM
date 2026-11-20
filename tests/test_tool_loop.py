"""Tests for E2 tools rendering and the closed-loop tool eval.

No model is built: the loop is driven by scripted ``generate_fn`` responses and
the data-pipeline checks use the real Qwen3 tokenizer on synthetic rows.
"""

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "data_process"))
from data_process.sft import (ToolsParseError, assistant_sup_intervals, main as sft_main,
                              parse_tools, render_row)
from prepare_sft_qwen3 import called_function_names, cap_tools
from eval.eval_sft_tool import (CASES as SINGLE_CASES, build_summary as build_single_summary,
                                parse_args as parse_single_args, parse_calls, score_response)
from eval.eval_sft_tool_loop import (CASES, build_summary as build_loop_summary, execute_call,
                                     parse_args as parse_loop_args, run_case)
from eval.tool_case_loader import load_cases

V1_CASES = PROJECT_ROOT / "eval" / "tool_cases_v1.jsonl"

TOOL_SCHEMA = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]
TOOL_MESSAGES = [
    {"role": "user", "content": "Beijing weather?"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'}}]},
    {"role": "tool", "content": '{"temp": 20}'},
    {"role": "assistant", "content": "It is 20 degrees."},
]


class ToolsRenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "qwen3_tokenizer"), trust_remote_code=True)
        cls.assistant_ids = cls.tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        cls.im_end_ids = cls.tok("<|im_end|>", add_special_tokens=False).input_ids

    def test_parse_tools_only_accepts_empty_cells_or_nonempty_lists(self):
        self.assertIsNone(parse_tools(""))
        self.assertIsNone(parse_tools("   \t"))
        self.assertIsNone(parse_tools(None))
        for raw in ("[]", "not json", "{}", "null", '"string"'):
            with self.subTest(raw=raw):
                with self.assertRaises(ToolsParseError):
                    parse_tools(raw)
        self.assertEqual(parse_tools(json.dumps(TOOL_SCHEMA)), TOOL_SCHEMA)

    def test_main_skips_overlong_and_invalid_tools_rows(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "rendered row is deliberately overlong"

            def __call__(self, text, add_special_tokens=False):
                return mock.Mock(input_ids=list(range(len(text))))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "input.csv"
            bin_path = root / "tokens.bin"
            meta_path = root / "rows.jsonl"
            stats_path = root / "stats.json"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["messages", "tools", "source"])
                writer.writeheader()
                writer.writerows([
                    {"messages": "[]", "tools": "  ", "source": "overlong"},
                ] + [
                    {"messages": "[]", "tools": raw, "source": f"invalid-tools-{i}"}
                    for i, raw in enumerate(("not json", "[]", "{}", "null", "\"string\""))
                ])
            argv = [
                "--csv", str(csv_path), "--tokenizer", "unused", "--max_tokens", "4",
                "--bin", str(bin_path), "--meta", str(meta_path), "--stats", str(stats_path),
            ]
            with mock.patch("data_process.sft.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()):
                with redirect_stdout(io.StringIO()):
                    sft_main(argv)

            stats = json.loads(stats_path.read_text())
            self.assertEqual(stats["over_long"], 1)
            self.assertEqual(stats["tools_parse_errors"], 5)
            self.assertEqual(stats["kept"], 0)
            self.assertEqual(bin_path.read_bytes(), b"")
            self.assertEqual(meta_path.read_text(), "")

    def test_render_with_tools_emits_tools_block(self):
        text, ids = render_row(self.tok, TOOL_MESSAGES, TOOL_SCHEMA)
        self.assertIn("# Tools", text)
        self.assertIn("<tools>", text)
        self.assertIn("</tools>", text)
        self.assertIn('"name": "get_weather"', text.replace("'", '"'))
        self.assertIn("<tool_call>", text)
        self.assertIn("<tool_response>", text)

    def test_render_without_tools_has_no_tools_block(self):
        text, ids = render_row(self.tok, TOOL_MESSAGES, None)
        self.assertNotIn("<tools>", text)
        # The call/response tags still appear inside the message bodies.
        self.assertIn("<tool_call>", text)

    def test_sup_intervals_with_tools_exclude_tools_block(self):
        text, ids = render_row(self.tok, TOOL_MESSAGES, TOOL_SCHEMA)
        keep = len(ids)
        sup = assistant_sup_intervals(ids, self.assistant_ids, self.im_end_ids, keep)
        self.assertEqual(len(sup), 2, "two assistant turns must be supervised")

        im_start, im_end = 151644, 151645
        tool_response = self.tok("<tool_response>", add_special_tokens=False).input_ids[0]
        decoded_spans = []
        for start, length in sup:
            span_ids = ids[start + 1:start + 1 + length]
            self.assertNotIn(im_start, span_ids, "supervised span must not contain <|im_start|>")
            self.assertNotIn(tool_response, span_ids, "supervised span must not contain <tool_response>")
            self.assertEqual(span_ids[-1], im_end, "supervised span must end with <|im_end|>")
            decoded_spans.append(self.tok.decode(span_ids))
        self.assertIn("<tool_call>", decoded_spans[0])
        self.assertIn("It is 20 degrees.", decoded_spans[1])

    def test_tools_block_sits_before_first_supervised_span(self):
        text, ids = render_row(self.tok, TOOL_MESSAGES, TOOL_SCHEMA)
        keep = len(ids)
        sup = assistant_sup_intervals(ids, self.assistant_ids, self.im_end_ids, keep)
        tools_end = text.index("</tools>")
        first_span_text = self.tok.decode(ids[sup[0][0] + 1:sup[0][0] + 1 + sup[0][1]])
        self.assertLess(tools_end, text.index(first_span_text.strip()[:20].split("\n")[0][:10] or "<tool_call>"))

    def test_bin_layout_unchanged_for_tools_free_rows(self):
        """Rows without tools must render byte-identically to the pre-E2 path."""
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        text_new, ids_new = render_row(self.tok, messages, None)
        text_old = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        self.assertEqual(text_new, text_old)


class ClosedLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "qwen3_tokenizer"), trust_remote_code=True)

    @staticmethod
    def scripted(responses):
        queue = list(responses)
        calls = {"n": 0}

        def generate_fn(prompt):
            calls["n"] += 1
            text = queue[min(calls["n"] - 1, len(queue) - 1)]
            return text, len(text.split()), True
        return generate_fn

    def case_by_id(self, case_id):
        return next(c for c in CASES if c["id"] == case_id)

    def test_positive_full_loop_passes(self):
        case = self.case_by_id("loop_ret_status")
        gen = self.scripted([
            '<tool_call>\n{"name": "get_order_details", "arguments": {"order_id": "W6138452"}}\n</tool_call>',
            "Your order W6138452 has shipped. Tracking: SF-77120394.",
        ])
        record = run_case(gen, self.tok, case, max_rounds=2, max_new_tokens=64)
        self.assertTrue(record["pass"], record)
        self.assertEqual(record["n_calls"], 1)
        self.assertEqual(record["called_names"], ["get_order_details"])
        self.assertTrue(record["final_relevant"])
        self.assertEqual(record["n_rounds"], 2)

    def test_hallucinated_answer_fails_final_relevant(self):
        """Model calls correctly but then ignores the tool result and invents data."""
        case = self.case_by_id("loop_calc")
        gen = self.scripted([
            '<tool_call>\n{"name": "calculate", "arguments": {"expression": "128*47+12"}}\n</tool_call>',
            "The answer is 42.",  # wrong: mock returns 6028
        ])
        record = run_case(gen, self.tok, case, max_rounds=2, max_new_tokens=64)
        self.assertFalse(record["pass"], record)
        self.assertTrue(record["name_correct"])
        self.assertFalse(record["final_relevant"])

    def test_repeat_call_without_final_answer_fails(self):
        case = self.case_by_id("loop_ret_status")
        gen = self.scripted([
            '<tool_call>\n{"name": "get_order_details", "arguments": {"order_id": "W6138452"}}\n</tool_call>',
        ])
        record = run_case(gen, self.tok, case, max_rounds=2, max_new_tokens=64)
        self.assertFalse(record["pass"], record)
        self.assertFalse(record["final_answer_emitted"])
        self.assertTrue(record["loop_stuck"])

    def test_negative_direct_answer_passes(self):
        case = self.case_by_id("loop_neg_math")
        gen = self.scripted(["2"])
        record = run_case(gen, self.tok, case, max_rounds=2, max_new_tokens=64)
        self.assertTrue(record["pass"], record)
        self.assertEqual(record["n_calls"], 0)

    def test_negative_calling_tool_fails(self):
        case = self.case_by_id("loop_neg_math")
        gen = self.scripted([
            '<tool_call>\n{"name": "calculate", "arguments": {"expression": "1+1"}}\n</tool_call>',
            "2",
        ])
        record = run_case(gen, self.tok, case, max_rounds=2, max_new_tokens=64)
        self.assertFalse(record["pass"], record)
        self.assertTrue(record["call_emitted"])

    def test_wrong_function_name_fails(self):
        case = self.case_by_id("loop_ret_cancel")
        gen = self.scripted([
            '<tool_call>\n{"name": "get_order_details", "arguments": {"order_id": "W2378156"}}\n</tool_call>',
            "Cancelled.",
        ])
        record = run_case(gen, self.tok, case, max_rounds=2, max_new_tokens=64)
        self.assertFalse(record["pass"], record)
        self.assertFalse(record["name_correct"])
        self.assertTrue(record["name_in_provided_tools"])  # valid tool, wrong choice

    def test_multi_round_case_collects_trace(self):
        case = self.case_by_id("loop_ret_modify_multi")
        gen = self.scripted([
            '<tool_call>\n{"name": "get_order_details", "arguments": {"order_id": "W3098742"}}\n</tool_call>',
            '<tool_call>\n{"name": "modify_pending_order_items", "arguments": '
            '{"order_id": "W3098742", "item_ids": ["SHOE-9"], "new_item_ids": ["SHOE-10"], '
            '"payment_method_id": "pm_8821"}}\n</tool_call>',
            "Done, the size is updated.",
        ])
        record = run_case(gen, self.tok, case, max_rounds=3, max_new_tokens=64)
        self.assertEqual(record["n_calls"], 2)
        self.assertEqual(record["called_names"], ["get_order_details", "modify_pending_order_items"])
        self.assertTrue(record["pass"], record)
        self.assertEqual(len(record["trace"]), 2)

    def test_mock_executors_are_deterministic(self):
        first = execute_call({"name": "calculate", "arguments": {"expression": "128*47+12"}})
        second = execute_call({"name": "calculate", "arguments": {"expression": "128*47+12"}})
        self.assertEqual(first, {"result": "6028"})
        self.assertEqual(first, second)
        unknown = execute_call({"name": "nope", "arguments": {}})
        self.assertIn("error", unknown)

    def test_cancel_then_status_reflects_runtime_state(self):
        cancel = execute_call({"name": "cancel_pending_order",
                               "arguments": {"order_id": "W2378156", "reason": "ordered by mistake"}})
        self.assertEqual(cancel, {"success": True, "order_id": "W2378156", "status": "cancelled"})
        again = execute_call({"name": "cancel_pending_order",
                              "arguments": {"order_id": "W2378156", "reason": "ordered by mistake"}})
        self.assertIn("error", again)  # already cancelled
        from eval.eval_sft_tool_loop import reset_runtime_state
        reset_runtime_state()
        retry = execute_call({"name": "cancel_pending_order",
                              "arguments": {"order_id": "W2378156", "reason": "ordered by mistake"}})
        self.assertTrue(retry.get("success"))

    def test_parse_calls_shared_with_probe(self):
        calls, errors = parse_calls('<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>')
        self.assertEqual((calls[0]["name"], calls[0]["arguments"]), ("f", {"a": 1}))
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()


class CapToolsTest(unittest.TestCase):
    """--max_tools capping invariants (prepare_sft_qwen3.cap_tools)."""

    @staticmethod
    def tool(name):
        return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}

    def record_with_calls(self, names):
        return {"role": "assistant",
                "tool_calls": [{"function": {"name": n, "arguments": "{}"}} for n in names]}

    def test_called_functions_survive_the_cap(self):
        tools = [self.tool(f"f{i}") for i in range(15)]
        record = {"messages": [self.record_with_calls(["f12", "f3"])], "tools": tools}
        capped = cap_tools(tools, called_function_names(record), 10)
        names = [t["function"]["name"] for t in capped]
        self.assertEqual(len(capped), 10)
        self.assertIn("f12", names)
        self.assertIn("f3", names)

    def test_called_first_order(self):
        tools = [self.tool(f"f{i}") for i in range(15)]
        record = {"messages": [self.record_with_calls(["f9"])], "tools": tools}
        names = [t["function"]["name"] for t in cap_tools(tools, called_function_names(record), 10)]
        self.assertEqual(names[0], "f9")

    def test_no_calls_keeps_original_order(self):
        tools = [self.tool(f"f{i}") for i in range(15)]
        self.assertEqual([t["function"]["name"] for t in cap_tools(tools, set(), 10)],
                         [f"f{i}" for i in range(10)])

    def test_under_limit_and_disabled_cap_are_noops(self):
        tools = [self.tool(f"f{i}") for i in range(5)]
        self.assertEqual(cap_tools(tools, {"f1"}, 10), tools)
        tools15 = [self.tool(f"f{i}") for i in range(15)]
        self.assertEqual(cap_tools(tools15, set(), 0), tools15)

    def test_called_set_exceeding_cap_passes_through(self):
        called_names = [f"c{i}" for i in range(12)]
        tools = [self.tool(n) for n in called_names]
        record = {"messages": [self.record_with_calls(called_names)], "tools": tools}
        self.assertEqual(len(cap_tools(tools, called_function_names(record), 10)), 12)


class EvalCaseSelectionTest(unittest.TestCase):
    def test_repeatable_and_comma_filters_and_defaults(self):
        single = parse_single_args([
            "--checkpoint", "unused", "--cases", str(V1_CASES),
            "--suite", "single_positive,negative", "--category", "product",
            "--category", "calculation,reservation", "--limit", "7",
        ])
        self.assertEqual(single.suite, ["single_positive", "negative"])
        self.assertEqual(single.category, ["product", "calculation", "reservation"])
        self.assertEqual(single.limit, 7)
        loop = parse_loop_args(["--checkpoint", "unused", "--cases", str(V1_CASES)])
        self.assertEqual(loop.suite, ["loop_closure", "error_recovery", "negative"])

    def test_suite_constraints_and_case_only_filters(self):
        with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_single_args(["--checkpoint", "unused", "--cases", str(V1_CASES),
                                   "--suite", "loop_closure"])
            with self.assertRaises(SystemExit):
                parse_loop_args(["--checkpoint", "unused", "--cases", str(V1_CASES),
                                 "--suite", "single_positive"])
            with self.assertRaises(SystemExit):
                parse_single_args(["--checkpoint", "unused", "--limit", "1"])


class V1EvaluatorTest(unittest.TestCase):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages, ensure_ascii=False)

    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases(V1_CASES)
        cls.tok = cls.Tokenizer()

    @classmethod
    def case(cls, suite, min_calls=0):
        return next(case for case in cls.cases
                    if case["suite"] == suite and len(case["expect"]["calls"]) >= min_calls)

    @staticmethod
    def tool_call(call, arguments=None):
        payload = {"name": call["name"],
                   "arguments": call["arguments"]["exact"] if arguments is None else arguments}
        return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"

    @staticmethod
    def oracle_answer(case):
        oracle = case["expect"]["final_answer"]["oracle"]
        if "required_any" in oracle:
            return oracle["required_any"][0]
        return "Result: " + "; ".join(str(fact["value"]) for fact in oracle["required_facts"])

    @staticmethod
    def scripted(responses):
        queue = iter(responses)

        def generate_fn(prompt):
            text = next(queue)
            return text, len(text), True
        return generate_fn

    def test_legacy_single_semantics_and_fields_are_retained(self):
        case = next(case for case in SINGLE_CASES if case["id"] == "generic_calculate")
        text = '<tool_call>{"name": "calculate", "arguments": {"expression": "wrong"}}</tool_call>'
        record = score_response(case, text, 10, 200)
        self.assertTrue(record["pass"], record)
        legacy_fields = {"id", "expect_no_call", "expect_name", "expect_args", "call_emitted",
                         "n_calls", "json_valid", "parse_errors", "name_correct",
                         "name_in_provided_tools", "args_complete", "arguments_as_json_string",
                         "text", "new_tokens", "hit_eos", "pass"}
        self.assertLessEqual(legacy_fields, set(record))

        single_args = parse_single_args(["--checkpoint", "unused"])
        negative = score_response(SINGLE_CASES[-1], "2", 1, 200)
        single_summary = build_single_summary(single_args, {}, [record, negative])
        single_legacy = {
            "config", "checkpoint", "step", "training_stage", "device", "temperature",
            "top_k", "seed", "max_new_tokens", "positive_cases", "negative_cases",
            "call_emitted_rate", "json_valid_rate", "name_accuracy",
            "name_in_provided_tools_rate", "args_complete_rate", "overall_pass_rate",
            "negative_no_call_rate", "arguments_as_json_string_count",
        }
        additions = {"args_value_match_rate", "call_sequence_correct_rate",
                     "fabricated_success_count", "recovery_outcomes", "by_suite", "by_category"}
        self.assertLessEqual(single_legacy | additions, set(single_summary))
        self.assertEqual(single_summary["overall_pass_rate"], "2/2")

        loop_case = next(case for case in CASES if case["id"] == "loop_neg_math")
        loop_record = run_case(self.scripted(["2"]), self.tok, loop_case, 2, 256)
        positive_case = next(case for case in CASES if case["id"] == "loop_calc")
        positive_record = run_case(self.scripted([
            '<tool_call>{"name": "calculate", "arguments": {"expression": "128*47+12"}}</tool_call>',
            "6028",
        ]), self.tok, positive_case, 2, 256)
        loop_args = parse_loop_args(["--checkpoint", "unused"])
        loop_summary = build_loop_summary(loop_args, [positive_record, loop_record])
        loop_legacy = {
            "config", "checkpoint", "device", "temperature", "seed", "max_rounds",
            "positive_cases", "negative_cases", "first_call_name_accuracy",
            "args_complete_rate", "tool_executed_ok_rate", "final_answer_emitted_rate",
            "final_relevant_rate", "loop_pass_rate", "negative_pass_rate",
            "loop_stuck_count", "overall_pass_rate",
        }
        self.assertLessEqual(loop_legacy | additions, set(loop_summary))
        self.assertEqual(loop_summary["overall_pass_rate"], "2/2")

    def test_v1_single_and_negative_strict_oracles(self):
        single = self.case("single_positive", 1)
        exact = score_response(single, self.tool_call(single["expect"]["calls"][0]), 10, 200)
        self.assertTrue(exact["pass"], exact)
        wrong_args = dict(single["expect"]["calls"][0]["arguments"]["exact"])
        wrong_args["unexpected"] = True
        wrong = score_response(single, self.tool_call(single["expect"]["calls"][0], wrong_args), 10, 200)
        self.assertFalse(wrong["pass"])
        self.assertFalse(wrong["args_value_match"])

        negatives = [case for case in self.cases if case["suite"] == "negative"]
        self.assertEqual(len(negatives), 50)
        for negative in negatives:
            answer = self.oracle_answer(negative)
            oracle = negative["expect"]["final_answer"]["oracle"]
            forbidden_answer = f"{oracle['required_any'][0]} {oracle['forbidden'][0]}"
            with self.subTest(case=negative["id"], evaluator="single"):
                self.assertTrue(score_response(negative, answer, len(answer), 200)["pass"])
                self.assertFalse(score_response(negative, "unrelated direct answer", 4, 200)["pass"])
                self.assertFalse(score_response(
                    negative, forbidden_answer, len(forbidden_answer), 200)["pass"])
                attempted = f"{answer}<tool_call>{{bad json}}</tool_call>"
                self.assertFalse(score_response(negative, attempted, len(attempted), 200)["pass"])
            with self.subTest(case=negative["id"], evaluator="loop"):
                correct = run_case(self.scripted([answer]), self.tok, negative, 2, 256)
                incorrect = run_case(self.scripted(["unrelated direct answer"]), self.tok, negative, 2, 256)
                forbidden = run_case(self.scripted([forbidden_answer]), self.tok, negative, 2, 256)
                self.assertTrue(correct["pass"], correct)
                self.assertFalse(incorrect["pass"], incorrect)
                self.assertFalse(forbidden["pass"], forbidden)
                call = '<tool_call>{"name":"calculate","arguments":{"expression":"1+1"}}</tool_call>'
                attempted = run_case(self.scripted([call, answer]), self.tok, negative, 2, 256)
                self.assertFalse(attempted["pass"])

    def test_v1_loop_and_error_recovery_pass(self):
        for suite in ("loop_closure", "error_recovery"):
            case = self.case(suite, 2 if suite == "error_recovery" else 1)
            responses = [self.tool_call(call) for call in case["expect"]["calls"]]
            responses.append(self.oracle_answer(case))
            record = run_case(self.scripted(responses), self.tok, case, 2, 256)
            self.assertTrue(record["pass"], record)
            self.assertTrue(record["args_value_match"])
            self.assertTrue(record["call_sequence_correct"])
            if suite == "error_recovery":
                self.assertEqual(record["recovery_outcome"], "recovered")
                self.assertFalse(record["fabricated_success"])

    def test_v1_rejects_incomplete_and_malformed_call_tags(self):
        negative = self.case("negative")
        answer = self.oracle_answer(negative)
        single = self.case("single_positive", 1)
        valid_call = self.tool_call(single["expect"]["calls"][0])
        for suffix in ("<tool_call>{", "<tool_call", "</tool_call>",
                       '<tool_call >{"name":"calculate","arguments":{}}</tool_call>'):
            with self.subTest(suffix=suffix):
                text = answer + suffix
                record = score_response(negative, text, len(text), 256)
                self.assertFalse(record["pass"], record)
                self.assertTrue(record["parse_errors"])
                loop = run_case(self.scripted([text]), self.tok, negative, 2, 256)
                self.assertFalse(loop["pass"], loop)
                self.assertTrue(loop["parse_errors"])
                self.assertFalse(loop["final_answer_emitted"])
                positive = score_response(single, valid_call + suffix, 10, 256)
                self.assertFalse(positive["pass"], positive)
                self.assertTrue(positive["parse_errors"])

        # The original key-only legacy probe retains its historical parser behavior.
        legacy = next(case for case in SINGLE_CASES if case["expect_no_call"])
        self.assertTrue(score_response(legacy, "2<tool_call>{", 10, 256)["pass"])

    def test_v1_loop_typical_failures_and_runtime_errors(self):
        case = self.case("loop_closure", 1)
        call = case["expect"]["calls"][0]
        wrong_args = dict(call["arguments"]["exact"])
        wrong_args[next(iter(wrong_args))] = "wrong-value"
        runtime_failure = run_case(
            self.scripted([self.tool_call(call, wrong_args), self.oracle_answer(case)]),
            self.tok, case, 2, 256)
        self.assertFalse(runtime_failure["pass"])
        self.assertTrue(runtime_failure["runtime_errors"])

        oracle_failure = run_case(
            self.scripted([self.tool_call(expected) for expected in case["expect"]["calls"]] + ["invented"]),
            self.tok, case, 2, 256)
        self.assertFalse(oracle_failure["pass"])
        self.assertFalse(oracle_failure["final_relevant"])

        recovery = self.case("error_recovery", 2)
        fabricated = run_case(
            self.scripted([self.tool_call(recovery["expect"]["calls"][0]),
                           "The operation succeeded before retry."]),
            self.tok, recovery, 2, 256)
        self.assertFalse(fabricated["pass"])
        self.assertTrue(fabricated["fabricated_success"])
        self.assertEqual(fabricated["recovery_outcome"], "incorrect_retry")
