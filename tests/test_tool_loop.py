"""Tests for E2 tools rendering and the closed-loop tool eval.

No model is built: the loop is driven by scripted ``generate_fn`` responses and
the data-pipeline checks use the real Qwen3 tokenizer on synthetic rows.
"""

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
sys.path.insert(0, str(PROJECT_ROOT / "data_process"))

from data_process.sft import assistant_sup_intervals, parse_tools, render_row
from prepare_sft_qwen3 import called_function_names, cap_tools
from eval.eval_sft_tool import parse_calls
from eval.eval_sft_tool_loop import CASES, execute_call, run_case

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

    def test_parse_tools_accepts_empty_and_rejects_garbage(self):
        self.assertIsNone(parse_tools(""))
        self.assertIsNone(parse_tools(None))
        self.assertIsNone(parse_tools("[]"))
        self.assertIsNone(parse_tools("not json"))
        self.assertEqual(parse_tools(json.dumps(TOOL_SCHEMA)), TOOL_SCHEMA)

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
