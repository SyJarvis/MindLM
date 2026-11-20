"""Closed-loop tool-calling evaluation for a MindLM SFT checkpoint.

Extends ``eval_sft_tool.py`` from "did the first reply contain the right call?"
to the full agent loop:

    render(tools) -> generate -> parse <tool_call> -> execute (deterministic
    mock) -> append tool result -> generate again -> ... -> final answer

A positive case passes only when the *whole loop* works: the first call names
the expected function with complete arguments, the mock executes, and after the
tool result comes back the model produces a final answer (no call, EOS, and
content that actually uses the returned data). Negative cases must answer
directly without any call.

Usage:
  python3 eval/eval_sft_tool_loop.py --checkpoint <ckpt> [--config mindlm_0.2b_gdn]
      [--device cpu] [--temperature 0.1] [--seed 1337] [--max_rounds 3]
      [--json_out results.json]
"""

import argparse
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import default_tokenizer_path, inference_context, load_checkpoint_with_tokenizer_check, supported_configs
from eval_sft_tool import (GENERIC_TOOLS, RETAIL_TOOLS, _breakdown, _filter_values,
                           _nonnegative_int, _rate, parse_calls)
from modeling_mindlm import MindLM
from tool_case_loader import CATEGORIES, load_cases
from tool_runtime import (ToolCaseRuntime, args_exact_match, fabricated_success,
                          final_oracle_match, must_not_claim_violations)
from training_utils import build_model_config

# ---------------------------------------------------------------------------
# Deterministic mock tool runtime.  No wall-clock / randomness: the same
# arguments always return the same payload so eval runs are reproducible.
# ---------------------------------------------------------------------------

ORDERS = {
    "W3098742": {"order_id": "W3098742", "status": "pending",
                 "items": [{"item_id": "SHOE-9", "name": "running shoes (size 9)", "qty": 1},
                           {"item_id": "SOCK-2", "name": "sports socks", "qty": 2}],
                 "payment_method_id": "pm_8821"},
    "W6138452": {"order_id": "W6138452", "status": "shipped",
                 "items": [{"item_id": "TSHIRT-L", "name": "cotton t-shirt (L)", "qty": 1}],
                 "payment_method_id": "pm_1044", "tracking": "SF-77120394"},
    "W2378156": {"order_id": "W2378156", "status": "pending",
                 "items": [{"item_id": "LAMP-1", "name": "desk lamp", "qty": 1}],
                 "payment_method_id": "pm_3350"},
}


def _exec_get_order_details(args):
    order_id = str(args.get("order_id", ""))
    if order_id in ORDERS:
        return dict(ORDERS[order_id])
    return {"error": f"order {order_id} not found"}


def _exec_find_user_id_by_email(args):
    email = str(args.get("email", ""))
    return {"user_id": "u_778291", "email": email}


def _exec_cancel_pending_order(args):
    order_id = str(args.get("order_id", ""))
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"order {order_id} not found"}
    if order["status"] != "pending":
        return {"error": f"order {order_id} is {order['status']}, cannot cancel"}
    order["status"] = "cancelled"
    return {"success": True, "order_id": order_id, "status": "cancelled"}


def _exec_modify_pending_order_items(args):
    order_id = str(args.get("order_id", ""))
    order = ORDERS.get(order_id)
    if order is None or order["status"] != "pending":
        return {"error": "only pending orders can be modified"}
    return {"success": True, "order_id": order_id, "new_items": args.get("new_item_ids", [])}


def _exec_calculate(args):
    expression = str(args.get("expression", ""))
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        return {"error": "unsupported expression"}
    return {"result": str(int(eval(expression)))}  # noqa: S307 - regex-sanitized arithmetic


def _exec_get_reservation_details(args):
    return {"reservation_id": str(args.get("reservation_id", "")), "guest_name": "Chen Wei",
            "check_in": "2026-10-03", "nights": 2, "status": "confirmed"}


def _exec_get_product_details(args):
    return {"product_id": str(args.get("product_id", "")), "name": "running shoes",
            "available_sizes": ["8", "9", "10"], "price": 89.9}


MOCK_EXECUTORS = {
    "get_order_details": _exec_get_order_details,
    "find_user_id_by_email": _exec_find_user_id_by_email,
    "cancel_pending_order": _exec_cancel_pending_order,
    "modify_pending_order_items": _exec_modify_pending_order_items,
    "calculate": _exec_calculate,
    "get_reservation_details": _exec_get_reservation_details,
    "get_product_details": _exec_get_product_details,
}

# Deep-copy guard: ORDERS is mutated by cancels; reset between eval runs.
def reset_runtime_state():
    ORDERS["W3098742"]["status"] = "pending"
    ORDERS["W6138452"]["status"] = "shipped"
    ORDERS["W2378156"]["status"] = "pending"


# ---------------------------------------------------------------------------
# Cases.  ``expect_first``: the round-1 call.  ``expect_final_any``: tokens the
# final answer must contain at least one of (case-insensitive), which proves
# the model actually consumed the tool result instead of hallucinating.
# ---------------------------------------------------------------------------

CASES = [
    {"id": "loop_ret_status", "tools": RETAIL_TOOLS,
     "query": "Can you tell me the current status of order #W6138452?",
     "expect_name": "get_order_details", "expect_args": ["order_id"],
     "expect_final_any": ["shipped", "已发货", "发货", "tracking", "SF-77120394"],
     "expect_no_call": False},
    {"id": "loop_ret_cancel", "tools": RETAIL_TOOLS,
     "query": "Please cancel my pending order #W2378156. I ordered it by mistake.",
     "expect_name": "cancel_pending_order", "expect_args": ["order_id", "reason"],
     "expect_final_any": ["cancel", "cancelled", "取消", "success", "成功"],
     "expect_no_call": False},
    {"id": "loop_ret_user", "tools": RETAIL_TOOLS,
     "query": "I lost my order number. Could you look up my account? My email address is ben.turner9212@example.com.",
     "expect_name": "find_user_id_by_email", "expect_args": ["email"],
     "expect_final_any": ["u_778291"],
     "expect_no_call": False},
    {"id": "loop_ret_modify_multi", "tools": RETAIL_TOOLS, "max_rounds": 3,
     "query": "I need to change the size of the running shoes in my order #W3098742.",
     "expect_name": "get_order_details", "expect_args": ["order_id"],
     "expect_final_any": ["size", "success", "成功", "modified", "updated", "改"],
     "expect_no_call": False},
    {"id": "loop_calc", "tools": GENERIC_TOOLS,
     "query": "What is 128 * 47 + 12?",
     "expect_name": "calculate", "expect_args": ["expression"],
     "expect_final_any": ["6028"],
     "expect_no_call": False},
    {"id": "loop_reservation", "tools": GENERIC_TOOLS,
     "query": "Please pull up reservation 4W7RTG7X for me.",
     "expect_name": "get_reservation_details", "expect_args": ["reservation_id"],
     "expect_final_any": ["Chen Wei", "2026-10-03", "confirmed", "确认"],
     "expect_no_call": False},
    {"id": "loop_neg_chitchat", "tools": RETAIL_TOOLS,
     "query": "你好，你是谁？",
     "expect_name": None, "expect_args": [],
     "expect_final_any": ["我", "助手", "assistant", "模型", "AI"],
     "expect_no_call": True},
    {"id": "loop_neg_math", "tools": RETAIL_TOOLS,
     "query": "One plus one equals what? Please just answer with the number.",
     "expect_name": None, "expect_args": [],
     "expect_final_any": ["2", "二", "两"],
     "expect_no_call": True},
]

LOOP_CASE_SUITES = frozenset({"loop_closure", "error_recovery", "negative"})
DEFAULT_LOOP_CASE_SUITES = ("loop_closure", "error_recovery", "negative")


def render_messages(tokenizer, messages, tools):
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not isinstance(prompt, str):
        prompt = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in prompt)
    return prompt


def execute_call(call):
    fn = MOCK_EXECUTORS.get(call["name"])
    if fn is None:
        return {"error": f"unknown tool: {call['name']}"}
    try:
        return fn(call["arguments"])
    except Exception as exc:  # mock must never crash the eval loop
        return {"error": f"executor failed: {exc}"}


def _run_legacy_case(generate_fn, tokenizer, case, max_rounds, max_new_tokens):
    """Drive one closed-loop case.  ``generate_fn(prompt) -> (text, new_tokens, hit_eos)``.

    Pure w.r.t. the model: tests inject a scripted generate_fn, main() injects
    the real one.  Returns a result record with per-round traces.
    """
    messages = [{"role": "user", "content": case["query"]}]
    tool_names = {tool["function"]["name"] for tool in case["tools"]}
    rounds = []
    first = None
    trace = []

    rounds_allowed = case.get("max_rounds", max_rounds)
    final_text, final_hit_eos, final_new_tokens = None, False, 0
    exhausted = False
    parse_errors = []
    for round_index in range(rounds_allowed):
        prompt = render_messages(tokenizer, messages, case["tools"])
        text, new_tokens, hit_eos = generate_fn(prompt)
        rounds.append({"round": round_index + 1, "new_tokens": new_tokens, "hit_eos": hit_eos})
        calls, errors = parse_calls(text)
        parse_errors.extend(errors)
        if first is None:
            first = calls[0] if calls else None
        if not calls:
            final_text, final_hit_eos, final_new_tokens = text, hit_eos, new_tokens
            break
        # Emit a call: execute every block, append assistant + tool messages.
        for call in calls:
            result = execute_call(call)
            trace.append({"name": call["name"], "arguments": call["arguments"], "result": result})
            messages.append({"role": "assistant", "content": "",
                             "tool_calls": [{"type": "function",
                                             "function": {"name": call["name"],
                                                          "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}]})
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
    else:
        # Loop exhausted while still calling tools.
        exhausted = True
        final_text = text

    emitted = first is not None
    executed_ok = bool(trace) and all("error" not in entry["result"] for entry in trace)
    called_names = [entry["name"] for entry in trace]
    repeated_same_call = (len(trace) >= 2
                          and all(entry["name"] == trace[0]["name"] and entry["arguments"] == trace[0]["arguments"]
                                  for entry in trace))
    loop_stuck = exhausted or repeated_same_call
    final_relevant = bool(final_text) and any(
        token.lower() in final_text.lower() for token in case.get("expect_final_any", []))
    name_correct = (first["name"] == case["expect_name"]) if first else False
    args_complete = bool(first) and all(key in first["arguments"] for key in case["expect_args"])
    name_in_provided = bool(first) and first["name"] in tool_names

    record = {
        "id": case["id"],
        "expect_no_call": case["expect_no_call"],
        "n_rounds": len(rounds),
        "n_calls": len(trace),
        "called_names": called_names,
        "call_emitted": emitted,
        "json_valid": emitted and not parse_errors,
        "parse_errors": parse_errors,
        "name_correct": name_correct,
        "name_in_provided_tools": name_in_provided,
        "args_complete": args_complete,
        "tool_executed_ok": executed_ok,
        "final_answer_emitted": final_text is not None and not parse_calls(final_text)[0],
        "final_relevant": final_relevant,
        "final_hit_eos": final_hit_eos,
        "final_text": (final_text or "")[:400],
        "loop_stuck": loop_stuck,
        "trace": trace,
    }
    if case["expect_no_call"]:
        record["pass"] = (not emitted) and final_relevant and record["final_hit_eos"]
    else:
        record["pass"] = (emitted and name_correct and name_in_provided and args_complete
                          and executed_ok and record["final_answer_emitted"]
                          and final_relevant and not loop_stuck)
    return record


def _mutable_json(value):
    if isinstance(value, Mapping):
        return {key: _mutable_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_mutable_json(child) for child in value]
    return value


def _run_corpus_case(generate_fn, tokenizer, case, max_rounds, max_new_tokens):
    expected_calls = case["expect"]["calls"]
    expect_no_call = case["suite"] == "negative"
    rounds_allowed = max(max_rounds, len(expected_calls) + 1)
    runtime = ToolCaseRuntime(case)
    messages = [{"role": "user", "content": case["query"]}]
    rounds, trace, parse_errors, runtime_errors = [], [], [], []
    final_text, final_hit_eos = None, False
    exhausted = False

    for round_index in range(rounds_allowed):
        prompt = render_messages(tokenizer, messages, case["tools"])
        text, new_tokens, hit_eos = generate_fn(prompt)
        rounds.append({"round": round_index + 1, "new_tokens": new_tokens, "hit_eos": hit_eos})
        calls, errors = parse_calls(text, strict=True)
        parse_errors.extend(errors)
        if not calls:
            final_text, final_hit_eos = text, hit_eos
            break
        for call in calls:
            step = len(trace) + 1
            try:
                result = _mutable_json(runtime.execute(call["name"], call["arguments"], step))
                runtime_error = None
            except (RuntimeError, TypeError, ValueError) as exc:
                result = {"error": {"type": "runtime_error", "message": str(exc)}}
                runtime_error = str(exc)
                runtime_errors.append(runtime_error)
            trace.append({"name": call["name"], "arguments": call["arguments"],
                          "result": result, "runtime_error": runtime_error})
            messages.append({"role": "assistant", "content": "",
                             "tool_calls": [{"type": "function", "function": {
                                 "name": call["name"],
                                 "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}]})
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
    else:
        exhausted = True
        final_text = text

    actual_calls = [{"name": item["name"], "arguments": item["arguments"]} for item in trace]
    call_sequence_correct = (len(actual_calls) == len(expected_calls)
                             and all(actual["name"] == expected["name"]
                                     for actual, expected in zip(actual_calls, expected_calls)))
    args_value_match = (len(actual_calls) == len(expected_calls)
                        and all(args_exact_match(actual["arguments"], expected["arguments"]["exact"])
                                for actual, expected in zip(actual_calls, expected_calls)))
    expected_first = expected_calls[0] if expected_calls else None
    first = actual_calls[0] if actual_calls else None
    expected_names = {tool["function"]["name"] for tool in case["tools"]}
    observed_results = [item["result"] for item in trace]
    final_calls, final_errors = parse_calls(final_text or "", strict=True)
    final_answer_emitted = final_text is not None and not final_calls and not final_errors
    final_relevant = final_oracle_match(
        final_text or "", case["expect"]["final_answer"]["oracle"], case["fixture"])
    violations = must_not_claim_violations(final_text or "", case["expect"]["must_not_claim"])
    fabricated = fabricated_success(final_text or "", observed_results)
    executed_ok = (bool(trace) and not runtime_errors
                   and all("error" not in item["result"] for item in trace))
    loop_stuck = exhausted
    if case["suite"] == "error_recovery":
        retry_exact = (len(actual_calls) == 2 and actual_calls[0]["name"] == actual_calls[1]["name"]
                       and args_exact_match(actual_calls[0]["arguments"], actual_calls[1]["arguments"]))
        first_failed = len(observed_results) >= 1 and "error" in observed_results[0]
        retry_succeeded = len(observed_results) == 2 and "error" not in observed_results[1]
        if runtime_errors:
            recovery_outcome = "runtime_error"
        elif not first_failed:
            recovery_outcome = "missing_injected_error"
        elif not retry_exact:
            recovery_outcome = "incorrect_retry"
        elif not retry_succeeded:
            recovery_outcome = "retry_failed"
        elif fabricated or violations:
            recovery_outcome = "fabricated_or_forbidden_claim"
        elif not final_relevant:
            recovery_outcome = "final_oracle_failed"
        else:
            recovery_outcome = "recovered"
    else:
        retry_exact = False
        recovery_outcome = None

    record = {
        "id": case["id"], "suite": case["suite"], "category": case["category"],
        "expect_no_call": expect_no_call, "n_rounds": len(rounds), "n_calls": len(trace),
        "called_names": [item["name"] for item in trace], "call_emitted": bool(trace),
        "json_valid": bool(trace) and not parse_errors, "parse_errors": parse_errors,
        "name_correct": bool(first and expected_first and first["name"] == expected_first["name"]),
        "name_in_provided_tools": bool(first) and first["name"] in expected_names,
        "args_complete": bool(first and expected_first) and all(
            key in first["arguments"] for key in expected_first["arguments"]["required"]),
        "args_value_match": args_value_match,
        "call_sequence_correct": call_sequence_correct,
        "tool_executed_ok": executed_ok,
        "final_answer_emitted": final_answer_emitted,
        "final_relevant": final_relevant, "final_hit_eos": final_hit_eos,
        "final_text": (final_text or "")[:400], "loop_stuck": loop_stuck,
        "trace": trace, "runtime_errors": runtime_errors,
        "fabricated_success": fabricated, "must_not_claim_violations": violations,
        "retry_exact": retry_exact, "recovery_outcome": recovery_outcome,
    }
    if expect_no_call:
        record["pass"] = (not trace and not parse_errors and final_answer_emitted
                          and final_relevant)
    elif case["suite"] == "loop_closure":
        record["pass"] = (not parse_errors and call_sequence_correct and args_value_match and executed_ok
                          and final_answer_emitted and final_relevant and not loop_stuck)
    else:
        record["pass"] = (not parse_errors and call_sequence_correct and args_value_match and retry_exact
                          and recovery_outcome == "recovered" and final_answer_emitted
                          and not loop_stuck)
    return record


def run_case(generate_fn, tokenizer, case, max_rounds, max_new_tokens):
    if "suite" in case:
        return _run_corpus_case(generate_fn, tokenizer, case, max_rounds, max_new_tokens)
    record = _run_legacy_case(generate_fn, tokenizer, case, max_rounds, max_new_tokens)
    record.update(
        args_value_match=record["args_complete"],
        call_sequence_correct=(not record["call_emitted"] if record["expect_no_call"]
                               else record["name_correct"]),
        fabricated_success=False,
        recovery_outcome=None,
        runtime_errors=[],
    )
    return record


def build_parser():
    parser = argparse.ArgumentParser(description="MindLM closed-loop tool-calling eval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", choices=supported_configs(), default="mindlm_0.2b_gdn")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json_out", default=None)
    parser.add_argument("--cases", default=None, help="validated v1 JSONL case corpus")
    parser.add_argument("--suite", action="append", help="suite filter; repeat or comma-separate")
    parser.add_argument("--category", action="append", help="category filter; repeat or comma-separate")
    parser.add_argument("--limit", type=_nonnegative_int, default=0)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.suite, args.category = _filter_values(args.suite), _filter_values(args.category)
    if not args.cases:
        if args.suite is not None or args.category is not None or args.limit:
            parser.error("--suite, --category, and --limit require --cases")
        return args
    if args.suite is None:
        args.suite = list(DEFAULT_LOOP_CASE_SUITES)
    unknown_suites = set(args.suite) - LOOP_CASE_SUITES
    if unknown_suites:
        parser.error(f"unsupported suite for this evaluator: {sorted(unknown_suites)}")
    if args.category is not None and set(args.category) - set(CATEGORIES):
        parser.error(f"unknown category: {sorted(set(args.category) - set(CATEGORIES))}")
    return args


def selected_cases(args):
    if not args.cases:
        return CASES
    return load_cases(args.cases, suites=args.suite, categories=args.category, limit=args.limit)


def build_summary(args, results):
    positives = [r for r in results if not r["expect_no_call"]]
    negatives = [r for r in results if r["expect_no_call"]]
    return {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_rounds": args.max_rounds,
        "positive_cases": len(positives),
        "negative_cases": len(negatives),
        "first_call_name_accuracy": _rate(positives, "name_correct"),
        "args_complete_rate": _rate(positives, "args_complete"),
        "tool_executed_ok_rate": _rate(positives, "tool_executed_ok"),
        "final_answer_emitted_rate": _rate(positives, "final_answer_emitted"),
        "final_relevant_rate": _rate(positives, "final_relevant"),
        "loop_pass_rate": _rate(positives, "pass"),
        "negative_pass_rate": _rate(negatives, "pass"),
        "loop_stuck_count": sum(r["loop_stuck"] for r in results),
        "overall_pass_rate": _rate(results, "pass"),
        "args_value_match_rate": _rate(positives, "args_value_match"),
        "call_sequence_correct_rate": _rate(results, "call_sequence_correct"),
        "fabricated_success_count": sum(r["fabricated_success"] for r in results),
        "recovery_outcomes": {
            outcome: sum(r["recovery_outcome"] == outcome for r in results)
            for outcome in sorted({r["recovery_outcome"] for r in results
                                   if r["recovery_outcome"] is not None})
        },
        "by_suite": _breakdown(results, "suite"),
        "by_category": _breakdown(results, "category"),
    }


def main(argv=None):
    args = parse_args(argv)

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(default_tokenizer_path(args.tokenizer_path), trust_remote_code=True)
    model = MindLM(build_model_config(args.config, tokenizer)).to(args.device)
    metadata = load_checkpoint_with_tokenizer_check(model, args.checkpoint, args.device, tokenizer)
    model.eval()
    print(f"config={args.config} checkpoint={args.checkpoint} device={args.device} "
          f"stage={metadata.get('training_stage') if isinstance(metadata, dict) else None} "
          f"temperature={args.temperature} seed={args.seed} max_rounds={args.max_rounds}\n")

    def generate_fn(prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        with inference_context(args.device):
            generated = model.generate(
                input_ids=input_ids,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        response_ids = generated[0, input_ids.size(1):]
        text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return text, int(response_ids.numel()), int(response_ids.numel()) < args.max_new_tokens

    reset_runtime_state()
    results = [run_case(generate_fn, tokenizer, case, args.max_rounds, args.max_new_tokens)
               for case in selected_cases(args)]
    reset_runtime_state()

    for record in results:
        verdict = "PASS" if record["pass"] else "FAIL"
        print(f"[{verdict}] {record['id']}  rounds={record['n_rounds']} calls={record['n_calls']} "
              f"names={record['called_names']} final_relevant={record['final_relevant']}")
        print(f"    终答: {record['final_text'][:200]}\n")

    summary = build_summary(args, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
