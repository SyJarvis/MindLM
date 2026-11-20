"""Tool-calling probe for MindLM SFT checkpoint.

Uses functions that actually appear in the training data (top ~3,500 functions)
so the model is evaluated on what it was actually taught.

Metrics:
  * call_emitted  - the reply contains a <tool_call> block
  * json_valid    - every emitted block parses as JSON with name + arguments
  * name_correct  - the first block names the expected function
  * args_complete - the expected argument keys are present
  * negative cases must NOT emit a call
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import default_tokenizer_path, inference_context, load_checkpoint_with_tokenizer_check, supported_configs
from modeling_mindlm import MindLM
from training_utils import build_model_config
from tool_case_loader import CATEGORIES, load_cases
from tool_runtime import (ToolCaseRuntime, args_exact_match, args_required_match,
                          final_oracle_match, fabricated_success)

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# ---------------------------------------------------------------------------
# Training-data functions (top ~20 from the actual training corpus)
# ---------------------------------------------------------------------------
TRAINING_TOOLS = [
    {"type": "function", "function": {
        "name": "get_current_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}},
            "required": ["city"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "calculate_math",
        "description": "Evaluate a mathematical expression.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate"}},
            "required": ["expression"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "unit_converter",
        "description": "Convert between units of measurement.",
        "parameters": {"type": "object", "properties": {
            "value": {"type": "number", "description": "The value to convert"},
            "from_unit": {"type": "string", "description": "Source unit"},
            "to_unit": {"type": "string", "description": "Target unit"}},
            "required": ["value", "from_unit", "to_unit"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "get_exchange_rate",
        "description": "Get the current exchange rate between two currencies.",
        "parameters": {"type": "object", "properties": {
            "from_currency": {"type": "string", "description": "Source currency code"},
            "to_currency": {"type": "string", "description": "Target currency code"}},
            "required": ["from_currency", "to_currency"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "translate_text",
        "description": "Translate text from one language to another.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "Text to translate"},
            "source_lang": {"type": "string", "description": "Source language code"},
            "target_lang": {"type": "string", "description": "Target language code"}},
            "required": ["text", "target_lang"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for information.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query"}},
            "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Get the current time for a timezone.",
        "parameters": {"type": "object", "properties": {
            "timezone": {"type": "string", "description": "Timezone"}},
            "required": ["timezone"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "random_number",
        "description": "Generate a random number within a range.",
        "parameters": {"type": "object", "properties": {
            "min": {"type": "integer", "description": "Minimum value"},
            "max": {"type": "integer", "description": "Maximum value"}},
            "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "text_length",
        "description": "Calculate the length of a text string.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "The text to measure"}},
            "required": ["text"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "get_news",
        "description": "Get the latest news headlines.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "description": "News category"}},
            "required": ["category"], "additionalProperties": False}}},
]

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
POSITIVE_CASES = [
    {"id": "weather_beijing", "tools": TRAINING_TOOLS,
     "query": "What is the weather like in Beijing today?",
     "expect_name": "get_current_weather", "expect_args": ["city"], "expect_no_call": False},
    {"id": "weather_nyc", "tools": TRAINING_TOOLS,
     "query": "What is the temperature in New York in Fahrenheit?",
     "expect_name": "get_current_weather", "expect_args": ["city"], "expect_no_call": False},
    {"id": "weather_paris", "tools": TRAINING_TOOLS,
     "query": "巴黎今天天气怎么样？",
     "expect_name": "get_current_weather", "expect_args": ["city"], "expect_no_call": False},
    {"id": "math_simple", "tools": TRAINING_TOOLS,
     "query": "What is 128 * 47 + 12?",
     "expect_name": "calculate_math", "expect_args": ["expression"], "expect_no_call": False},
    {"id": "math_complex", "tools": TRAINING_TOOLS,
     "query": "Can you calculate (15 + 7) * 3 / 2?",
     "expect_name": "calculate_math", "expect_args": ["expression"], "expect_no_call": False},
    {"id": "math_chinese", "tools": TRAINING_TOOLS,
     "query": "请计算 256 除以 8 的结果",
     "expect_name": "calculate_math", "expect_args": ["expression"], "expect_no_call": False},
    {"id": "convert_meters_to_feet", "tools": TRAINING_TOOLS,
     "query": "Convert 100 meters to feet.",
     "expect_name": "unit_converter", "expect_args": ["value", "from_unit", "to_unit"], "expect_no_call": False},
    {"id": "convert_miles_to_km", "tools": TRAINING_TOOLS,
     "query": "How many kilometers is 50 miles?",
     "expect_name": "unit_converter", "expect_args": ["value", "from_unit", "to_unit"], "expect_no_call": False},
    {"id": "convert_celsius_fahrenheit", "tools": TRAINING_TOOLS,
     "query": "Convert 30 degrees Celsius to Fahrenheit.",
     "expect_name": "unit_converter", "expect_args": ["value", "from_unit", "to_unit"], "expect_no_call": False},
    {"id": "convert_kg_pounds", "tools": TRAINING_TOOLS,
     "query": "把 75 公斤换算成磅",
     "expect_name": "unit_converter", "expect_args": ["value", "from_unit", "to_unit"], "expect_no_call": False},
    {"id": "exchange_usd_eur", "tools": TRAINING_TOOLS,
     "query": "What is the current exchange rate from USD to EUR?",
     "expect_name": "get_exchange_rate", "expect_args": ["from_currency", "to_currency"], "expect_no_call": False},
    {"id": "exchange_cny_usd", "tools": TRAINING_TOOLS,
     "query": "人民币兑美元的汇率是多少？",
     "expect_name": "get_exchange_rate", "expect_args": ["from_currency", "to_currency"], "expect_no_call": False},
    {"id": "translate_en_fr", "tools": TRAINING_TOOLS,
     "query": "Translate 'Hello, how are you?' to French.",
     "expect_name": "translate_text", "expect_args": ["text", "target_lang"], "expect_no_call": False},
    {"id": "translate_en_zh", "tools": TRAINING_TOOLS,
     "query": "把 'Artificial intelligence' 翻译成中文",
     "expect_name": "translate_text", "expect_args": ["text", "target_lang"], "expect_no_call": False},
    {"id": "search_news", "tools": TRAINING_TOOLS,
     "query": "Search for the latest news about climate change.",
     "expect_name": "web_search", "expect_args": ["query"], "expect_no_call": False},
    {"id": "time_shanghai", "tools": TRAINING_TOOLS,
     "query": "What time is it in Shanghai?",
     "expect_name": "get_current_time", "expect_args": ["timezone"], "expect_no_call": False},
    {"id": "random_dice", "tools": TRAINING_TOOLS,
     "query": "Generate a random number between 1 and 6.",
     "expect_name": "random_number", "expect_args": [], "expect_no_call": False},
    {"id": "text_len", "tools": TRAINING_TOOLS,
     "query": "What is the length of the string 'Hello World'?",
     "expect_name": "text_length", "expect_args": ["text"], "expect_no_call": False},
    {"id": "news_tech", "tools": TRAINING_TOOLS,
     "query": "What are the latest technology news?",
     "expect_name": "get_news", "expect_args": ["category"], "expect_no_call": False},
    {"id": "weather_then_convert", "tools": TRAINING_TOOLS,
     "query": "What is the weather in Tokyo? Also convert 25 Celsius to Fahrenheit.",
     "expect_name": "get_current_weather", "expect_args": ["city"], "expect_no_call": False},
]

NEGATIVE_CASES = [
    {"id": "neg_greeting", "tools": TRAINING_TOOLS,
     "query": "你好，你是谁？", "expect_name": None, "expect_args": [], "expect_no_call": True},
    {"id": "neg_how_are_you", "tools": TRAINING_TOOLS,
     "query": "How are you doing today?", "expect_name": None, "expect_args": [], "expect_no_call": True},
    {"id": "neg_name", "tools": TRAINING_TOOLS,
     "query": "What is your name?", "expect_name": None, "expect_args": [], "expect_no_call": True},
    {"id": "neg_philosophy", "tools": TRAINING_TOOLS,
     "query": "What is the meaning of life?", "expect_name": None, "expect_args": [], "expect_no_call": True},
    {"id": "neg_capital", "tools": TRAINING_TOOLS,
     "query": "What is the capital of France?", "expect_name": None, "expect_args": [], "expect_no_call": True},
    {"id": "neg_hello_zh", "tools": TRAINING_TOOLS,
     "query": "你好，今天天气真好", "expect_name": None, "expect_args": [], "expect_no_call": True},
    {"id": "neg_poem", "tools": TRAINING_TOOLS,
     "query": "Can you write a poem about spring?", "expect_name": None, "expect_args": [], "expect_no_call": True},
    {"id": "neg_math_simple", "tools": TRAINING_TOOLS,
     "query": "One plus one equals what? Please just answer with the number.",
     "expect_name": None, "expect_args": [], "expect_no_call": True},
]

TRAINING_CASES = POSITIVE_CASES + NEGATIVE_CASES


RETAIL_TOOLS = [
    {"type": "function", "function": {
        "name": "find_user_id_by_email",
        "description": "Find a user id by their email address.",
        "parameters": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}}},
    {"type": "function", "function": {
        "name": "get_order_details",
        "description": "Get the details of a retail order.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "get_product_details",
        "description": "Get the details of a product.",
        "parameters": {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]}}},
    {"type": "function", "function": {
        "name": "cancel_pending_order",
        "description": "Cancel a pending order.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"}, "reason": {"type": "string", "enum": ["no longer needed", "ordered by mistake"]}},
            "required": ["order_id", "reason"]}}},
    {"type": "function", "function": {
        "name": "modify_pending_order_items",
        "description": "Change the items of a pending order.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"}, "item_ids": {"type": "array", "items": {"type": "string"}},
            "new_item_ids": {"type": "array", "items": {"type": "string"}},
            "payment_method_id": {"type": "string"}},
            "required": ["order_id", "item_ids", "new_item_ids", "payment_method_id"]}}},
]

GENERIC_TOOLS = [
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_reservation_details",
        "description": "Get the details of a reservation.",
        "parameters": {"type": "object", "properties": {"reservation_id": {"type": "string"}}, "required": ["reservation_id"]}}},
]

LEGACY_CASES = [
    {"id": "retail_order_lookup", "tools": RETAIL_TOOLS,
     "query": "Hi, I need to change the size of the running shoes in my order #W3098742. My email is maria.chan@outlook.com.",
     "expect_name": "get_order_details", "expect_args": ["order_id"], "expect_no_call": False},
    {"id": "retail_user_lookup", "tools": RETAIL_TOOLS,
     "query": "I lost my order number. Could you look up my account? My email address is ben.turner9212@example.com.",
     "expect_name": "find_user_id_by_email", "expect_args": ["email"], "expect_no_call": False},
    {"id": "retail_cancel", "tools": RETAIL_TOOLS,
     "query": "Please cancel my pending order #W2378156. I ordered it by mistake.",
     "expect_name": "cancel_pending_order", "expect_args": ["order_id"], "expect_no_call": False},
    {"id": "retail_return_status", "tools": RETAIL_TOOLS,
     "query": "Can you tell me the current status of order #W6138452?",
     "expect_name": "get_order_details", "expect_args": ["order_id"], "expect_no_call": False},
    {"id": "generic_calculate", "tools": GENERIC_TOOLS,
     "query": "What is 128 * 47 + 12?",
     "expect_name": "calculate", "expect_args": ["expression"], "expect_no_call": False},
    {"id": "generic_reservation", "tools": GENERIC_TOOLS,
     "query": "Please pull up reservation 4W7RTG7X for me.",
     "expect_name": "get_reservation_details", "expect_args": ["reservation_id"], "expect_no_call": False},
    {"id": "negative_chitchat", "tools": RETAIL_TOOLS,
     "query": "你好，你是谁？", "expect_name": None, "expect_args": [], "expect_no_call": True},
    {"id": "negative_math_plain", "tools": RETAIL_TOOLS,
     "query": "One plus one equals what? Please just answer with the number.",
     "expect_name": None, "expect_args": [], "expect_no_call": True},
]


# Preserve the original public probe while the CLI keeps its training-data defaults.
CASES = LEGACY_CASES


def render_prompt(tokenizer, tools, query):
    messages = [{"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not isinstance(prompt, str):
        prompt = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in prompt)
    return prompt


def parse_calls(text, strict=False):
    calls, errors = [], []
    blocks = TOOL_CALL_RE.findall(text)
    if strict and (len(re.findall(r"</?tool_call\b", text)) != 2 * len(blocks)
                   or text.count("<tool_call>") != len(blocks)
                   or text.count("</tool_call>") != len(blocks)):
        errors.append("tool_call tags are incomplete or malformed")
    for block in blocks:
        try:
            payload = json.loads(block.strip())
        except json.JSONDecodeError as exc:
            errors.append(f"json: {exc}")
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            errors.append("payload must be an object with a string 'name'")
            continue
        arguments = payload.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                errors.append(f"arguments string is not JSON: {exc}")
                continue
        if not isinstance(arguments, dict):
            errors.append("arguments must be an object (or a JSON string of one)")
            continue
        calls.append({"name": payload["name"], "arguments": arguments,
                      "arguments_was_string": isinstance(payload.get("arguments"), str)})
    return calls, errors


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _filter_values(values):
    if values is None:
        return None
    return [value.strip() for item in values for value in item.split(",") if value.strip()]


def _breakdown(records, field):
    breakdown = {}
    for value in sorted({record.get(field, "legacy") for record in records}):
        group = [record for record in records if record.get(field, "legacy") == value]
        breakdown[value] = {"cases": len(group), "passed": sum(record["pass"] for record in group),
                            "pass_rate": _rate(group, "pass")}
    return breakdown


SINGLE_CASE_SUITES = frozenset({"single_positive", "negative"})


def build_parser():
    parser = argparse.ArgumentParser(description=(
        f"MindLM SFT tool-calling probe (default: {len(TRAINING_CASES)} training-data cases; "
        "use --cases for the validated v1 corpus)"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", choices=supported_configs(), default="mindlm_0.2b_gdn")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
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
        args.suite = ["single_positive", "negative"]
    if set(args.suite) - SINGLE_CASE_SUITES:
        parser.error(f"unsupported suite for this evaluator: {sorted(set(args.suite) - SINGLE_CASE_SUITES)}")
    if args.category is not None and set(args.category) - set(CATEGORIES):
        parser.error(f"unknown category: {sorted(set(args.category) - set(CATEGORIES))}")
    return args


def selected_cases(args):
    if args.cases:
        return load_cases(args.cases, suites=args.suite, categories=args.category, limit=args.limit)
    return TRAINING_CASES


def score_response(case, text, new_tokens, max_new_tokens):
    corpus_case = "suite" in case
    calls, errors = parse_calls(text, strict=corpus_case)
    first = calls[0] if calls else None
    emitted = bool(calls)
    attempted = bool(re.search(r"</?tool_call\b", text)) if corpus_case else bool(TOOL_CALL_RE.search(text))
    tool_names = {tool["function"]["name"] for tool in case["tools"]}
    if corpus_case:
        expected_calls = case["expect"]["calls"]
        expected_first = expected_calls[0] if expected_calls else None
        expect_no_call = case["suite"] == "negative"
        expect_name = expected_first["name"] if expected_first else None
        expect_args = expected_first["arguments"]["required"] if expected_first else []
        expected_arguments = expected_first["arguments"] if expected_first else None
    else:
        expected_calls = []
        expect_no_call, expect_name, expect_args = case["expect_no_call"], case["expect_name"], case["expect_args"]
        expected_arguments = None

    args_complete = bool(first) and all(key in first["arguments"] for key in expect_args)
    required_match = (bool(first) and args_required_match(first["arguments"], expected_arguments)
                      if expected_arguments is not None else args_complete)
    exact_match = (bool(first) and args_exact_match(first["arguments"], expected_arguments["exact"])
                   if expected_arguments is not None else args_complete)
    name_correct = first["name"] == expect_name if first else False
    expected_names = [call["name"] for call in expected_calls]
    sequence_correct = ([call["name"] for call in calls] == expected_names if corpus_case else
                        (not emitted if expect_no_call else len(calls) == 1 and name_correct))
    runtime_errors = []
    if corpus_case and calls:
        runtime = ToolCaseRuntime(case)
        for step, call in enumerate(calls, 1):
            try:
                runtime.execute(call["name"], call["arguments"], step)
            except (RuntimeError, TypeError, ValueError) as exc:
                runtime_errors.append(str(exc))
    final_relevant = (final_oracle_match(text, case["expect"]["final_answer"]["oracle"], case["fixture"])
                      if corpus_case else None)
    record = {
        "id": case["id"], "expect_no_call": expect_no_call,
        "expect_name": expect_name, "expect_args": expect_args,
        "call_emitted": emitted, "n_calls": len(calls),
        "json_valid": emitted and not errors, "parse_errors": errors,
        "name_correct": name_correct,
        "name_in_provided_tools": bool(first) and first["name"] in tool_names,
        "args_complete": args_complete, "args_required_match": required_match,
        "args_value_match": exact_match, "call_sequence_correct": sequence_correct,
        "arguments_as_json_string": bool(first and first["arguments_was_string"]),
        "fabricated_success": fabricated_success(text, []) if not calls else False,
        "recovery_outcome": None, "runtime_errors": runtime_errors,
        "final_relevant": final_relevant,
        "text": text, "new_tokens": int(new_tokens),
        "hit_eos": int(new_tokens) < max_new_tokens,
    }
    if corpus_case:
        record.update(suite=case["suite"], category=case["category"])
        if expect_no_call:
            record["pass"] = not attempted and not errors and final_relevant
        else:
            record["pass"] = (emitted and record["json_valid"] and name_correct
                              and record["name_in_provided_tools"] and required_match
                              and exact_match and sequence_correct and not runtime_errors)
    elif expect_no_call:
        record["pass"] = not emitted
    else:
        record["pass"] = (emitted and record["json_valid"] and name_correct
                          and args_complete and record["name_in_provided_tools"])
    return record


def _rate(records, field):
    return f"{sum(bool(record[field]) for record in records)}/{len(records)}"


def build_summary(args, metadata, results):
    positives = [r for r in results if not r["expect_no_call"]]
    negatives = [r for r in results if r["expect_no_call"]]
    result_ids = {record["id"] for record in results}
    case_mode = ("v1_corpus" if args.cases else
                 "legacy_probe" if result_ids <= {case["id"] for case in LEGACY_CASES} else
                 "training_data" if result_ids <= {case["id"] for case in TRAINING_CASES} else "mixed")
    return {
        "case_mode": case_mode, "case_corpus": args.cases, "case_count": len(results),
        "config": args.config, "checkpoint": args.checkpoint,
        "step": metadata.get("step") if isinstance(metadata, dict) else None,
        "training_stage": metadata.get("training_stage") if isinstance(metadata, dict) else None,
        "device": args.device, "temperature": args.temperature,
        "top_k": args.top_k, "seed": args.seed, "max_new_tokens": args.max_new_tokens,
        "positive_cases": len(positives), "negative_cases": len(negatives),
        "call_emitted_rate": _rate(positives, "call_emitted"),
        "json_valid_rate": _rate(positives, "json_valid"),
        "name_accuracy": _rate(positives, "name_correct"),
        "name_in_provided_tools_rate": _rate(positives, "name_in_provided_tools"),
        "args_complete_rate": _rate(positives, "args_complete"),
        "overall_pass_rate": _rate(results, "pass"),
        "negative_no_call_rate": _rate(negatives, "pass"),
        "arguments_as_json_string_count": sum(r["arguments_as_json_string"] for r in results),
        "args_value_match_rate": _rate(positives, "args_value_match"),
        "call_sequence_correct_rate": _rate(results, "call_sequence_correct"),
        "fabricated_success_count": sum(r["fabricated_success"] for r in results),
        "recovery_outcomes": {},
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
          f"step={metadata.get('step') if isinstance(metadata, dict) else None} "
          f"stage={metadata.get('training_stage') if isinstance(metadata, dict) else None} "
          f"temperature={args.temperature} seed={args.seed}\n")

    results = []
    for case in selected_cases(args):
        prompt = render_prompt(tokenizer, case["tools"], case["query"])
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
        record = score_response(case, text, response_ids.numel(), args.max_new_tokens)
        results.append(record)
        verdict = "PASS" if record["pass"] else "FAIL"
        calls, _ = parse_calls(text)
        first = calls[0] if calls else None
        print(f"[{verdict}] {case['id']}  emitted={record['call_emitted']} json_valid={record['json_valid']} "
              f"name={first['name'] if first else None}")
        print(f"    回复: {text[:200]}\n")

    summary = build_summary(args, metadata, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
