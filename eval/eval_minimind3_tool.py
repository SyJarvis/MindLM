"""Tool-calling probe for minimind-3 (HuggingFace AutoModel).
Same test cases as eval_sft_tool.py for direct comparison."""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

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

CASES = POSITIVE_CASES + NEGATIVE_CASES


def render_prompt(tokenizer, tools, query):
    messages = [{"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not isinstance(prompt, str):
        prompt = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in prompt)
    return prompt


def parse_calls(text):
    calls, errors = [], []
    for block in TOOL_CALL_RE.findall(text):
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


def score_response(case, text, new_tokens, max_new_tokens):
    calls, errors = parse_calls(text)
    first = calls[0] if calls else None
    emitted = bool(calls)
    tool_names = {tool["function"]["name"] for tool in case["tools"]}
    expect_no_call = case["expect_no_call"]
    expect_name = case["expect_name"]
    expect_args = case["expect_args"]

    name_correct = (first["name"] == expect_name) if first and expect_name else (first is None and expect_name is None)
    name_in_provided = bool(first) and first["name"] in tool_names
    args_complete = bool(first) and (not expect_args or all(key in first["arguments"] for key in expect_args))

    if expect_no_call:
        passed = not emitted
    else:
        passed = (emitted and not errors and name_correct and name_in_provided and args_complete)

    return {
        "id": case["id"], "expect_no_call": expect_no_call,
        "expect_name": expect_name, "expect_args": expect_args,
        "call_emitted": emitted, "n_calls": len(calls),
        "json_valid": emitted and not errors, "parse_errors": errors,
        "name_correct": name_correct,
        "name_in_provided_tools": name_in_provided,
        "args_complete": args_complete,
        "text": text, "new_tokens": int(new_tokens),
        "hit_eos": int(new_tokens) < max_new_tokens,
        "pass": passed,
    }


def _rate(records, field):
    return f"{sum(bool(record[field]) for record in records)}/{len(records)}"


def main():
    parser = argparse.ArgumentParser(description="minimind-3 tool-calling probe")
    parser.add_argument("--checkpoint", default="/root/.cache/modelscope/models/gongjy--minimind-3/snapshots/master")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json_out", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print(f"Loading model from {args.checkpoint} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True)
    model.eval()
    print(f"Model loaded. device={args.device} temperature={args.temperature} seed={args.seed}\n")

    cases = CASES[:args.limit] if args.limit else CASES
    results = []
    for case in cases:
        prompt = render_prompt(tokenizer, case["tools"], case["query"])
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                do_sample=True,
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

    positives = [r for r in results if not r["expect_no_call"]]
    negatives = [r for r in results if r["expect_no_call"]]
    summary = {
        "model": "minimind-3", "checkpoint": args.checkpoint,
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
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
