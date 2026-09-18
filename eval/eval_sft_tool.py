"""Tool-calling probe for a MindLM SFT checkpoint.

Feeds the model the Qwen3 ``tools`` template block (the same rendering path used
for the UltraData agent samples) and checks whether the reply is a well-formed

    <tool_call>{"name": ..., "arguments": {...}}</tool_call>

block with the expected function name and required argument keys.

Metrics (see the printed summary / JSON output):
  * ``call_emitted``   - the reply contains a ``<tool_call>`` block at all
  * ``json_valid``     - every emitted block parses as JSON with name + arguments
  * ``name_correct``   - the first block names the expected function
  * ``args_complete``  - the expected argument keys are present
  * negative cases must NOT emit a call (plain chat must stay plain chat)

Usage:
  python3 eval/eval_sft_tool.py --checkpoint <ckpt> [--config mindlm_0.2b_gdn]
      [--device cpu] [--temperature 0.1] [--seed 1337] [--json_out results.json]
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

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# Schemas mirror the functions that appear in the UltraData agent samples used by
# training (retail-agent + a couple of generic ones), so the probe measures the
# behaviour the model was actually taught rather than an unseen schema.
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

CASES = [
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


def render_prompt(tokenizer, tools, query):
    messages = [{"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not isinstance(prompt, str):
        prompt = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in prompt)
    return prompt


def parse_calls(text):
    """Return (calls, errors) for every <tool_call> block found in ``text``."""
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
        calls.append({"name": payload["name"], "arguments": arguments, "arguments_was_string": isinstance(payload.get("arguments"), str)})
    return calls, errors


def main():
    parser = argparse.ArgumentParser(description="MindLM SFT tool-calling probe")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", choices=supported_configs(), default="mindlm_0.2b_gdn")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()

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
    for case in CASES:
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
        calls, errors = parse_calls(text)
        first = calls[0] if calls else None
        emitted = bool(calls)
        tool_names = {tool["function"]["name"] for tool in case["tools"]}
        record = {
            "id": case["id"],
            "expect_no_call": case["expect_no_call"],
            "expect_name": case["expect_name"],
            "expect_args": case["expect_args"],
            "call_emitted": emitted,
            "n_calls": len(calls),
            "json_valid": emitted and not errors,
            "parse_errors": errors,
            "name_correct": (first["name"] == case["expect_name"]) if first else False,
            # A name outside the supplied schemas cannot be dispatched by a real
            # tool runtime, so it is tracked separately from a plain wrong pick.
            "name_in_provided_tools": bool(first) and first["name"] in tool_names,
            "args_complete": bool(first) and all(key in first["arguments"] for key in case["expect_args"]),
            "arguments_as_json_string": bool(first and first["arguments_was_string"]),
            "text": text,
            "new_tokens": int(response_ids.numel()),
            "hit_eos": int(response_ids.numel()) < args.max_new_tokens,
        }
        if case["expect_no_call"]:
            record["pass"] = not emitted
        else:
            record["pass"] = (emitted and record["json_valid"] and record["name_correct"]
                              and record["args_complete"] and record["name_in_provided_tools"])
        results.append(record)

        verdict = "PASS" if record["pass"] else "FAIL"
        print(f"[{verdict}] {case['id']}  emitted={emitted} json_valid={record['json_valid']} "
              f"name={first['name'] if first else None} args={first['arguments'] if first else None}")
        print(f"    回复: {text[:300]}\n")

    positives = [r for r in results if not r["expect_no_call"]]
    negatives = [r for r in results if r["expect_no_call"]]
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "step": metadata.get("step") if isinstance(metadata, dict) else None,
        "training_stage": metadata.get("training_stage") if isinstance(metadata, dict) else None,
        "device": args.device,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "positive_cases": len(positives),
        "negative_cases": len(negatives),
        "call_emitted_rate": f"{sum(r['call_emitted'] for r in positives)}/{len(positives)}",
        "json_valid_rate": f"{sum(r['json_valid'] for r in positives)}/{len(positives)}",
        "name_accuracy": f"{sum(r['name_correct'] for r in positives)}/{len(positives)}",
        "name_in_provided_tools_rate": f"{sum(r['name_in_provided_tools'] for r in positives)}/{len(positives)}",
        "args_complete_rate": f"{sum(r['args_complete'] for r in positives)}/{len(positives)}",
        "overall_pass_rate": f"{sum(r['pass'] for r in results)}/{len(results)}",
        "negative_no_call_rate": f"{sum(r['pass'] for r in negatives)}/{len(negatives)}",
        "arguments_as_json_string_count": sum(r["arguments_as_json_string"] for r in results),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
