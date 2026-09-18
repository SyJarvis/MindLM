#!/usr/bin/env python3
"""从 perfectblend_pretrain_clean.csv 抽离 math/code 子集为带标注 CSV。

输出 data/perfectblend_mathcode.csv，字段：id, text, len, type, language
  - id:   pb_{源文件行号:07d}（含 header 偏移，可回溯源文件）
  - text: 纯文本（已去除 ChatML 标记 `<|im_start|>role\n` / `<|im_end|>`，
          模板由 SFT 阶段 apply_chat_template 动态添加）
  - len:  字符数（不加载 tokenizer）
  - type: code / math（子串启发式，规则与画像脚本一致：先 code 后 math）
  - language: en / zh / mixed / other（逐行字符比例检测）

统计写 docs/perfectblend_mathcode_stats.json。

用法：/Users/mac/base/bin/python3 scripts/extract_perfectblend_mathcode.py
"""
import csv
import json
import os
import re
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "perfectblend_pretrain_clean.csv")
OUT = os.path.join(BASE, "data", "perfectblend_mathcode.csv")
STATS = os.path.join(BASE, "docs", "perfectblend_mathcode_stats.json")


# 规则与 profile_pretrain_data.py 保持一致（code 优先于 math）
CODE_PATS = ["```", " def ", "class ", "import ", "function ", "def main", "public "]
MATH_PATS = ["solve", "calculate", "equation", "integral", "derivative",
             "probability", "math", "sum of", "average of"]

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
LATIN_RE = re.compile(r"[A-Za-z]")

# ChatML 标记剥离：<|im_start|>role\n 与 <|im_end|>（模板由 SFT 的 chat_template 动态加；容忍 \r\n）
CHATML_RE = re.compile(r"<\|im_start\|>(user|assistant|system)\r?\n|<\|im_end\|>")


def strip_chatml(text):
    return CHATML_RE.sub("", text)


def classify(text):
    if any(p in text for p in CODE_PATS):
        return "code"
    if any(p in text for p in MATH_PATS):
        return "math"
    return None


def language(text):
    n = max(len(text), 1)
    cjk = len(CJK_RE.findall(text)) / n
    lat = len(LATIN_RE.findall(text)) / n
    if cjk > 0.3:
        return "zh"
    if lat > 0.5:
        return "en"
    if cjk > 0.05:
        return "mixed"
    return "other"


def main():
    t0 = time.time()

    stats = {
        "source": os.path.relpath(SRC, BASE),
        "output": os.path.relpath(OUT, BASE),
        "rules": {"code_first": True, "code_pats": CODE_PATS, "math_pats": MATH_PATS},
        "len_definition": "characters",
        "src_rows": 0,
        "kept": 0,
        "by_type": {},
        "by_language": {},
        "chars_by_type": {},
        "chars_total": 0,
    }

    out_rows_buf = []

    def flush_buf():
        """统计并写出缓冲区中的行（5 元组：id, text, len, type, language）"""
        if not out_rows_buf:
            return
        for rid, text, L, typ, lang in out_rows_buf:
            stats["kept"] += 1
            stats["by_type"][typ] = stats["by_type"].get(typ, 0) + 1
            stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1
            stats["chars_by_type"][typ] = stats["chars_by_type"].get(typ, 0) + L
            stats["chars_total"] += L
        writer.writerows(out_rows_buf)
        out_rows_buf.clear()

    with open(SRC, newline="", encoding="utf-8", errors="replace") as f, \
         open(OUT, "w", newline="", encoding="utf-8") as fo:
        rd = csv.reader(f)
        next(rd, None)  # header
        writer = csv.writer(fo, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "text", "len", "type", "language"])
        for lineno, row in enumerate(rd, start=2):  # 源文件物理行号（1-based，header=1）
            stats["src_rows"] += 1
            text = row[0] if row else ""
            typ = classify(text)
            if typ is None:
                continue
            text = strip_chatml(text)
            out_rows_buf.append((f"pb_{lineno:07d}", text, len(text), typ, language(text)))
            if len(out_rows_buf) >= 20000:
                flush_buf()
                if stats["src_rows"] % 200_000 == 0:
                    print(f"  src {stats['src_rows']:,} rows, kept {stats['kept']:,}, "
                          f"{time.time()-t0:.0f}s", flush=True)
        flush_buf()

    stats["duration_s"] = round(time.time() - t0, 1)
    stats["out_size_GB"] = round(os.path.getsize(OUT) / 1e9, 2)
    with open(STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "rules"},
                     ensure_ascii=False, indent=2))
    print(f"saved -> {OUT} , {STATS}")


if __name__ == "__main__":
    main()
