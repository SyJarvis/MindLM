#!/usr/bin/env python3
"""三个预训练语料的内容分布画像（V2 打包前核查）。

对每个文件：
1. 全量一遍：行/文档数、按规则去重率（pretrain_data 用前150字符MD5，t2t 用全文MD5）、
   启发式分类计数（code/math/multi_turn/writing...，子串规则，非金标准）
2. 等距抽样：token 长度分布（qwen3 分词器）、字符长度、中英文字符占比
3. 汇总输出到 stdout + docs/pretrain_v2_data_profile.md + docs/data_profile_v2.json

用法：/Users/mac/base/bin/python3 scripts/profile_pretrain_data.py [--quick]
  （base 环境：Python 3.12 + tokenizers；--quick: 每文件只读前 200 万文档，快速 sanity check）
"""
import csv
import hashlib
import json
import os
import re
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_PATH = os.path.join(BASE, "qwen3_tokenizer", "tokenizer.json")
OUT_MD = os.path.join(BASE, "docs", "pretrain_v2_data_profile.md")
OUT_JSON = os.path.join(BASE, "docs", "data_profile_v2.json")

SOURCES = [
    # (key, path, format, sample_stride)
    ("pretrain_data", os.path.join(BASE, "data", "pretrain_data.csv"), "csv", 1000),
    ("pretrain_t2t", os.path.join(BASE, "data", "pretrain_t2t.jsonl"), "jsonl", 400),
    ("perfectblend_clean", os.path.join(BASE, "perfectblend_pretrain_clean.csv"), "csv", 70),
]

QUICK_LIMIT = 2_000_000

# ---------- 启发式分类规则（子串匹配，首个命中即归类，另统计多标签） ----------

T2T_RULES = [  # (标签, 规则列表：任一命中)
    ("code", ["```", "def ", "代码", "编程", "Python", "python", "函数", "算法"]),
    ("math", ["计算", "数学", "方程", "求解", "积分", "概率", "几何", "等于", "表达式"]),
    ("multi_turn", ["好的。现在", "好的，现在", "接下来请", "现在我会", "下一", "第二", "再次"]),
    ("writing", ["写作", "撰写", "写一篇", "写一封", "文章", "作文", "邮件", "摘要", "总结", "故事", "小说", "标题"]),
]

PB_RULES = [
    ("code", ["```", " def ", "class ", "import ", "function ", "def main", "public "]),
    ("math", ["solve", "calculate", "equation", "integral", "derivative", "probability", "math", "sum of", "average of"]),
]

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
LATIN_RE = re.compile(r"[A-Za-z]")


def classify(text, rules):
    """返回 (首个命中标签或 other, 所有命中标签列表)"""
    hits = [lab for lab, pats in rules if any(p in text for p in pats)]
    return (hits[0] if hits else "other"), hits


def iter_docs(path, fmt):
    if fmt == "csv":
        with open(path, "r", encoding="utf-8", newline="", errors="replace") as f:
            rd = csv.reader(f)
            next(rd, None)  # header
            for row in rd:
                yield row[0] if row else ""
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    yield json.loads(line).get("text", "")
                except Exception:
                    yield ""


def md5_head(s):
    return hashlib.md5(s[:150].encode("utf-8", "ignore")).digest()[:8]


def md5_full(s):
    return hashlib.md5(s.encode("utf-8", "ignore")).digest()[:8]


def profile(key, path, fmt, stride, quick=False):
    t0 = time.time()
    n = 0
    dup_prefix = dup_full = 0
    seen_prefix, seen_full = set(), set()
    cat_first, cat_multi, cat_docs = {}, {}, {}
    samples = []           # (text,)
    for text in iter_docs(path, fmt):
        n += 1
        # 分类
        rules = T2T_RULES if key == "pretrain_t2t" else PB_RULES
        if key == "pretrain_data":
            first, hits = "plain", []      # 通用网页语料不做内容分类
        else:
            first, hits = classify(text, rules)
            cat_first[first] = cat_first.get(first, 0) + 1
            for h in hits:
                cat_multi[h] = cat_multi.get(h, 0) + 1
            if len(cat_docs.get(first, [])) < 2:
                cat_docs.setdefault(first, []).append(text[:120].replace("\n", "⏎"))
        # 去重
        if key == "pretrain_data":
            h = md5_head(text)
            if h in seen_prefix:
                dup_prefix += 1
            else:
                seen_prefix.add(h)
            fh = md5_full(text)
            if fh in seen_full:
                dup_full += 1
            else:
                seen_full.add(fh)
        elif key == "pretrain_t2t":
            fh = md5_full(text)
            if fh in seen_full:
                dup_full += 1
            else:
                seen_full.add(fh)
        # 抽样
        if n % stride == 0:
            samples.append(text)
        if quick and n >= QUICK_LIMIT:
            break
        if n % 2_000_000 == 0:
            print(f"  [{key}] {n:,} docs, {time.time()-t0:.0f}s", flush=True)
    del seen_prefix, seen_full

    # 抽样统计：token / 字符 / 语言
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    enc = tok.encode_batch(samples, add_special_tokens=False)
    tl = [len(e.ids) for e in enc]
    cl = [len(s) for s in samples]
    cjk = [len(CJK_RE.findall(s)) / max(c, 1) for s, c in zip(samples, cl)]
    lat = [len(LATIN_RE.findall(s)) / max(c, 1) for s, c in zip(samples, cl)]

    def pct(a, p):
        b = sorted(a)
        return b[min(int(len(b) * p), len(b) - 1)] if b else 0

    mean_tok = sum(tl) / max(len(tl), 1)
    res = {
        "path": os.path.relpath(path, BASE),
        "docs": n,
        "quick_mode": quick,
        "scan_seconds": round(time.time() - t0, 1),
        "dup": {
            "prefix150_md5": dup_prefix if key == "pretrain_data" else None,
            "full_md5": dup_full if key in ("pretrain_data", "pretrain_t2t") else None,
        },
        "category_first_hit": cat_first,
        "category_multi_label": cat_multi,
        "category_examples": cat_docs,
        "sample": {
            "n": len(samples),
            "stride": stride,
            "tokens_mean": round(mean_tok, 1),
            "tokens_p50": pct(tl, 0.50),
            "tokens_p90": pct(tl, 0.90),
            "tokens_p99": pct(tl, 0.99),
            "chars_mean": round(sum(cl) / max(len(cl), 1), 1),
            "cjk_ratio_mean": round(sum(cjk) / max(len(cjk), 1), 4),
            "latin_ratio_mean": round(sum(lat) / max(len(lat), 1), 4),
            "docs_lt_20tok": sum(1 for t in tl if t < 20),
            "docs_gt_4096tok": sum(1 for t in tl if t > 4096),
        },
        "tokens_total_est_B": round(mean_tok * n / 1e9, 2),
    }
    print(f"  [{key}] done {n:,} docs in {time.time()-t0:.0f}s", flush=True)
    return res


def fmt_pct(x, total):
    return f"{100.0 * x / max(total, 1):.1f}%"


FORMAT_SECTION = """
## 数据格式（逐文件核实，2026-09-07）

| 来源 | 容器格式 | 行尾 | 文档边界 | 结构说明 |
|---|---|---|---|---|
| `data/pretrain_data.csv`（4.66GB） | CSV 单列 `text`，首行 header `text` | CRLF（列内 LF 为 `\\n` 转义） | 一行 = 一篇文档（多段落在引号内以 LF 分隔） | 5,364,883 个逻辑文档；文件物理行 2202 万，平均 ~4.1 物理行/文档（"2202 万行"是物理行口径）。**无任何 chat 模板标记** |
| `data/pretrain_t2t.jsonl`（8.28GB） | JSONL 单键 `{"text": ...}` | LF | 一行 = 一篇文档（多轮对话已拼接为纯文本，无角色标记） | 前 100 万行 0 解析失败、键恒为 `text`；**无任何 chat 模板标记** |
| `perfectblend_pretrain_clean.csv`（2.98GB） | CSV 单列 `text`，首行 header `text` | CRLF | 一行 = 一篇对话 | 全量含 `<|im_start|>/<|im_end|>` ChatML 标记（抽样 100% 命中）；对话结构：82% 单轮 `(1 user, 1 assistant)`，4.7% 2 轮，4.6% 3 轮，4.1% 仅 user 无回复 `(1,0)`；另混入 `### Instruction`(0.035%)、`<|endoftext|>`(1 例) 等异源残留 |

**对打包脚本 `build_mixed_v2.py` 的直接含义**：
1. 三个文件统一"一行=一文档"读取即可，`csv.reader`（Python 自动处理 CRLF+引号转义）与逐行 `json.loads` 均安全（实测 0 失败）。
2. 文档内 `\\n` 是真实换行（转义层在 CSV 引号/JSON 字符串里），tokenize 前不要做二次转义处理。
3. perfectblend 的 `<|im_start|>/<|im_end|>` 是真实特殊 token（qwen3 词表内），V1 打包即原样保留，V2 沿用；t2t 与 pretrain_data 无模板标记，两类语料在"格式"维度天然分层，无需清洗对齐。
"""


def render_md(results):
    L = ["# 预训练三源数据画像（实测）", ""]
    L.append("> 生成：`scripts/profile_pretrain_data.py`；分类为子串启发式（非人工标注），token 用 qwen3 分词器抽样统计。")
    L.append(FORMAT_SECTION)
    L.append("## 总览")
    L.append("")
    L.append("| 来源 | 文档数 | 估总 tokens | p50/p90/p99 tok | 中文占比 | 英文占比 | <20tok | >4096tok |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        s = r["sample"]
        L.append(
            f"| {r['path']} | {r['docs']:,} | {r['tokens_total_est_B']}B | "
            f"{s['tokens_p50']}/{s['tokens_p90']}/{s['tokens_p99']} | "
            f"{s['cjk_ratio_mean']:.1%} | {s['latin_ratio_mean']:.1%} | "
            f"{fmt_pct(s['docs_lt_20tok'], s['n'])} | {fmt_pct(s['docs_gt_4096tok'], s['n'])} |"
        )
    L.append("")
    L.append("## 去重（按 V2 方案规则）")
    L.append("")
    L.append("| 来源 | 前150字符MD5重复率 | 全文MD5重复率 |")
    L.append("|---|---|---|")
    for r in results:
        d = r["dup"]
        L.append(f"| {r['path']} | {fmt_pct(d['prefix150_md5'] or 0, r['docs']) if d['prefix150_md5'] is not None else '—'} | "
                 f"{fmt_pct(d['full_md5'] or 0, r['docs']) if d['full_md5'] is not None else '—'} |")
    for r in results:
        if r["category_first_hit"]:
            L.append("")
            L.append(f"## 内容分类：`{r['path']}`")
            L.append("")
            L.append("| 类别（首命中） | 文档数 | 占比 |（多标签命中数）|")
            L.append("|---|---|---|---|")
            for k, v in sorted(r["category_first_hit"].items(), key=lambda kv: -kv[1]):
                m = r["category_multi_label"].get(k, v)
                L.append(f"| {k} | {v:,} | {fmt_pct(v, r['docs'])} | {m:,} |")
            L.append("")
            L.append("示例（截断120字符）：")
            for k, exs in r["category_examples"].items():
                for e in exs:
                    L.append(f"- [{k}] {e}")
    return "\n".join(L) + "\n"


def main():
    quick = "--quick" in sys.argv
    results = []
    for key, path, fmt, stride in SOURCES:
        print(f"profiling {key} ...", flush=True)
        results.append(profile(key, path, fmt, stride, quick=quick))
    md = render_md(results)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(md)
    print(f"saved -> {OUT_MD} , {OUT_JSON}")


if __name__ == "__main__":
    main()
