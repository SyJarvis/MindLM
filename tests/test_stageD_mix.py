"""Stage D mix-sampling tests: deterministic per-source subset extraction.

Runs the real CLI (data_process/stageD_sample_mix.py) on a synthetic CSV and
checks the realized per-source counts, structure preservation and seed
determinism.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "data_process" / "stageD_sample_mix.py"


def build_csv(path):
    rows = []
    for i in range(60):
        rows.append({"messages": json.dumps([{"role": "user", "content": f"q{i}"},
                                             {"role": "assistant", "content": f"a{i}"}]),
                     "tools": "", "source": "t2t_qa", "domain": "x"})
    for i in range(40):
        rows.append({"messages": json.dumps([{"role": "user", "content": f"q{i}"},
                                             {"role": "assistant", "content": f"a{i}"}]),
                     "tools": "", "source": "legacy", "domain": "x"})
    for i in range(20):
        rows.append({"messages": json.dumps([{"role": "user", "content": f"en{i}"}]),
                     "tools": "", "source": "perfectblend_en", "domain": "x"})
    for i in range(5):
        rows.append({"messages": "[]",
                     "tools": json.dumps([{"type": "function", "function": {"name": "f"}}]),
                     "source": "xlam_fc", "domain": "x"})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["messages", "tools", "source", "domain"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


class StageDMixTest(unittest.TestCase):
    def run_script(self, tmp, name, frac="0.5", cap="0.5"):
        out = os.path.join(tmp, f"{name}.csv")
        stats = os.path.join(tmp, f"{name}.json")
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--csv", os.path.join(tmp, "in.csv"),
             "--out", out, "--stats", stats,
             "--zh_qa_keep_frac", frac, "--perfectblend_en_cap", cap],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        return out, json.load(open(stats, encoding="utf-8"))

    def test_counts_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_csv(os.path.join(tmp, "in.csv"))
            out1, stats1 = self.run_script(tmp, "out1")
            out2, stats2 = self.run_script(tmp, "out2")
            self.assertEqual(stats1["realized"]["t2t_qa"], 30)
            self.assertEqual(stats1["realized"]["legacy"], 20)
            self.assertEqual(stats1["realized"]["perfectblend_en"], 10)
            self.assertEqual(stats1["realized"]["xlam_fc"], 5)
            self.assertEqual(stats1["realized"]["ultra"], 0)
            self.assertEqual(stats1["output_rows"], 65)
            # Same seed → identical files (deterministic order too: pass-2 streams).
            with open(out1, encoding="utf-8") as f1, open(out2, encoding="utf-8") as f2:
                self.assertEqual(f1.read(), f2.read())
            # Kept rows preserve the original columns verbatim.
            with open(out1, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 65)
            tools_rows = [r for r in rows if r["source"] == "xlam_fc"]
            self.assertEqual(len(tools_rows), 5)
            self.assertTrue(all(r["tools"] for r in tools_rows))

    def test_keep_all_when_frac_is_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_csv(os.path.join(tmp, "in.csv"))
            _, stats = self.run_script(tmp, "out", frac="1.0", cap="1.0")
            self.assertEqual(stats["output_rows"], 125)
            self.assertEqual(stats["realized"]["t2t_qa"], 60)
            self.assertEqual(stats["realized"]["legacy"], 40)

    def test_rejects_fraction_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_csv(os.path.join(tmp, "in.csv"))
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--csv", os.path.join(tmp, "in.csv"),
                 "--out", os.path.join(tmp, "o.csv"), "--stats", os.path.join(tmp, "s.json"),
                 "--zh_qa_keep_frac", "1.5"],
                capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("zh_qa_keep_frac", proc.stderr)

    def test_rejects_unmapped_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "in.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["messages", "tools", "source", "domain"])
                writer.writeheader()
                writer.writerow({"messages": "[]", "tools": "", "source": "mystery", "domain": ""})
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--csv", path,
                 "--out", os.path.join(tmp, "o.csv"), "--stats", os.path.join(tmp, "s.json")],
                capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("unmapped", proc.stderr)


if __name__ == "__main__":
    unittest.main()
