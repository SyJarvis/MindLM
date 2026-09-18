"""Import smoke tests for evaluation entry points and compatibility shims."""

import importlib


def test_tool_call_report_imports():
    module = importlib.import_module("eval.tool_call_report")
    assert callable(module.main)
    assert "Passed:" in module.build_report({"results": []})


def test_root_report_entry_point_imports():
    module = importlib.import_module("gen_tool_call_report")
    assert callable(module.main)
