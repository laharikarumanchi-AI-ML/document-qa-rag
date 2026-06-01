"""Smoke test for the Streamlit demo.

We can't run the demo in unit tests (Streamlit needs its runtime to
render anything), but we CAN verify the file parses and imports
cleanly. This catches:

- Syntax errors in the demo file
- Missing imports
- Wrong module references (e.g., refactor that renamed a function
  ragqa.cli depends on)

It's the cheapest possible safety net for the demo.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_demo_app_file_exists():
    demo_app = Path(__file__).parent.parent / "demo" / "app.py"
    assert demo_app.is_file()


@pytest.mark.skipif(
    importlib.util.find_spec("streamlit") is None,
    reason="streamlit not installed (skip in dev-only environments)",
)
def test_demo_app_imports_without_error():
    """Verify the demo's imports + top-level code parses without raising.

    Streamlit's top-level functions (st.set_page_config etc.) are safe
    to call at import time — they're no-ops outside the streamlit
    runtime. So importing the module IS a meaningful test."""
    demo_app = Path(__file__).parent.parent / "demo" / "app.py"
    spec = importlib.util.spec_from_file_location("demo_app", demo_app)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    # Will raise on any import failure or top-level syntax/name error.
    spec.loader.exec_module(module)
