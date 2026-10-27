"""Small real-CLI smoke test for the zvec-grep tool."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coding_agent.zvec import ZvecGrepSettings, ZvecGrepTool


class ZvecGrepSmokeTest(unittest.TestCase):
    def test_managed_ripgrep_finds_python_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "demo.py").write_text(
                "def zvec_smoke_symbol():\n    return 'ok'\n",
                encoding="utf-8",
            )
            tool = ZvecGrepTool(ZvecGrepSettings(timeout_seconds=15))
            result = tool.query(directory, "zvec_smoke_symbol", route="rg")

        self.assertIn("zvec_smoke_symbol", result)
        self.assertIn("demo.py", result)


if __name__ == "__main__":
    unittest.main()
