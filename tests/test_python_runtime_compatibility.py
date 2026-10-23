from __future__ import annotations

import ast
import unittest
from pathlib import Path


class PythonRuntimeCompatibilityTests(unittest.TestCase):
    def test_source_and_scripts_parse_with_supported_python_312_grammar(self):
        root = Path(__file__).resolve().parents[1]
        sources = sorted(
            path
            for directory in (root / "src", root / "scripts")
            for path in directory.rglob("*.py")
        )
        self.assertTrue(sources, "Expected Python source and executable scripts")
        for source in sources:
            with self.subTest(source=source.relative_to(root).as_posix()):
                ast.parse(
                    source.read_text(encoding="utf-8-sig"),
                    filename=str(source),
                    feature_version=(3, 12),
                )


if __name__ == "__main__":
    unittest.main()
