from __future__ import annotations

import unittest

from autombse.pipeline.long_period import extract_sysml_code
from autombse.sysml.package_tree import parse_packages


class TestLongPeriodUtils(unittest.TestCase):
    def test_extract_sysml_code_prefers_sysml_block(self) -> None:
        text = """
        prefix
        ```sysml
        package p { part a : B; }
        ```
        ```python
        print("ignore")
        ```
        """.strip()
        extracted = extract_sysml_code(text)
        self.assertIn("package p", extracted)
        self.assertNotIn('print("ignore")', extracted)

    def test_parse_packages_sets_belongsto(self) -> None:
        sysml = """
        package p {
          part a : B;
          attribute x : Real;
        }
        """.strip()
        packages = parse_packages(sysml, {})
        self.assertEqual(len(packages), 1)
        self.assertEqual(packages[0].type, "package")
        part = next(node for node in packages[0].children if node.type == "part")
        attr = next(node for node in packages[0].children if node.type == "attribute")
        self.assertEqual(part.belongsto, "B")
        self.assertEqual(attr.belongsto, "Real")

    def test_parse_packages_preserves_qualified_belongsto(self) -> None:
        sysml = """
        package p {
          package types {
            part def B;
          }
          part a : types::B;
        }
        """.strip()
        packages = parse_packages(sysml, {})
        self.assertEqual(len(packages), 1)
        root = packages[0]
        part = next(node for node in root.children if node.type == "part" and node.name == "a")
        self.assertEqual(part.belongsto, "types::B")


if __name__ == "__main__":
    unittest.main()
