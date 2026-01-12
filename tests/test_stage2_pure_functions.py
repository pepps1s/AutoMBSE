from __future__ import annotations

import unittest

from autombse.sysml.code_blocks import extractStage, extractStageWoStage
from autombse.sysml.parts import partComponentDepose


class TestStage2PureFunctions(unittest.TestCase):
    def test_part_component_depose_removes_comments(self) -> None:
        text = """
        part a : B; // comment
        part c : D;
        """.strip()
        parts = partComponentDepose(text)
        self.assertEqual(len(parts), 1)
        self.assertNotIn("//", parts[0])
        self.assertTrue(parts[0].startswith("part"))
        self.assertIn("part a", parts[0])
        self.assertIn("part c", parts[0])

    def test_part_component_depose_braces(self) -> None:
        text = """
        part a : B {
          part inner : C {
            part leaf : D;
          }
        }
        """.strip()
        parts = partComponentDepose(text)
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0].startswith("part a"))
        self.assertTrue(parts[0].endswith("}"))
        self.assertTrue(parts[1].startswith("part inner"))

    def test_extract_stage_wo_stage_multiple_blocks(self) -> None:
        text = """
        ```sysml
        part a : B;
        ```
        blah
        ```sysml
        part c : D;
        ```
        """.strip()
        blocks = extractStageWoStage(text)
        self.assertEqual(len(blocks), 2)
        self.assertIn("part a", blocks[0])
        self.assertIn("part c", blocks[1])

    def test_extract_stage_assigns_stage_labels(self) -> None:
        text = """
        ```sysml
        part a : B;
        part c : D;
        ```
        ```sysml
        part e : F;
        part g : H;
        ```
        """.strip()
        items = extractStage(text)
        self.assertEqual([i["stage"] for i in items], ["RD1", "RD2"])
        self.assertIn("part a", items[0]["part"])
        self.assertIn("part e", items[1]["part"])


if __name__ == "__main__":
    unittest.main()
