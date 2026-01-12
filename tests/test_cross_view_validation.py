from __future__ import annotations

import os
import tempfile
import unittest

from autombse.pipeline.long_period import _flatten_packages
from autombse.sysml.package_tree import parse_packages
from autombse.verification.engine import Rules


class TestCrossViewValidation(unittest.TestCase):
    def test_parse_packages_supports_requirement_connector_signal(self) -> None:
        sysml = """
        package p {
          requirement R1;
          connector C1;
          connection C2;
          signal S1;
        }
        """.strip()
        packages = parse_packages(sysml, {})
        self.assertEqual(len(packages), 1)
        root = packages[0]
        self.assertEqual(root.type, "package")
        child_types = [child.type for child in root.children]
        self.assertIn("requirement", child_types)
        self.assertEqual(child_types.count("connector"), 2)
        self.assertIn("signal", child_types)

    def test_cross_rules_pass_on_minimal_consistent_model(self) -> None:
        sysml_rd = """
        package rd {
          requirement R1;
          constraint C1;
        }
        """.strip()
        sysml_bdd = """
        package bdd {
          block R1 {
            attribute C1 : Real;
          }
        }
        """.strip()
        sysml_ibd = """
        package ibd {
          part C1 : R1;
        }
        """.strip()

        with tempfile.TemporaryDirectory() as tmpdir:
            old_state = os.environ.get("AUTOMBSE_RULE_STATE_FILE")
            os.environ["AUTOMBSE_RULE_STATE_FILE"] = os.path.join(tmpdir, "rule_states.json")
            try:
                flat = []
                flat.extend(_flatten_packages(parse_packages(sysml_rd, {})))
                flat.extend(_flatten_packages(parse_packages(sysml_bdd, {})))
                flat.extend(_flatten_packages(parse_packages(sysml_ibd, {})))
                errors, warnings = Rules(flat).validate_by_type("cross")
                self.assertEqual(errors, [])
                self.assertEqual(warnings, [])
            finally:
                if old_state is None:
                    os.environ.pop("AUTOMBSE_RULE_STATE_FILE", None)
                else:
                    os.environ["AUTOMBSE_RULE_STATE_FILE"] = old_state

    def test_cross_rules_flag_missing_attribute_instantiation(self) -> None:
        sysml_bdd = """
        package bdd {
          block R1 {
            attribute C1 : Real;
          }
        }
        """.strip()

        with tempfile.TemporaryDirectory() as tmpdir:
            old_state = os.environ.get("AUTOMBSE_RULE_STATE_FILE")
            os.environ["AUTOMBSE_RULE_STATE_FILE"] = os.path.join(tmpdir, "rule_states.json")
            try:
                flat = _flatten_packages(parse_packages(sysml_bdd, {}))
                errors, _warnings = Rules(flat).validate_by_type("cross")
                self.assertTrue(any(err.get("rule_id") == "BBD-1" for err in errors))
            finally:
                if old_state is None:
                    os.environ.pop("AUTOMBSE_RULE_STATE_FILE", None)
                else:
                    os.environ["AUTOMBSE_RULE_STATE_FILE"] = old_state

    def test_cross_rules_accept_qualified_part_instantiation(self) -> None:
        sysml_bdd = """
        package bdd {
          block Design {
            attribute x : Real;
          }
        }
        """.strip()
        sysml_ibd = """
        package ibd {
          part d : bdd::Design;
        }
        """.strip()

        with tempfile.TemporaryDirectory() as tmpdir:
            old_state = os.environ.get("AUTOMBSE_RULE_STATE_FILE")
            os.environ["AUTOMBSE_RULE_STATE_FILE"] = os.path.join(tmpdir, "rule_states.json")
            try:
                flat = []
                flat.extend(_flatten_packages(parse_packages(sysml_bdd, {})))
                flat.extend(_flatten_packages(parse_packages(sysml_ibd, {})))
                errors, warnings = Rules(flat).validate_by_type("cross")
                self.assertEqual(errors, [])
                self.assertEqual(warnings, [])
            finally:
                if old_state is None:
                    os.environ.pop("AUTOMBSE_RULE_STATE_FILE", None)
                else:
                    os.environ["AUTOMBSE_RULE_STATE_FILE"] = old_state

    def test_cross_rules_accept_satisfy_relations(self) -> None:
        sysml_rd = """
        package rd {
          requirement def R1 {
            attribute x : Real;
            require constraint{
              x >= 0.0
            }
          }
        }
        """.strip()
        sysml_bdd = """
        package bdd {
          block Design {
            attribute x : Real;
          }
          satisfy R1
            by Design::x;
        }
        """.strip()
        sysml_ibd = """
        package ibd {
          part x : Design;
        }
        """.strip()

        with tempfile.TemporaryDirectory() as tmpdir:
            old_state = os.environ.get("AUTOMBSE_RULE_STATE_FILE")
            os.environ["AUTOMBSE_RULE_STATE_FILE"] = os.path.join(tmpdir, "rule_states.json")
            try:
                flat = []
                flat.extend(_flatten_packages(parse_packages(sysml_rd, {})))
                flat.extend(_flatten_packages(parse_packages(sysml_bdd, {})))
                flat.extend(_flatten_packages(parse_packages(sysml_ibd, {})))
                errors, warnings = Rules(flat).validate_by_type("cross")
                self.assertEqual(errors, [])
                self.assertEqual(warnings, [])
            finally:
                if old_state is None:
                    os.environ.pop("AUTOMBSE_RULE_STATE_FILE", None)
                else:
                    os.environ["AUTOMBSE_RULE_STATE_FILE"] = old_state


if __name__ == "__main__":
    unittest.main()
