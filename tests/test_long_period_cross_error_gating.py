from __future__ import annotations

import os
import tempfile
import unittest

from autombse.pipeline.long_period import _filter_cross_errors_for_stage, _flatten_packages
from autombse.sysml.package_tree import parse_packages
from autombse.verification.engine import Rules


class TestLongPeriodCrossErrorGating(unittest.TestCase):
    def test_rd_stage_defers_rd_cross_rules(self) -> None:
        sysml_rd = """
        package rd {
          requirement R1;
          constraint C1;
        }
        """.strip()

        with tempfile.TemporaryDirectory() as tmpdir:
            old_state = os.environ.get("AUTOMBSE_RULE_STATE_FILE")
            os.environ["AUTOMBSE_RULE_STATE_FILE"] = os.path.join(tmpdir, "rule_states.json")
            try:
                flat = _flatten_packages(parse_packages(sysml_rd, {}))
                errors, _warnings = Rules(flat).validate_by_type("cross")
                rule_ids = {e.get("rule_id") for e in errors if isinstance(e, dict)}
                self.assertIn("RD-1", rule_ids)
                self.assertIn("RD-2", rule_ids)

                self.assertEqual(_filter_cross_errors_for_stage(errors, stage="RD"), [])
                enforced = _filter_cross_errors_for_stage(errors, stage="BDD")
                enforced_ids = {e.get("rule_id") for e in enforced if isinstance(e, dict)}
                self.assertIn("RD-1", enforced_ids)
                self.assertIn("RD-2", enforced_ids)
            finally:
                if old_state is None:
                    os.environ.pop("AUTOMBSE_RULE_STATE_FILE", None)
                else:
                    os.environ["AUTOMBSE_RULE_STATE_FILE"] = old_state

    def test_bdd_stage_defers_bbd_cross_rules_until_ibd(self) -> None:
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
                rule_ids = {e.get("rule_id") for e in errors if isinstance(e, dict)}
                self.assertIn("BBD-1", rule_ids)

                self.assertEqual(_filter_cross_errors_for_stage(errors, stage="BDD"), [])
                enforced = _filter_cross_errors_for_stage(errors, stage="IBD")
                enforced_ids = {e.get("rule_id") for e in enforced if isinstance(e, dict)}
                self.assertIn("BBD-1", enforced_ids)
            finally:
                if old_state is None:
                    os.environ.pop("AUTOMBSE_RULE_STATE_FILE", None)
                else:
                    os.environ["AUTOMBSE_RULE_STATE_FILE"] = old_state


if __name__ == "__main__":
    unittest.main()

