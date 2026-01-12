from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from autombse.pipeline.long_period import TokenCounter, TokenLedger, _estimate_component_breakdown


class TestLongPeriodTokenTrace(unittest.TestCase):
    def test_estimate_component_breakdown_sections(self) -> None:
        token_counter = TokenCounter(model="test")
        token_counter._encoder = None  # force deterministic fallback
        context = "## Target System\n\nAAA\n\n## Input Specs\n\nBBB"
        breakdown = _estimate_component_breakdown(context, token_counter)
        self.assertEqual([b["name"] for b in breakdown], ["Target System", "Input Specs"])
        self.assertTrue(all(isinstance(b.get("tokens_est"), int) for b in breakdown))
        self.assertTrue(all(int(b.get("tokens_est") or 0) > 0 for b in breakdown))

    def test_token_ledger_upgrades_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "token_trace.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "ts",
                        "event_type",
                        "span_id",
                        "budget_total",
                        "prompt_est",
                        "prompt_actual",
                        "completion_actual",
                        "total_actual",
                        "delta_vs_prev",
                    ]
                )
                writer.writerow(
                    [
                        "2020-01-01T00:00:00Z",
                        "llm.call",
                        "span0",
                        "10",
                        "1",
                        "1",
                        "1",
                        "2",
                        "",
                    ]
                )

            ledger = TokenLedger(csv_path=path)
            with path.open("r", newline="", encoding="utf-8") as f:
                header = next(csv.reader(f))
            self.assertIn("prompt_est_system", header)
            self.assertIn("prompt_est_task", header)
            self.assertIn("prompt_est_components", header)
            self.assertIn("prompt_est_parts_total", header)
            self.assertIn("components_breakdown", header)

            ledger.append(
                {
                    "ts": "2020-01-01T00:00:01Z",
                    "event_type": "llm.call",
                    "span_id": "span1",
                    "budget_total": 10,
                    "prompt_est": 7,
                    "prompt_actual": 6,
                    "completion_actual": 1,
                    "total_actual": 7,
                    "delta_vs_prev": 5,
                    "prompt_est_system": 1,
                    "prompt_est_task": 2,
                    "prompt_est_components": 3,
                    "prompt_est_parts_total": 6,
                    "components_breakdown": "[]",
                }
            )

            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["span_id"], "span1")
            self.assertEqual(rows[-1]["prompt_est_system"], "1")
            self.assertEqual(rows[-1]["components_breakdown"], "[]")


if __name__ == "__main__":
    unittest.main()

