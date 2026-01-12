from __future__ import annotations

import unittest

from autombse.pipeline.long_period import (
    PHASE_SEQUENCE,
    DiagramPlanItem,
    _map_type_to_stage,
    _phase_target_distribution,
    _prune_unknown_dependencies_with_known,
)


class TestLongPeriodPhasePlanningHelpers(unittest.TestCase):
    def test_phase_target_distribution_sums_to_total(self) -> None:
        counts = _phase_target_distribution(120)
        self.assertEqual(set(counts.keys()), set(PHASE_SEQUENCE))
        self.assertEqual(sum(counts.values()), 120)
        self.assertTrue(all(isinstance(v, int) for v in counts.values()))

    def test_phase_target_distribution_small_total(self) -> None:
        counts = _phase_target_distribution(3)
        self.assertEqual(set(counts.keys()), set(PHASE_SEQUENCE))
        self.assertEqual(sum(counts.values()), 3)

    def test_map_type_to_stage_accepts_prefixes(self) -> None:
        self.assertEqual(_map_type_to_stage("RD-NF"), "RD")
        self.assertEqual(_map_type_to_stage("RD_F"), "RD")
        self.assertEqual(_map_type_to_stage("bdd-subsystem"), "BDD")
        self.assertEqual(_map_type_to_stage("IBD-01"), "IBD")
        self.assertEqual(_map_type_to_stage("ad_ops"), "AD")
        self.assertEqual(_map_type_to_stage("PD-BUD"), "PD")

    def test_prune_unknown_dependencies_with_known(self) -> None:
        items = [
            DiagramPlanItem(
                diagram_id="X",
                type="BDD",
                title="t",
                goal="g",
                dependencies=["A", "B", "X", "C"],
                acceptance_checks=[],
            )
        ]
        pruned = _prune_unknown_dependencies_with_known(items, known={"A", "B", "X"})
        self.assertEqual(items[0].dependencies, ["A", "B"])
        self.assertEqual(len(pruned), 1)


if __name__ == "__main__":
    unittest.main()

