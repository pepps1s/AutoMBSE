from __future__ import annotations

import math
import unittest

from autombse.pipeline.semantic_entropy import semantic_entropy_confidence


class TestSemanticEntropy(unittest.TestCase):
    def test_semantic_entropy_all_same_confidence_one(self) -> None:
        sysml = "package p {\n  part a : A;\n}\n"
        res = semantic_entropy_confidence([sysml, sysml, sysml], similarity_threshold=0.85)
        self.assertEqual(res.samples, 3)
        self.assertEqual(res.clusters, 1)
        self.assertEqual(res.cluster_sizes, [3])
        self.assertAlmostEqual(res.entropy_bits, 0.0, places=6)
        self.assertAlmostEqual(res.normalized_entropy, 0.0, places=6)
        self.assertAlmostEqual(res.confidence, 1.0, places=6)

    def test_semantic_entropy_two_vs_one(self) -> None:
        a = "package p {\n  part a : A;\n}\n"
        b = "package p {\n  part b : B;\n}\n"
        res = semantic_entropy_confidence([a, a, b], similarity_threshold=0.85)
        self.assertEqual(res.samples, 3)
        self.assertEqual(res.clusters, 2)
        self.assertEqual(sorted(res.cluster_sizes), [1, 2])
        expected_entropy = -(2 / 3) * math.log2(2 / 3) - (1 / 3) * math.log2(1 / 3)
        expected_norm = expected_entropy / math.log2(3)
        self.assertAlmostEqual(res.entropy_bits, expected_entropy, places=6)
        self.assertAlmostEqual(res.normalized_entropy, expected_norm, places=6)
        self.assertAlmostEqual(res.confidence, 1.0 - expected_norm, places=6)

    def test_semantic_entropy_counts_invalid_samples(self) -> None:
        valid = "package p {\n  part a : A;\n}\n"
        invalid = ""
        res = semantic_entropy_confidence([valid, invalid], similarity_threshold=0.85)
        self.assertEqual(res.samples, 2)
        self.assertGreaterEqual(res.invalid_samples, 1)


if __name__ == "__main__":
    unittest.main()
