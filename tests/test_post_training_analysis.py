import unittest

import numpy as np

from hccr.evaluation.post_training_analysis import (
    overlap_statistics, paired_writer_interval, prediction_statistics,
)


class PostTrainingAnalysisTests(unittest.TestCase):
    def test_recall_false_positives_and_selected_classes(self):
        targets = np.array([0, 0, 1, 2])
        predictions = np.array([[0, 1, 2, 3, 4], [1, 0, 2, 3, 4],
                                [1, 0, 2, 3, 4], [1, 2, 0, 3, 4]])
        result = prediction_statistics(targets, predictions, [0], classes=5)
        self.assertEqual(result["top1"], 0.5)
        self.assertEqual(result["top5"], 1.0)
        self.assertEqual(result["in_H"], {"samples": 2, "top1": 0.5})
        self.assertEqual(result["per_class_false_positives"], [0, 2, 0, 0, 0])

    def test_bootstrap_uses_writers_instead_of_independent_images(self):
        # One stratum, two equal-size clusters. Resampling either writer twice
        # reaches -100 or +100 pp; independent-image resampling has narrower tails.
        result = paired_writer_interval(
            [1] * 100 + [-1] * 100, ["a"] * 100 + ["b"] * 100,
            ["HWDB1.0"] * 200, repetitions=2000, seed=1,
        )
        self.assertEqual(result["ci95_pp"], [-100.0, 100.0])
        self.assertEqual(result["difference_pp"], 0.0)
        self.assertEqual(result["writers"], 2)

    def test_cross_version_writer_is_one_cluster(self):
        result = paired_writer_interval(
            [1, 0, -1], ["a", "a", "b"], ["v1", "v2", "v1"],
            repetitions=10,
        )
        self.assertEqual(result["strata_writer_counts"], {"v1+v2": 1, "v1": 1})
        self.assertEqual(result["ci95_pp"], [0.0, 0.0])

    def test_mask_overlap(self):
        self.assertEqual(overlap_statistics({7: [1, 2], 17: [2, 3]})[0]["intersection"], 1)

    def test_reject_misaligned_inputs(self):
        with self.assertRaises(ValueError):
            paired_writer_interval([1, 0], ["a"], ["v1", "v1"])


if __name__ == "__main__":
    unittest.main()
