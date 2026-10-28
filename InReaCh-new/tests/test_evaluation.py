"""框定位指标按手工例子验证，不把框伪装成像素掩码。"""
import unittest
import numpy as np
from contracts import Box, Prediction
import evaluation


def prediction(values, name='a'):
    scores = np.asarray(values, dtype=float)
    return Prediction(scores, np.ones(scores.shape, dtype=bool), name)


class EvaluationTests(unittest.TestCase):
    def test_maximum_matching_reassigns_overlapping_boxes(self):
        scores = np.zeros((3, 7))
        scores[1, 1] = 9
        scores[1, 5] = 8
        result = evaluation.evaluate_predictions({'a': prediction(scores)},
            {'a': (Box(0, 0, 0, 6, 3), Box(0, 1, 1, 2, 2))}, 5)
        self.assertEqual(result['matched_boxes'], 2)
        self.assertEqual(result['box_recall'], 1.)
        self.assertEqual(result['unmatched_regions_per_image'], 0.)
        self.assertAlmostEqual(result['anomaly_area_fraction'], 2/21)

    def test_diagonal_is_one_region_and_peak_tie_is_row_major(self):
        p = prediction([[8, 0, 0], [0, 8, 0], [0, 0, 0]])
        result = evaluation.evaluate_predictions({'a': p},
            {'a': (Box(0, 0, 0, 1, 1), Box(0, 1, 1, 2, 2))}, 5)
        self.assertEqual(result['predicted_regions'], 1)
        self.assertEqual(result['matched_boxes'], 1)
        self.assertEqual(result['box_recall'], .5)

    def test_unknown_labels_and_partial_coverage_are_not_normal(self):
        p = prediction([[1, 3]])
        result = evaluation.evaluate_predictions({'a': p}, {'a': None}, 2)
        self.assertIsNone(result['box_recall'])
        self.assertIsNone(result['image_auroc'])
        self.assertEqual(result['unknown_label_images'], 1)
        p.valid_map[0, 1] = False
        p.score_map[0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, '覆盖'):
            evaluation.evaluate_predictions({'a': p}, {'a': None}, 2)

    def test_image_auc_requires_complete_labels_and_both_classes(self):
        ps = {'a': prediction([[1]], 'a'), 'b': prediction([[4]], 'b')}
        boxes = {'a': (), 'b': (Box(0, 0, 0, 1, 1),)}
        self.assertIsNone(evaluation.evaluate_predictions(ps, boxes, 2)['image_auroc'])
        self.assertEqual(evaluation.evaluate_predictions(ps, boxes, 2, annotations_complete=True)['image_auroc'], 1.)

    def test_threshold_chosen_on_validation_under_explicit_budget(self):
        p = prediction([[8, 0, 3]])
        selection = evaluation.select_threshold({'a': p}, {'a': (Box(0, 0, 0, 1, 1),)},
                      [1, 5, 9], max_unmatched_regions_per_image=0, annotations_complete=True)
        self.assertEqual(selection['threshold'], 5)
        self.assertEqual(len(selection['curve']), 3)
        with self.assertRaises(ValueError):
            evaluation.select_threshold({'a': p}, {'a': None}, [5],
                      max_unmatched_regions_per_image=0, annotations_complete=True)


if __name__ == '__main__': unittest.main()
