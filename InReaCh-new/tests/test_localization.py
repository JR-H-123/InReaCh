"""用原图像素手算定位区域，不依赖颜色外观。"""
import unittest
import numpy as np
from contracts import Prediction
from localization import extract_regions


class LocalizationTests(unittest.TestCase):
    def test_diagonal_peak_tie_bbox_area_and_invalid_high_score(self):
        scores = np.zeros((3, 5))
        scores[0, 0] = scores[1, 1] = 5
        scores[2, 4] = 8
        scores[0, 4] = 100
        valid = np.ones((3, 5), dtype=bool)
        valid[0, 4] = False
        regions = extract_regions(Prediction(scores, valid, 'original'), 5)
        self.assertEqual(regions, [
            {'region_id': 1, 'peak_xy': [0.5, 0.5], 'peak_score': 5.,
             'bbox_xyxy': [0, 0, 2, 2], 'area_px': 2},
            {'region_id': 2, 'peak_xy': [4.5, 2.5], 'peak_score': 8.,
             'bbox_xyxy': [4, 2, 5, 3], 'area_px': 1}])

    def test_threshold_boundary_and_all_invalid(self):
        p = Prediction(np.array([[2., 3.]]), np.array([[True, True]]), 'a')
        self.assertEqual(extract_regions(p, 3)[0]['peak_xy'], [1.5, .5])
        p.valid_map[:] = False
        p.score_map[:] = np.nan
        self.assertEqual(extract_regions(p, 3), [])

    def test_rejects_nonfinite_valid_values_and_threshold(self):
        p = Prediction(np.array([[np.nan]]), np.array([[True]]), 'a')
        with self.assertRaises(ValueError):
            extract_regions(p, 1)
        p.score_map[:] = 1
        with self.assertRaises(ValueError):
            extract_regions(p, float('nan'))


if __name__ == '__main__': unittest.main()
