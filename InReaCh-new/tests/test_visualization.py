"""导出只映射显示，定位与原始分数独立于颜色范围。"""
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from contracts import Prediction
from visualization import export_prediction, render_prediction, select_display_range


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = np.full((5, 7, 3), (255, 0, 0), dtype=np.uint8)
        values = np.zeros((5, 7))
        values[1, 2] = 100.
        valid = np.ones((5, 7), dtype=bool)
        valid[4, 6] = False
        values[4, 6] = np.nan
        self.p = Prediction(values, valid, 'original/中文')

    def test_fixed_scale_preserves_same_score_color_across_images(self):
        first = Prediction(np.array([[5., 1.]]), np.ones((1, 2), dtype=bool), 'a')
        second = Prediction(np.array([[5., 1000.]]), np.ones((1, 2), dtype=bool), 'b')
        image = np.zeros((1, 2, 3), dtype=np.uint8)
        a, _ = render_prediction(image, first, (0., 10.))
        b, _ = render_prediction(image, second, (0., 10.))
        np.testing.assert_array_equal(a[0, 0], b[0, 0])
        self.assertFalse(np.array_equal(a[0, 0], b[0, 1]))

    def test_export_preserves_rgb_coordinates_raw_score_and_invalidity(self):
        original_image = self.image.copy()
        original_scores = self.p.score_map.copy()
        files = export_prediction(self.image, self.p, self.root, 'sample',
                                  score_range=(0., 10.), threshold=50., alpha=0.)
        overlay = cv2.cvtColor(cv2.imread(str(self.root/files['overlay_file'])), cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(overlay[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(overlay[4, 6], [128, 128, 128])
        self.assertEqual(overlay.shape, (5, 7, 3))
        data = json.loads((self.root/files['detections_file']).read_text())
        self.assertEqual(data['image_id'], 'original/中文')
        self.assertEqual(data['regions'][0]['peak_xy'], [2.5, 1.5])
        self.assertEqual(data['regions'][0]['peak_score'], 100.)
        self.assertEqual(data['regions'][0]['bbox_xyxy'], [2, 1, 3, 2])
        self.assertEqual(data['coverage'], 34/35)
        marked = cv2.cvtColor(cv2.imread(str(self.root/files['localization_file'])), cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(marked[1, 2], [255, 255, 0])
        np.testing.assert_array_equal(marked[4, 6], [128, 128, 128])
        np.testing.assert_array_equal(self.image, original_image)
        np.testing.assert_array_equal(self.p.score_map, original_scores)

    def test_missing_threshold_is_pending_not_empty_detection(self):
        files = export_prediction(self.image, self.p, self.root, 'pending', score_range=(0., 10.))
        data = json.loads((self.root/files['detections_file']).read_text())
        self.assertEqual(data['localization_status'], 'threshold_not_selected')
        self.assertIsNone(data['regions'])
        self.assertIsNone(files['localization_file'])

    def test_constant_and_all_invalid_maps_have_defined_output(self):
        for all_invalid in (False, True):
            with self.subTest(all_invalid=all_invalid):
                p = Prediction(np.zeros((5, 7)), np.full((5, 7), not all_invalid), 'a')
                if all_invalid:
                    p.score_map[:] = np.nan
                files = export_prediction(self.image, p, self.root, str(all_invalid),
                                          score_range=(0., 1.), threshold=1.)
                data = json.loads((self.root/files['detections_file']).read_text())
                self.assertEqual(data['regions'], [])
                self.assertEqual(data['coverage'], 0. if all_invalid else 1.)
                if all_invalid:
                    gray = cv2.imread(str(self.root/files['heatmap_file']))
                    np.testing.assert_array_equal(gray, np.full((5, 7, 3), 128))

    def test_display_range_uses_only_valid_validation_values(self):
        self.assertEqual(select_display_range([self.p]), (0., 100.))
        self.p.valid_map[:] = False
        self.assertEqual(select_display_range([self.p]), (0., 1.))
        self.p.valid_map[:] = True
        self.p.score_map[:] = 0.
        self.assertEqual(select_display_range([self.p]), (0., 1.))

    def test_bad_geometry_ranges_alpha_identity_and_stem_are_rejected(self):
        for bounds in ((1., 1.), (float('nan'), 1.), (0., float('inf'))):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                render_prediction(self.image, self.p, bounds)
        with self.assertRaises(ValueError):
            render_prediction(self.image, self.p, (0., 10.), alpha=2.)
        with self.assertRaises(ValueError):
            render_prediction(self.image[:-1], self.p, (0., 10.))
        with self.assertRaises(ValueError):
            export_prediction(self.image, self.p, self.root, '../escape', score_range=(0., 10.))
        self.p.image_id = None
        with self.assertRaises(ValueError):
            export_prediction(self.image, self.p, self.root, 'bad', score_range=(0., 10.))


if __name__ == '__main__': unittest.main()
