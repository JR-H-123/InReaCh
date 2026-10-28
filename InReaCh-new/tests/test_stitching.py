"""原图融合的几何、常量和缺失观测测试。"""
import unittest
import numpy as np
from contracts import TileMeta, TileScores
from stitching import stitch_scores


class StitchingTests(unittest.TestCase):
    def test_constant_shifted_overlap_covers_rectangular_original(self):
        tiles = [TileScores(TileMeta('a', i, (x, 0), (8, 13), (8, 8), (0, 0, 0, 0)),
                            np.full((2, 4), 7.), np.ones((2, 4), dtype=bool))
                 for i, x in enumerate((0, 5))]
        out = stitch_scores(tiles, (8, 13))
        self.assertTrue(out.valid_map.all())
        np.testing.assert_allclose(out.score_map, 7.)
        self.assertEqual(out.image_id, 'a')
        self.assertAlmostEqual(out.image_score, 7.)
        self.assertEqual(out.coverage, 1.)

    def test_interpolation_matches_cell_center_coordinates(self):
        meta = TileMeta('a', 0, (0, 0), (8, 8), (8, 8), (0, 0, 0, 0))
        out = stitch_scores([TileScores(meta, np.array([[0., 8.]]), np.array([[True, True]]))], (8, 8))
        np.testing.assert_allclose(out.score_map[0], [0, 0, 1, 3, 5, 7, 8, 8])

    def test_invalid_values_do_not_pollute_and_padding_is_cropped(self):
        meta = TileMeta('a', 0, (0, 0), (3, 5), (3, 5), (0, 0, 3, 5))
        values = np.full((4, 4), np.nan)
        values[0, :2] = 7.
        valid = np.zeros((4, 4), dtype=bool)
        valid[0, :2] = True
        out = stitch_scores([TileScores(meta, values, valid)], (3, 5))
        self.assertEqual(out.score_map.shape, (3, 5))
        self.assertTrue(out.valid_map.all())
        np.testing.assert_allclose(out.score_map, 7.)

    def test_uncovered_area_is_nan_and_mixed_images_are_rejected(self):
        meta = TileMeta('a', 0, (0, 0), (8, 8), (8, 8), (0, 0, 0, 0))
        tile = TileScores(meta, np.array([[7., np.nan]]), np.array([[True, False]]))
        out = stitch_scores([tile], (8, 8))
        self.assertTrue(out.valid_map[:, :6].all())
        self.assertFalse(out.valid_map[:, 6:].any())
        self.assertTrue(np.isnan(out.score_map[:, 6:]).all())
        other = TileScores(TileMeta('b', 1, (0, 0), (8, 8), (8, 8), (0, 0, 0, 0)),
                           np.ones((1, 1)), np.ones((1, 1), dtype=bool))
        with self.assertRaises(ValueError):
            stitch_scores([tile, other], (8, 8))
        with self.assertRaises(ValueError):
            stitch_scores([tile, tile], (8, 8))


if __name__ == '__main__': unittest.main()
