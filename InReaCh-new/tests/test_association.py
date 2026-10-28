"""人工描述子验证跨原图互近邻及空间约束。"""
import unittest
import numpy as np
import torch
from config import ExperimentConfig
from contracts import DescriptorSet
from association import associate_channels


def group(image_id, values, xy, hw=(32, 32)):
    n = len(values)
    return DescriptorSet(torch.tensor(values, dtype=torch.float32).reshape(n, -1),
                         np.array(xy, dtype=np.float64), image_id,
                         np.zeros(n, dtype=np.int64), {0: (1, n)},
                         np.ones(n, dtype=np.bool_), original_hw=hw)


class AssociationTests(unittest.TestCase):
    def test_mutual_nearest_builds_channels_by_image(self):
        a = group('a', [0, 10], [(2, 2), (20, 2)])
        b = group('b', [.1, 9.9], [(3, 2), (21, 2)])
        c = group('c', [.2, 10.2], [(2, 3), (20, 3)])
        channels = associate_channels([a, b, c], ExperimentConfig(assoc_depth=1, radius_px=2))
        self.assertEqual([[m.image_id for m in ch.members] for ch in channels],
                         [['a', 'b', 'c'], ['a', 'b', 'c']])
        self.assertEqual([[m.descriptor_index for m in ch.members] for ch in channels],
                         [[0, 0, 0], [1, 1, 1]])

    def test_non_mutual_and_empty_spatial_candidates_do_not_match(self):
        a = group('a', [0], [(0, 0)])
        b = group('b', [.1, .2], [(0, 0), (0, 0)])
        channels = associate_channels([a, b], ExperimentConfig(assoc_depth=1, radius_px=0))
        self.assertEqual([m.descriptor_index for m in channels[0].members], [0, 0])
        self.assertEqual(associate_channels([a, group('far', [.1], [(8, 0)])],
                                            ExperimentConfig(assoc_depth=1, radius_px=2)), [])

    def test_target_chooses_best_seed_once(self):
        a = group('a', [0], [(0, 0)])
        b = group('b', [1], [(0, 0)])
        c = group('c', [.9], [(0, 0)])
        channels = associate_channels([a, b, c], ExperimentConfig(assoc_depth=2, radius_px=None))
        self.assertEqual([[m.image_id for m in ch.members] for ch in channels],
                         [['a', 'b'], ['b', 'c']])

    def test_tie_is_stable_across_chunks_and_invalid_rows_are_excluded(self):
        a = group('a', [0, 0, 10], [(0, 0), (0, 0), (2, 0)])
        b = group('b', [0, 10], [(0, 0), (2, 0)])
        b.valid[1] = False
        for chunk in (1, 8):
            cs = associate_channels([a, b], ExperimentConfig(assoc_depth=1, match_chunk_size=chunk))
            self.assertEqual([[m.descriptor_index for m in c.members] for c in cs], [[0, 0]])

    def test_rejects_duplicate_image_dimensions_and_coordinate_system(self):
        a = group('a', [0], [(0, 0)])
        with self.assertRaisesRegex(ValueError, '重复'):
            associate_channels([a, a], ExperimentConfig())
        with self.assertRaisesRegex(ValueError, '维度'):
            associate_channels([a, group('b', [[0, 1]], [(0, 0)])], ExperimentConfig())
        with self.assertRaisesRegex(ValueError, '尺寸'):
            associate_channels([a, group('b', [0], [(0, 0)], hw=(33, 32))], ExperimentConfig())
        with self.assertRaises(ValueError):
            associate_channels([a], ExperimentConfig())


if __name__ == '__main__':
    unittest.main()
