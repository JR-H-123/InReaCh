"""验证距离单位、原图坐标约束与完整网格恢复。"""
import unittest
import numpy as np
import torch
from config import ExperimentConfig
from contracts import Channel, ChannelMember, DescriptorSet, TileMeta
from memory_bank import build_memory_bank
from prediction import predict_tiles


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ExperimentConfig(tile_size=8, stride=4, min_channel_length=1, radius_px=0)
        self.bank = build_memory_bank([Channel(tuple(
            ChannelMember(i, 0, torch.zeros(2), (2., 4.), (8, 13)) for i in ('a', 'b')))], self.cfg)
        self.d = DescriptorSet(torch.tensor([[3., 4.], [6., 8.]]),
                               np.array([[2., 4.], [6., 4.]]), 'test', np.array([0, 0]),
                               {0: (1, 2)}, np.array([True, True]), (8, 13))
        self.meta = (TileMeta('test', 0, (0, 0), (8, 13), (8, 8), (0, 0, 0, 0)),)

    def test_euclidean_five_and_no_candidate_invalid(self):
        output = predict_tiles([self.d], self.bank, self.meta, self.cfg)[0]
        self.assertEqual(output.scores.shape, (1, 2))
        self.assertEqual(output.scores[0, 0], 5.)
        self.assertTrue(np.isnan(output.scores[0, 1]))
        np.testing.assert_array_equal(output.valid, [[True, False]])
        self.assertEqual(output.meta.image_id, 'test')

    def test_invalid_input_is_not_queried(self):
        self.d.valid[0] = False
        output = predict_tiles([self.d], self.bank, self.meta, self.cfg)[0]
        self.assertFalse(output.valid.any())
        self.assertTrue(np.isnan(output.scores).all())

    def test_rejects_sparse_rows_and_metadata_mismatch(self):
        sparse = DescriptorSet(self.d.features[:1], self.d.xy[:1], 'test', np.array([0]),
                               {0: (1, 2)}, np.array([True]), (8, 13))
        with self.assertRaises(ValueError):
            predict_tiles([sparse], self.bank, self.meta, self.cfg)
        wrong = (TileMeta('other', 0, (0, 0), (8, 13), (8, 8), (0, 0, 0, 0)),)
        with self.assertRaises(ValueError):
            predict_tiles([self.d], self.bank, wrong, self.cfg)
        self.d.xy[0, 0] += 1
        with self.assertRaises(ValueError):
            predict_tiles([self.d], self.bank, self.meta, self.cfg)

    def test_rejects_changed_query_geometry_or_bank_rule(self):
        cfg = ExperimentConfig(tile_size=8, stride=4, min_channel_length=1, radius_px=4)
        with self.assertRaisesRegex(ValueError, '配置'):
            predict_tiles([self.d], self.bank, self.meta, cfg)
        self.d.original_hw = (8, 14)
        with self.assertRaisesRegex(ValueError, '尺寸'):
            predict_tiles([self.d], self.bank, self.meta, self.cfg)


if __name__ == '__main__': unittest.main()
