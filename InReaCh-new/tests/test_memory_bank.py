"""手算球面离散度与缓存失配测试。"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from config import ExperimentConfig
from contracts import Channel, ChannelMember
import memory_bank


def channel(values):
    return Channel(tuple(ChannelMember(str(i), 0, torch.tensor([float(v)]), (2., 2.), (8, 8))
                         for i, v in enumerate(values)))


class MemoryBankTests(unittest.TestCase):
    def test_spherical_std_removes_outlier_and_preserves_sources(self):
        bank = memory_bank.build_memory_bank([channel([0, 0, 0, 10])],
                ExperimentConfig(min_channel_length=2, max_channel_std=2))
        self.assertEqual(bank.features.flatten().tolist(), [0, 0, 0])
        self.assertEqual(bank.image_ids, ('0', '1', '2'))
        np.testing.assert_array_equal(bank.xy, [[2, 2]]*3)

    def test_strict_length_and_identical_zero_dispersion(self):
        cfg = ExperimentConfig(min_channel_length=2)
        with self.assertRaisesRegex(ValueError, '空|合格'):
            memory_bank.build_memory_bank([channel([1, 1])], cfg)
        bank = memory_bank.build_memory_bank([channel([1, 1, 1])], cfg)
        self.assertEqual(len(bank.features), 3)
        # 距中心等距但互不相同的点不属于完全一致的零半径例外。
        with self.assertRaises(ValueError):
            memory_bank.build_memory_bank([channel([0, 2])], ExperimentConfig(min_channel_length=1))

    def test_rejects_empty_and_duplicate_image_support(self):
        with self.assertRaises(ValueError):
            memory_bank.build_memory_bank([], ExperimentConfig())
        ch = channel([0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, '原图|重复'):
            memory_bank.build_memory_bank([Channel(ch.members + ch.members[:1])], ExperimentConfig())

    def test_cache_roundtrip_and_configuration_or_data_mismatch(self):
        cfg = ExperimentConfig(min_channel_length=1)
        bank = memory_bank.build_memory_bank([channel([1, 1, 1])], cfg)
        bank.provenance = {'reference_images': [{'image_id': '0', 'sha256': 'abc'}],
                           'model_sha256': 'weights-v1'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bank.npz'
            memory_bank.save_memory_bank(bank, path)
            restored = memory_bank.load_memory_bank(path, cfg, bank.provenance)
            torch.testing.assert_close(restored.features, bank.features)
            self.assertEqual(restored.image_ids, bank.image_ids)
            self.assertEqual(restored.provenance, bank.provenance)
            with self.assertRaisesRegex(ValueError, '配置'):
                memory_bank.load_memory_bank(path, ExperimentConfig(min_channel_length=1, radius_px=4), bank.provenance)
            with self.assertRaisesRegex(ValueError, '来源|签名'):
                memory_bank.load_memory_bank(path, cfg, dict(bank.provenance, model_sha256='changed'))


if __name__ == '__main__': unittest.main()
