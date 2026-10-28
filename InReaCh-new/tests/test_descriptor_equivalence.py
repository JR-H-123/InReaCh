"""小尺寸原公式作数值参考，性能由独立基准验证。"""
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from config import ExperimentConfig
from contracts import TileMeta
from descriptors import build_descriptors


def original_level(feature, target_hw):
    _, h, w = feature.shape
    patches = F.unfold(feature[None], kernel_size=3, padding=1).reshape(1, -1, h, w)
    if (h, w) != target_hw:
        patches = F.interpolate(patches, size=target_hw, mode='bilinear', align_corners=False)
    rows = patches[0].permute(1, 2, 0).reshape(-1, 1, patches.shape[1])
    return F.adaptive_avg_pool1d(rows, 1024).squeeze(1)


class DescriptorEquivalenceTests(unittest.TestCase):
    def test_original_formula_coordinates_and_validity_with_mixed_grids(self):
        cfg = ExperimentConfig(tile_size=8, stride=4)
        metadata = (TileMeta('edge', 0, (5, 0), (8, 13), (8, 8), (0, 0, 0, 0)),
                    TileMeta('tiny', 0, (0, 0), (3, 5), (3, 5), (0, 0, 3, 5)))
        generator = torch.Generator().manual_seed(101)
        for transpose in (False, True):
            with self.subTest(noncontiguous=transpose):
                maps = {'first': torch.randn(2, 4, 3, 5, generator=generator),
                        'middle': torch.randn(2, 5, 2, 3, generator=generator),
                        'deep': torch.randn(2, 8, 1, 2, generator=generator)}
                if transpose:
                    maps = {name: value.transpose(-1, -2) for name, value in maps.items()}
                hw = tuple(maps['first'].shape[-2:])
                actual = build_descriptors(maps, metadata, cfg)
                for batch, item in enumerate(actual):
                    levels = torch.stack([original_level(value[batch], hw) for value in maps.values()], dim=1)
                    expected = F.adaptive_avg_pool1d(levels.reshape(hw[0]*hw[1], 1, -1), 1024).squeeze(1)
                    torch.testing.assert_close(item.features, expected, atol=1e-6, rtol=1e-5)
                    self.assertEqual(item.grid_hw, {0: hw})
                    np.testing.assert_array_equal(item.tile_ids, np.zeros(hw[0]*hw[1]))
                self.assertTrue(actual[0].valid.all())
                self.assertEqual(actual[0].original_hw, (8, 13))
                if not transpose:
                    np.testing.assert_allclose(actual[0].xy[0], [5.8, 4/3])
                    np.testing.assert_allclose(actual[0].xy[-1], [12.2, 20/3])
                    np.testing.assert_array_equal(actual[1].valid, [True, True, True]+[False]*12)


if __name__ == '__main__': unittest.main()
