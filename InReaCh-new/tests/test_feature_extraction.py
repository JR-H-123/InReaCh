"""验证原版骨干网络的特征节点和空间尺寸."""

import unittest

import torch

from feature_extraction import extract_features, load_feature_model
from pipeline import _model_digest


class FeatureExtractionTests(unittest.TestCase):
    """检查单张 224x224 图像在原版 13 个节点上的输出."""

    def test_original_nodes_and_feature_shapes(self):
        """用途: 防止骨干、节点或推理模式偏离原版."""
        model = load_feature_model(torch.device("cpu"))
        image = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
        features = extract_features(model, image)

        self.assertFalse(model.training)
        self.assertEqual(list(features), [f"Level_{index}" for index in range(1, 14)])
        self.assertEqual(
            [tuple(feature.shape) for feature in features.values()],
            [(1, 256, 56, 56)] * 3
            + [(1, 512, 28, 28)] * 4
            + [(1, 1024, 14, 14)] * 5
            + [(1, 2048, 7, 7)],
        )
        self.assertTrue(all(not feature.requires_grad for feature in features.values()))
        # 默认工厂已提供前向配置，仍需散列真实权重，不能只凭配置名兼容。
        before = _model_digest(model)
        with torch.no_grad():
            next(model.parameters()).view(-1)[0] += 1.
        self.assertNotEqual(before, _model_digest(model))


if __name__ == "__main__":
    unittest.main()
