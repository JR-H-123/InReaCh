"""验证论文尺寸处理和 RGB 模型输入, 不依赖具体数据集目录."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from image_processing import load_image, preprocess_image, to_model_input


class ImageProcessingTests(unittest.TestCase):
    def test_resize_then_center_crop_keeps_expected_pixels(self):
        """检查 256x256 图像中心裁剪为 224x224, 避免裁剪位置写错."""
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[16, 16] = [10, 20, 30]
        image[239, 239] = [40, 50, 60]
        image[0, 0] = [70, 80, 90]

        result = preprocess_image(image)

        self.assertEqual(result.shape, (224, 224, 3))
        np.testing.assert_array_equal(result[0, 0], [10, 20, 30])
        np.testing.assert_array_equal(result[-1, -1], [40, 50, 60])

    def test_tensor_shape_and_rgb_imagenet_normalization(self):
        """检查 HWC->NCHW 和预训练权重要求的 RGB 归一化."""
        image = np.array([[[0, 128, 255]]], dtype=np.uint8)

        result = to_model_input(image)

        self.assertEqual(tuple(result.shape), (1, 3, 1, 1))
        np.testing.assert_allclose(
            result.numpy().reshape(3),
            [(0 / 255 - 0.485) / 0.229,
             (128 / 255 - 0.456) / 0.224,
             (255 / 255 - 0.406) / 0.225],
            rtol=1e-6,
        )

    def test_load_image_reads_a_path(self):
        """检查 OpenCV 读取的 BGR 图像被转换为 RGB."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bmp"
            image = np.full((3, 5, 3), [1, 2, 3], dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), image))

            result = load_image(path)

            np.testing.assert_array_equal(result, image[:, :, ::-1])


if __name__ == "__main__":
    unittest.main()
