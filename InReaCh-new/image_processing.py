"""按 InReaCh 论文的图像尺寸流程准备模型输入."""

from pathlib import Path

import cv2
import numpy as np
import torch


RESIZE_SIZE = 256
"""图像缩放后的正方形边长, 单位为像素; 即缩放到 256x256."""

CROP_SIZE = 224
"""中心裁剪后的正方形边长, 单位为像素; 即裁剪到 224x224."""


def load_image(image_path: str | Path) -> np.ndarray:
    """用途: 从路径读取一张彩色图像.

    参数: image_path (图像路径, str | pathlib.Path).
    返回: 图像数组 (np.ndarray), 形状为 (高, 宽, 3), 通道顺序为 RGB.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    # OpenCV 读取为 BGR, 交换红蓝通道后交给预训练模型.
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """用途: 将图像缩放到 256x256, 再从中心裁剪为 224x224.

    参数: image (原始 RGB 图像, np.ndarray), 形状为 (高, 宽, 3).
    返回: 裁剪后的 RGB 图像 (np.ndarray), 形状为 (224, 224, 3).
    """
    # NumPy 图像按（高，宽，通道）排列；OpenCV 的 resize 参数按（宽，高）排列。
    resized = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_LINEAR)
    offset = (RESIZE_SIZE - CROP_SIZE) // 2
    # 高和宽使用相同的中心偏移，与论文中的方形中心裁剪一致。
    return resized[offset:offset + CROP_SIZE, offset:offset + CROP_SIZE]


def to_model_input(image: np.ndarray) -> torch.Tensor:
    """用途: 按预训练权重的通道参数, 把 RGB 图像转换为模型输入.

    参数: image (预处理后的 RGB 图像, np.ndarray), 形状为 (高, 宽, 3).
    返回: 模型输入 (torch.Tensor), float32, 形状为 (1, 3, 高, 宽).
    """
    # 按 RGB 顺序使用 ImageNet 均值与标准差, 对应当前预训练权重.
    values = image / 255.0
    values = (values - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    # HWC 转为 CHW，再增加批量维度 N。
    values = np.transpose(values, (2, 0, 1))[None, ...]
    return torch.tensor(values).float()
