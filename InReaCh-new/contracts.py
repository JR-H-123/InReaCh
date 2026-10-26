"""模块之间的数据契约; 坐标为 (x,y), 尺寸为 (H,W), 不用 sqrt 推断网格."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    image_path: Path
    label_path: Path | None  # None 表示没有找到标注, 不能认定为正常图.


@dataclass(frozen=True)
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class TileMeta:
    image_id: str
    tile_id: int
    origin_xy: tuple[int, int]
    original_hw: tuple[int, int]
    valid_hw: tuple[int, int]
    padding: tuple[int, int, int, int]  # left, top, right, bottom


@dataclass
class Tile:
    image: np.ndarray  # RGB uint8, (tile_size,tile_size,3)
    meta: TileMeta


@dataclass
class TileBatch:
    inputs: torch.Tensor  # (B,3,tile_size,tile_size), 已完成 RGB 归一化
    metadata: tuple[TileMeta, ...]


@dataclass
class DescriptorSet:
    """单原图的描述子; P 可跨多个块, grid_hw 是各块的真实特征网格."""
    features: torch.Tensor  # (P,D)
    xy: np.ndarray  # (P,2), 原图浮点坐标
    image_id: str
    tile_ids: np.ndarray  # (P,)
    grid_hw: dict[int, tuple[int, int]]
    valid: np.ndarray  # (P,), bool; 补边和无效项不得参与建库


@dataclass(frozen=True)
class ChannelMember:
    image_id: str
    descriptor_index: int
    feature: torch.Tensor  # (D,)
    xy: tuple[float, float]


@dataclass
class Channel:
    members: tuple[ChannelMember, ...]  # 同一通道每张原图最多一项


@dataclass
class MemoryBank:
    features: torch.Tensor  # (K,D)
    xy: np.ndarray  # (K,2)


@dataclass
class TileScores:
    meta: TileMeta
    scores: np.ndarray  # (Hf,Wf), 未做每块 min-max 归一化
    valid: np.ndarray  # (Hf,Wf), bool


@dataclass
class Prediction:
    score_map: np.ndarray  # (H,W), 原图大小
    valid_map: np.ndarray  # (H,W), bool; 无候选不能当正常分数
