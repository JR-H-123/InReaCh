"""在原图尺度切块: 保留细小缺陷像素, 并为热图还原保留坐标."""
from collections.abc import Iterator
import numpy as np
from config import ExperimentConfig
from contracts import Tile, TileMeta


def _starts(length: int, size: int, stride: int) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length-size+1, stride))
    if starts[-1] != length-size:
        starts.append(length-size)  # 末块贴边; 起点不一定是 stride 的整数倍.
    return starts


def iter_tiles(image: np.ndarray, image_id: str, config: ExperimentConfig) -> Iterator[Tile]:
    """输入 RGB uint8 (H,W,3), 逐块输出原尺度像素与 TileMeta, 不修改原图."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) <= 0:
        raise ValueError('需要非空 RGB uint8 图像 (H,W,3)')
    h, w = image.shape[:2]
    size = config.tile_size
    tile_id = 0
    for y in _starts(h,size,config.stride):
        for x in _starts(w,size,config.stride):
            crop = image[y:y+size,x:x+size]
            vh, vw = crop.shape[:2]
            right, bottom = size-vw, size-vh
            # 只在小图上复制边缘补齐; valid_hw 让后续排除补边描述子.
            padded = np.pad(crop,((0,bottom),(0,right),(0,0)),mode='edge')
            yield Tile(padded,TileMeta(image_id,tile_id,(x,y),(h,w),(vh,vw),(0,0,right,bottom)))
            tile_id += 1
