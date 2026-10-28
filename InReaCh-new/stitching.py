"""有效观测的双线性插值与正中心权重融合，不做块级分数归一化。"""
import numpy as np
import torch
import torch.nn.functional as F
from contracts import Prediction, TileScores


def stitch_scores(scores: list[TileScores], original_hw: tuple[int, int]) -> Prediction:
    """块特征网格分数 -> 原图 (H,W) 分数/有效性; 使用真实坐标和非零边缘权重.

    补边不参与融合, 无有效权重区域保留 invalid; 默认不额外平滑.
    """
    if not scores or len(original_hw) != 2 or min(original_hw) <= 0:
        raise ValueError('需要非空块分数和有效原图尺寸')
    image_ids = {s.meta.image_id for s in scores}
    if len(image_ids) != 1:
        raise ValueError('不能融合不同原图的图块')
    if len({s.meta.tile_id for s in scores}) != len(scores):
        raise ValueError('重复图块不能重复贡献融合权重')
    accumulated = np.zeros(original_hw, dtype=np.float64)
    weight_sum = np.zeros(original_hw, dtype=np.float64)
    for tile in scores:
        meta = tile.meta
        if (tile.scores.ndim != 2 or min(tile.scores.shape) <= 0 or
            tile.valid.shape != tile.scores.shape or tile.valid.dtype != np.bool_ or
            not np.isfinite(tile.scores[tile.valid]).all()):
            raise ValueError('块分数网格、有效性或有限数值无效')
        x, y = meta.origin_xy
        vh, vw = meta.valid_hw
        left, top, right, bottom = meta.padding
        if (meta.original_hw != original_hw or min(x, y) < 0 or min(vh, vw) <= 0 or
            min(right, bottom) < 0 or left != 0 or top != 0 or
            x+vw > original_hw[1] or y+vh > original_hw[0]):
            raise ValueError('图块原图尺寸、起点或补边无效')
        h, w = vh+bottom, vw+right
        gh, gw = tile.scores.shape
        inside = (((np.arange(gh)+.5)*h/gh < vh)[:, None] &
                  ((np.arange(gw)+.5)*w/gw < vw)[None, :])
        if np.any(tile.valid & ~inside):
            raise ValueError('补边网格不能作为有效融合证据')
        # 先清除无效NaN，再插值分子与支持权重；NaN*0本身仍是NaN。
        planes = np.stack((np.where(tile.valid, tile.scores, 0.), tile.valid.astype(np.float64)))
        interpolated = F.interpolate(torch.from_numpy(planes)[None], size=(h, w),
                                     mode='bilinear', align_corners=False)[0].numpy()
        # 像素中心采样的三角窗边缘也严格大于0，不制造原图边缘覆盖孔。
        wy = 1 - np.abs(2*(np.arange(h)+.5)/h-1)
        wx = 1 - np.abs(2*(np.arange(w)+.5)/w-1)
        window = (wy[:, None]*wx[None, :])[:vh, :vw]
        accumulated[y:y+vh, x:x+vw] += window * interpolated[0, :vh, :vw]
        weight_sum[y:y+vh, x:x+vw] += window * interpolated[1, :vh, :vw]
    valid = weight_sum > 0
    result = np.full(original_hw, np.nan, dtype=np.float64)
    np.divide(accumulated, weight_sum, out=result, where=valid)
    return Prediction(result, valid, next(iter(image_ids)))
