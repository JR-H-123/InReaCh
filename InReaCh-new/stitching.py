"""待实现: 将未经块级归一化的分数加权还原到原图."""
from contracts import Prediction, TileScores


def stitch_scores(scores: list[TileScores], original_hw: tuple[int, int]) -> Prediction:
    """块特征网格分数 -> 原图 (H,W) 分数/有效性; 使用真实坐标和非零边缘权重.

    补边不参与融合, 无有效权重区域保留 invalid; 默认不额外平滑.
    """
    raise NotImplementedError('尚未实现热图插值、加权融合与覆盖率检查')
