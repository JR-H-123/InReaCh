"""按与建库一致的原图空间规则和欧氏距离计算异常分数。"""
import numpy as np
from config import ExperimentConfig
from contracts import DescriptorSet, MemoryBank, TileMeta, TileScores
from descriptors import _check_descriptor_set
from matching import nearest_neighbors
from memory_bank import validate_memory_bank


def predict_tiles(descriptors: list[DescriptorSet], bank: MemoryBank,
                  metadata: tuple[TileMeta, ...], config: ExperimentConfig) -> list[TileScores]:
    """测试描述子与参考库 -> 各块特征网格分数及有效性, 无候选不得返回正常零分.

    推理描述子保留全部有效重叠观测, 建库位置归属应作为独立后续步骤.
    """
    validate_memory_bank(bank, config)
    if not descriptors or not metadata:
        raise ValueError('预测描述子和完整图块元数据不能为空')
    metas = {(m.image_id, m.tile_id): m for m in metadata}
    if len(metas) != len(metadata):
        raise ValueError('重复图块元数据')
    if len({d.image_id for d in descriptors}) != len(descriptors):
        raise ValueError('重复原图，请先跨批汇总')
    expected = {(d.image_id, t) for d in descriptors for t in d.grid_hw}
    if expected != set(metas):
        raise ValueError('图块元数据与描述子不一致')
    outputs = {}
    for d in descriptors:
        _check_descriptor_set(d)  # 归属筛选后的稀疏行不能用于网格恢复。
        if d.features.shape[1] != bank.features.shape[1]:
            raise ValueError('预测与参考库特征维度不一致')
        if config.radius_px is not None and d.original_hw != bank.source_hw[0]:
            raise ValueError('空间预测与参考库原图尺寸不一致')
        rows = np.flatnonzero(d.valid)
        indices, distances = nearest_neighbors(d.features[rows], bank.features, d.xy[rows], bank.xy,
                                               config.radius_px, config.match_chunk_size)
        valid = np.zeros(len(d.features), dtype=np.bool_)
        scores = np.full(len(d.features), np.nan, dtype=np.float64)
        found = (indices >= 0) & np.isfinite(distances)
        valid[rows[found]] = True
        scores[rows[found]] = distances[found]
        for tile_id, (h, w) in d.grid_hw.items():
            meta = metas[(d.image_id, tile_id)]
            if meta.original_hw != d.original_hw:
                raise ValueError('描述子与图块元数据的原图尺寸不一致')
            x, y = meta.origin_xy
            vh, vw = meta.valid_hw
            if (min(x, y) < 0 or min(vh, vw) <= 0 or max(vh, vw) > config.tile_size or
                y+vh > meta.original_hw[0] or x+vw > meta.original_hw[1] or
                meta.padding != (0, 0, config.tile_size-vw, config.tile_size-vh)):
                raise ValueError('图块几何或补边无效')
            mask = d.tile_ids == tile_id
            gx, gy = np.meshgrid((np.arange(w)+.5)*config.tile_size/w,
                                 (np.arange(h)+.5)*config.tile_size/h)
            expected_xy = np.column_stack((gx.ravel()+x, gy.ravel()+y))
            if not np.allclose(d.xy[mask], expected_xy, rtol=0, atol=1e-9):
                raise ValueError('描述子坐标或行顺序与图块网格不一致')
            inside = (gx.ravel() < vw) & (gy.ravel() < vh)
            if np.any(d.valid[mask] & ~inside):
                raise ValueError('补边描述子不能标为有效')
            outputs[(d.image_id, tile_id)] = TileScores(meta, scores[mask].reshape(h, w),
                                                       valid[mask].reshape(h, w))
    return [outputs[(m.image_id, m.tile_id)] for m in metadata]
