"""有界内存的精确欧氏最近邻; 空间门控采用原图浮点坐标。"""
from collections import defaultdict
import numpy as np
import torch


def nearest_neighbors(query: torch.Tensor, target: torch.Tensor,
                      query_xy: np.ndarray, target_xy: np.ndarray,
                      radius: float | None, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """有效的CPU特征行 -> 最近目标行下标与欧氏距离; 无候选为(-1,inf)。

    半径非空时先按坐标分桶，只比较相邻桶，最后仍用精确两轴门控。
    每个距离矩阵最多chunk_size²，不构造整图两两距离或(P,Q,D)数组。
    目标行升序、只在距离严格变小时更新，保证分块大小不改变平局结果。
    """
    indices = np.full(len(query), -1, dtype=np.int64)
    distances = np.full(len(query), np.inf, dtype=np.float64)
    if not len(query) or not len(target):
        return indices, distances
    if radius is None:
        groups = [(np.arange(len(query)), np.arange(len(target)))]
    else:
        qbins, tbins = defaultdict(list), defaultdict(list)
        qkeys = query_xy if radius == 0 else np.floor(query_xy / radius)
        tkeys = target_xy if radius == 0 else np.floor(target_xy / radius)
        for i, key in enumerate(qkeys):
            qbins[tuple(key)].append(i)
        for i, key in enumerate(tkeys):
            tbins[tuple(key)].append(i)
        def spatial_groups():
            for (x, y), qrows in qbins.items():
                offsets = [(0, 0)] if radius == 0 else [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
                candidates = sorted(i for dx, dy in offsets for i in tbins.get((x+dx, y+dy), ()))
                if candidates:
                    yield np.asarray(qrows), np.asarray(candidates)
        groups = spatial_groups()
    with torch.inference_mode():
        for qrows, trows in groups:
            for start in range(0, len(qrows), chunk_size):
                qi = qrows[start:start+chunk_size]
                for offset in range(0, len(trows), chunk_size):
                    ti = trows[offset:offset+chunk_size]
                    # 直接L2避免矩阵乘法近零消减误差及随分块阈值改变计算路径。
                    d = torch.cdist(query[qi].float(), target[ti].float(),
                                    compute_mode='donot_use_mm_for_euclid_dist').numpy()
                    if radius is not None:
                        allowed = np.all(np.abs(query_xy[qi, None] - target_xy[ti]) <= radius, axis=2)
                        d[~allowed] = np.inf
                    arg = d.argmin(axis=1)
                    best = d[np.arange(len(qi)), arg]
                    take = best < distances[qi]
                    distances[qi[take]] = best[take]
                    indices[qi[take]] = ti[arg[take]]
    return indices, distances
