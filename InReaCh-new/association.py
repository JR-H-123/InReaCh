"""不同原图之间建立互近邻通道, 不接收评估标签。"""
import numpy as np
from config import ExperimentConfig
from contracts import Channel, ChannelMember, DescriptorSet
from descriptors import _check_descriptor_set
from matching import nearest_neighbors


def associate_channels(descriptors: list[DescriptorSet], config: ExperimentConfig) -> list[Channel]:
    """输入按原图汇总的描述子, 输出候选通道.

    必须排除同 image_id 关联; 每张原图在每条通道最多贡献一个成员.
    radius_px 控制原图坐标门控, None 为无门控对照; 无候选不放宽约束.
    """
    if len(descriptors) < 2:
        raise ValueError('关联至少需要两张独立原图')
    ids = [d.image_id for d in descriptors]
    if len(set(ids)) != len(ids):
        raise ValueError('重复 image_id; 必须先按原图汇总')
    for d in descriptors:
        _check_descriptor_set(d, full_grid=False)
    if len({d.features.shape[1] for d in descriptors}) != 1:
        raise ValueError('描述子维度不一致')
    if config.radius_px is not None:
        shapes = {d.original_hw for d in descriptors}
        if None in shapes or len(shapes) != 1:
            raise ValueError('空间门控要求已知且一致的原图尺寸和像素坐标系')
    rows = [np.flatnonzero(d.valid) for d in descriptors]
    channels = {}
    def member(image, row):
        d = descriptors[image]
        return ChannelMember(d.image_id, int(row), d.features[row].detach().clone(),
                             tuple(map(float, d.xy[row])), d.original_hw)
    # 清单顺序即种子顺序，沿用旧版 seed < target 的配对方向。
    for target_i in range(1, len(descriptors)):
        target = descriptors[target_i]
        tr = rows[target_i]
        winners = {}
        for seed_i in range(min(config.assoc_depth, target_i)):
            seed, sr = descriptors[seed_i], rows[seed_i]
            forward, distance = nearest_neighbors(target.features[tr], seed.features[sr],
                                                  target.xy[tr], seed.xy[sr],
                                                  config.radius_px, config.match_chunk_size)
            backward, _ = nearest_neighbors(seed.features[sr], target.features[tr],
                                             seed.xy[sr], target.xy[tr],
                                             config.radius_px, config.match_chunk_size)
            for j, k in enumerate(forward):
                if k < 0 or backward[k] != j:
                    continue
                candidate = (float(distance[j]), seed_i, int(sr[k]))
                target_row = int(tr[j])
                if target_row not in winners or candidate < winners[target_row]:
                    winners[target_row] = candidate
        for target_row, (_, seed_i, seed_row) in sorted(winners.items()):
            key = (seed_i, seed_row)
            channels.setdefault(key, [member(seed_i, seed_row)]).append(member(target_i, target_row))
    return [Channel(tuple(channels[key])) for key in sorted(channels)]
