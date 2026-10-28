"""实验配置: 将切块几何与计算批大小分开, 避免为了省显存缩小缺陷."""
from dataclasses import dataclass
from pathlib import Path
import math


@dataclass(frozen=True)
class ExperimentConfig:
    """输入与算法配置; radius_px=None 表示关联和预测不使用空间门控。"""
    data_root: Path = Path(__file__).resolve().parent / 'data' / 'pic1'
    tile_size: int = 224
    stride: int = 112
    batch_size: int = 4
    radius_px: float | None = 8.0
    assoc_depth: int = 10
    match_chunk_size: int = 256
    min_channel_length: int = 3  # 沿用旧版: 成员数必须严格大于此值。
    max_channel_std: float = 5.0
    reference_feature_limit_mb: float = 2048.0

    def __post_init__(self):
        object.__setattr__(self, 'data_root', Path(self.data_root))
        for name in ('tile_size', 'stride', 'batch_size', 'assoc_depth', 'match_chunk_size', 'min_channel_length'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} 必须是正整数')
        if self.stride > self.tile_size:
            raise ValueError('stride 不能大于 tile_size, 否则原图会出现未覆盖区域')
        if self.radius_px is not None and (not math.isfinite(self.radius_px) or self.radius_px < 0):
            raise ValueError('radius_px 必须是非负有限数值或 None')
        if not math.isfinite(self.max_channel_std) or self.max_channel_std <= 0:
            raise ValueError('max_channel_std 必须是正有限数值')
        if not math.isfinite(self.reference_feature_limit_mb) or self.reference_feature_limit_mb <= 0:
            raise ValueError('reference_feature_limit_mb 必须是正有限数值')


def algorithm_signature(config: ExperimentConfig) -> dict:
    """缓存/预测兼容性契约; 批大小与分块大小不改变数学结果，不纳入签名。"""
    return {'version': 1, 'tile_size': config.tile_size, 'stride': config.stride,
            'radius_px': config.radius_px, 'assoc_depth': config.assoc_depth,
            'min_channel_length': config.min_channel_length,
            'max_channel_std': config.max_channel_std, 'distance': 'euclidean_l2',
            'descriptor': 'patch3-pool1024-before-resize-pool1024-no-position-v2',
            'coordinates': 'original-xy-cell-centers-align_corners_false',
            'ownership': 'max-valid-edge-margin-smallest-tile-id'}
