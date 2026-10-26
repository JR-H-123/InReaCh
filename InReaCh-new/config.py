"""实验配置: 将切块几何与计算批大小分开, 避免为了省显存缩小缺陷."""
from dataclasses import dataclass
from pathlib import Path
import math


@dataclass(frozen=True)
class ExperimentConfig:
    """输入配置; radius_px=None 表示未来关联阶段不使用空间门控."""
    data_root: Path = Path(__file__).resolve().parent / 'data' / 'pic1'
    tile_size: int = 224
    stride: int = 112
    batch_size: int = 4
    radius_px: float | None = 8.0

    def __post_init__(self):
        object.__setattr__(self, 'data_root', Path(self.data_root))
        for name in ('tile_size', 'stride', 'batch_size'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} 必须是正整数')
        if self.stride > self.tile_size:
            raise ValueError('stride 不能大于 tile_size, 否则原图会出现未覆盖区域')
        if self.radius_px is not None and (not math.isfinite(self.radius_px) or self.radius_px < 0):
            raise ValueError('radius_px 必须是非负有限数值或 None')
