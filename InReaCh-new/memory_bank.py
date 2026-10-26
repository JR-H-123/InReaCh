"""待实现: 筛选通道并建立正常参考库, 不以全部输入特征代替正常特征."""
from config import ExperimentConfig
from contracts import Channel, MemoryBank


def build_memory_bank(channels: list[Channel], config: ExperimentConfig) -> MemoryBank:
    """候选通道 -> 正常描述子与位置; 后续补充通道阈值配置、索引及缓存契约.

    依据独立原图支持数与通道离散程度筛选. 空结果应报错, 不能用全量特征回退.
    """
    raise NotImplementedError('尚未实现正常通道筛选、参考索引和缓存')
