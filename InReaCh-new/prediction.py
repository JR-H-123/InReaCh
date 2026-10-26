"""待实现: 按与建库一致的空间规则计算异常分数."""
from config import ExperimentConfig
from contracts import DescriptorSet, MemoryBank, TileMeta, TileScores


def predict_tiles(descriptors: list[DescriptorSet], bank: MemoryBank,
                  metadata: tuple[TileMeta, ...], config: ExperimentConfig) -> list[TileScores]:
    """测试描述子与参考库 -> 各块特征网格分数及有效性, 无候选不得返回正常零分.

    推理描述子保留全部有效重叠观测, 建库位置归属应作为独立后续步骤.
    """
    raise NotImplementedError('尚未实现正常参考检索和块级异常分数')
