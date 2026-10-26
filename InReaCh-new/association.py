"""待实现: 不同原图之间建立互近邻通道, 不接收评估标签."""
from config import ExperimentConfig
from contracts import Channel, DescriptorSet


def associate_channels(descriptors: list[DescriptorSet], config: ExperimentConfig) -> list[Channel]:
    """输入按原图汇总的描述子, 输出候选通道.

    必须排除同 image_id 关联; 每张原图在每条通道最多贡献一个成员.
    radius_px 控制原图坐标门控, None 为无门控对照; 无候选不放宽约束.
    """
    raise NotImplementedError('尚未实现跨图候选搜索和互近邻通道')
