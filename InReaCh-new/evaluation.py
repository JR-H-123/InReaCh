"""待实现: 用原图 YOLO 框评估定位, 不把检测框冒充像素分割真值."""
from contracts import Box, Prediction


def evaluate_predictions(predictions: dict[str, Prediction],
                         boxes: dict[str, tuple[Box, ...] | None],
                         threshold: float) -> dict[str, float | None]:
    """原图分数/真实框/验证集确定的阈值 -> 框召回与未匹配区域等指标.

    首版拟用 8 连通区域峰值落框的一对一匹配. pic1 全部含缺陷,
    无法计算正常/异常图像级 AUROC; 未知或不完整标注不能作为确定误报依据.
    """
    raise NotImplementedError('尚未实现框匹配、阈值选择和指标计算')
