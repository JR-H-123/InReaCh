"""从原图异常分数提取区域；评估与导出共用同一定位规则。"""
import math
import cv2
import numpy as np
from contracts import Prediction


def validate_prediction(prediction: Prediction, image_id: str | None = None) -> None:
    """检查网格、有效分数与可选原图身份；无效位置允许保存NaN。"""
    if (prediction.score_map.ndim != 2 or min(prediction.score_map.shape) <= 0 or
        prediction.valid_map.shape != prediction.score_map.shape or
        prediction.valid_map.dtype != np.bool_ or
        not np.isfinite(prediction.score_map[prediction.valid_map]).all() or
        (image_id is not None and prediction.image_id is not None and prediction.image_id != image_id)):
        raise ValueError('预测原图身份、分数或有效性无效')


def extract_regions(prediction: Prediction, threshold: float) -> list[dict]:
    """原图分数/有效图/阈值 -> 8连通区域的峰值、半开外接框和面积。

    外接框描述阈值化区域，不是真实缺陷框。峰值分数保留原始距离，
    不解释成概率。有效覆盖不完整时仍可提取已有证据，不填补未知区域。
    """
    validate_prediction(prediction)
    if not math.isfinite(threshold):
        raise ValueError('定位阈值必须是有限数值')
    mask = prediction.valid_map & (prediction.score_map >= threshold)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    regions = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        local = labels[y:y+h, x:x+w] == label
        positions = np.flatnonzero(local)
        values = prediction.score_map[y:y+h, x:x+w].ravel()[positions]
        # positions行优先且argmax取首项，等高峰值时坐标决胜固定。
        index = int(values.argmax())
        row, col = np.unravel_index(positions[index], (h, w))
        regions.append({'peak_xy': [float(x+col+.5), float(y+row+.5)],
                        'peak_score': float(values[index]),
                        'bbox_xyxy': [x, y, x+w, y+h], 'area_px': area})
    regions.sort(key=lambda region: (region['peak_xy'][1], region['peak_xy'][0]))
    return [{'region_id': i, **region} for i, region in enumerate(regions, 1)]
