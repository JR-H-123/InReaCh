"""原图框监督定位指标; 峰值用像素中心(x+.5,y+.5)，不生成伪像素真值。"""
from collections import deque
import math
import numpy as np
from contracts import Box, Prediction
from localization import extract_regions, validate_prediction


def evaluate_predictions(predictions: dict[str, Prediction],
                         boxes: dict[str, tuple[Box, ...] | None],
                         threshold: float, *, annotations_complete: bool = False,
                         require_full_coverage: bool = True) -> dict[str, float | None]:
    """原图分数/真实框/验证集确定的阈值 -> 框召回与未匹配区域等指标.

    使用 8 连通区域峰值落框的一对一最大匹配. pic1 全部含缺陷,
    无法计算正常/异常图像级 AUROC; 未知或不完整标注不能作为确定误报依据.
    """
    if not predictions or set(predictions) != set(boxes) or not math.isfinite(threshold):
        raise ValueError('需要非空且原图身份一致的预测/标注和有限阈值')
    matched = box_count = regions = known = unknown = 0
    anomalous = valid_pixels = all_pixels = 0
    image_scores, image_labels = [], []
    for image_id, p in predictions.items():
        validate_prediction(p, image_id)
        if require_full_coverage and not p.valid_map.all():
            raise ValueError(f'{image_id}: 有效预测覆盖不足，不能报告完整图像指标')
        mask = p.valid_map & (p.score_map >= threshold)
        anomalous += int(mask.sum())
        valid_pixels += int(p.valid_map.sum())
        all_pixels += p.score_map.size
        truth = boxes[image_id]
        # 未确认完整标注时，空文件不能作为已知正常图。
        if truth is None or (len(truth) == 0 and not annotations_complete):
            unknown += 1
            continue
        h, w = p.score_map.shape
        for b in truth:
            if (b.class_id < 0 or not np.isfinite([b.x1, b.y1, b.x2, b.y2]).all() or
                not (0 <= b.x1 < b.x2 <= w and 0 <= b.y1 < b.y2 <= h)):
                raise ValueError(f'{image_id}: 原图标注框无效')
        peaks = [region['peak_xy'] for region in extract_regions(p, threshold)]
        candidates = [[j for j, b in enumerate(truth) if b.x1 <= x < b.x2 and b.y1 <= y < b.y2]
                      for x, y in peaks]
        matched += _maximum_matches(candidates)
        box_count += len(truth)
        regions += len(peaks)
        known += 1
        if p.valid_map.all():
            image_scores.append(p.image_score)
            image_labels.append(bool(truth))
    auc = None
    if annotations_complete and len(image_scores) == known:
        positives = np.array([s for s, label in zip(image_scores, image_labels) if label])
        negatives = np.sort([s for s, label in zip(image_scores, image_labels) if not label])
        if len(positives) and len(negatives):
            lower = np.searchsorted(negatives, positives, side='left')
            upper = np.searchsorted(negatives, positives, side='right')
            auc = float(np.sum((lower+upper)/2) / (len(positives)*len(negatives)))
    return {'matched_boxes': matched, 'total_boxes': box_count, 'predicted_regions': regions,
            'box_recall': matched/box_count if box_count else None,
            'unmatched_regions_per_image': (regions-matched)/known if known else None,
            'anomaly_area_fraction': anomalous/all_pixels, 'valid_coverage': valid_pixels/all_pixels,
            'known_label_images': known, 'unknown_label_images': unknown, 'image_auroc': auc}


def _maximum_matches(candidates: list[list[int]]) -> int:
    """增广路径最大基数匹配；简单贪心会漏掉重叠框可重分配的匹配。"""
    owner, assigned = {}, {}
    for root in range(len(candidates)):
        queue, seen, parent = deque([root]), {root}, {}
        free = None
        while queue and free is None:
            region = queue.popleft()
            for box in candidates[region]:
                if box in parent:
                    continue
                parent[box] = region
                if box not in owner:
                    free = box
                    break
                previous = owner[box]
                if previous not in seen:
                    seen.add(previous)
                    queue.append(previous)
        while free is not None:
            region = parent[free]
            old = assigned.get(region)
            owner[free] = region
            assigned[region] = free
            free = old
    return len(assigned)


def select_threshold(predictions: dict[str, Prediction], boxes: dict[str, tuple[Box, ...] | None],
                     candidates: list[float], *, max_unmatched_regions_per_image: float,
                     annotations_complete: bool) -> dict:
    """仅验证集：在显式未匹配区域预算内最大化框召回；返回阈值与完整曲线。

    平局先选更小异常面积，再选更高阈值。不在测试集重新调参。
    完整标注、全覆盖、至少一个缺陷框是这一阈值目标的前提。
    """
    if (not annotations_complete or any(b is None for b in boxes.values()) or not candidates or
        not math.isfinite(max_unmatched_regions_per_image) or max_unmatched_regions_per_image < 0):
        raise ValueError('阈值选择需要完整已知验证标注、候选阈值和非负有限区域预算')
    curve = []
    for threshold in sorted(set(candidates)):
        metrics = evaluate_predictions(predictions, boxes, threshold, annotations_complete=True)
        if metrics['box_recall'] is None:
            raise ValueError('验证集至少需要一个缺陷框才能优化框召回')
        curve.append({'threshold': float(threshold), **metrics})
    eligible = [row for row in curve if row['unmatched_regions_per_image'] <= max_unmatched_regions_per_image]
    if not eligible:
        raise ValueError('候选阈值没有满足验证集未匹配区域预算的结果')
    best = max(eligible, key=lambda row: (row['box_recall'], -row['anomaly_area_fraction'], row['threshold']))
    return {'threshold': best['threshold'], 'curve': curve,
            'objective': 'max_box_recall_under_unmatched_region_budget',
            'max_unmatched_regions_per_image': max_unmatched_regions_per_image}
