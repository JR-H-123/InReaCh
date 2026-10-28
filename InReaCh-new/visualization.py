"""原图分数的PNG/JSON导出；颜色范围只影响显示，不改变检测分数。"""
import argparse
import json
import math
from pathlib import Path
import cv2
import numpy as np
from contracts import Prediction
from image_processing import load_image
from localization import extract_regions, validate_prediction


INVALID_RGB = (128, 128, 128)


def validate_display_settings(score_range: tuple[float, float] | None,
                              alpha: float) -> tuple[float, float] | None:
    """校验共享显示范围和叠加比例；None允许流程稍后从验证集确定范围。"""
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('叠加alpha必须是[0,1]内的有限数值')
    if score_range is None:
        return None
    if not isinstance(score_range, (tuple, list)) or len(score_range) != 2:
        raise ValueError('显示范围必须为(min,max)')
    low, high = map(float, score_range)
    if not all(math.isfinite(v) for v in (low, high, high-low)) or high <= low:
        raise ValueError('显示范围需要有限数值且max大于min')
    return low, high


def select_display_range(predictions: list[Prediction]) -> tuple[float, float]:
    """仅验证集有效分数 -> 共用的[0,max]颜色范围；无正最大值时用[0,1]。"""
    if not predictions:
        raise ValueError('需要验证预测才能确定共享显示范围')
    maximum = 0.
    for prediction in predictions:
        validate_prediction(prediction)
        if prediction.valid_map.any():
            maximum = max(maximum, float(prediction.score_map[prediction.valid_map].max()))
    return 0., maximum if maximum > 0 else 1.


def render_prediction(image: np.ndarray, prediction: Prediction, score_range: tuple[float, float],
                      *, alpha: float = .5) -> tuple[np.ndarray, np.ndarray]:
    """RGB原图、Prediction、固定色标 -> 原图尺寸RGB热力图/叠加图，不修改输入。"""
    validate_prediction(prediction)
    bounds = validate_display_settings(score_range, alpha)
    if bounds is None:
        raise ValueError('渲染需要明确显示范围')
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or
        image.shape[:2] != prediction.score_map.shape):
        raise ValueError('需要与原图分数尺寸一致的RGB uint8图像')
    low, high = bounds
    # 先裁剪显示副本并清除无效NaN；不在原始分数上原地归一化。
    display = np.where(prediction.valid_map, np.clip(prediction.score_map, low, high), low)
    levels = np.rint((display-low)/(high-low)*255).astype(np.uint8)
    heatmap = cv2.cvtColor(cv2.applyColorMap(levels, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    overlay = np.rint((1-alpha)*image.astype(float)+alpha*heatmap.astype(float)).astype(np.uint8)
    heatmap[~prediction.valid_map] = INVALID_RGB
    overlay[~prediction.valid_map] = INVALID_RGB
    return heatmap, overlay


def _write_rgb(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f'无法写入图像: {path}')


def export_prediction(image: np.ndarray, prediction: Prediction, output_dir: Path, stem: str, *,
                      score_range: tuple[float, float], threshold: float | None = None,
                      alpha: float = .5) -> dict:
    """原图与预测 -> 热力图、叠加图、可选定位标注和完整原图坐标JSON。

    同stem的本次导出文件会被刷新，供验证阈值确定后补定位使用。
    调用者应使用专属新结果目录；CLI和流程入口拒绝已有结果目录。
    已有阈值撤回为None请换目录/前缀；当前不会删除此前的定位PNG。
    """
    if (not isinstance(stem, str) or not stem or stem in ('.', '..') or '/' in stem or '\\' in stem or
        not isinstance(prediction.image_id, str) or not prediction.image_id.strip()):
        raise ValueError('导出需要原图身份和不含路径的文件前缀')
    heatmap, overlay = render_prediction(image, prediction, score_range, alpha=alpha)
    regions = None if threshold is None else extract_regions(prediction, threshold)
    bounds = validate_display_settings(score_range, alpha)
    files = {'heatmap_file': stem+'_heatmap.png', 'overlay_file': stem+'_overlay.png',
             'localization_file': None if threshold is None else stem+'_localization.png',
             'detections_file': stem+'_detections.json', 'score_range': list(bounds)}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rgb(output_dir/files['heatmap_file'], heatmap)
    _write_rgb(output_dir/files['overlay_file'], overlay)
    if regions is not None:
        marked = image.copy()
        for region in regions:
            x1, y1, x2, y2 = region['bbox_xyxy']
            cv2.rectangle(marked, (x1, y1), (x2-1, y2-1), (255, 0, 0), 1)
            # 浮点位置是像素中心，画回整数像素下标用floor而不是四舍五入。
            x, y = (math.floor(v) for v in region['peak_xy'])
            cv2.drawMarker(marked, (x, y), (255, 255, 0), cv2.MARKER_CROSS, 5, 1)
        marked[~prediction.valid_map] = INVALID_RGB
        _write_rgb(output_dir/files['localization_file'], marked)
    status = ('threshold_not_selected' if threshold is None else
              'no_valid_prediction' if not prediction.valid_map.any() else
              'partial_coverage' if not prediction.valid_map.all() else 'available')
    metadata = {'image_id': prediction.image_id, 'original_hw': list(prediction.score_map.shape),
                'kind': 'thresholded_anomaly_regions', 'localization_status': status,
                'threshold': None if threshold is None else float(threshold),
                'threshold_rule': 'score >= threshold', 'coverage': prediction.coverage,
                'complete_coverage': bool(prediction.valid_map.all()), 'image_score': prediction.image_score,
                'coordinates': 'original_xy_pixel_centers', 'bbox_convention': 'half_open_xyxy',
                'regions': regions, 'scores_are_probabilities': False,
                'visualization': {'score_range': list(bounds), 'colormap': 'opencv_turbo',
                                  'alpha': alpha, 'invalid_rgb': list(INVALID_RGB), 'clipping': 'display_only'},
                'files': files}
    (output_dir/files['detections_file']).write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--prediction', type=Path, required=True, help='含image_id/score_map/valid_map的原图NPZ')
    parser.add_argument('--output', type=Path, required=True, help='尚不存在的新导出目录')
    parser.add_argument('--image-id', help='与NPZ一致的原图身份，默认使用原图文件stem')
    parser.add_argument('--threshold', type=float, help='已确定的原始分数阈值；省略时只导出显示图')
    parser.add_argument('--score-min', type=float, default=0.)
    parser.add_argument('--score-max', type=float, required=True, help='明确的显示上限，颜色不自动逐图拉伸')
    parser.add_argument('--alpha', type=float, default=.5)
    args = parser.parse_args()
    with np.load(args.prediction, allow_pickle=False) as data:
        prediction = Prediction(data['score_map'].copy(), data['valid_map'].copy(), str(data['image_id'].item()))
    if prediction.image_id != (args.image_id or args.image.stem):
        parser.error('原图身份与NPZ不一致；请核对输入或显式提供--image-id')
    image = load_image(args.image)
    validate_display_settings((args.score_min, args.score_max), args.alpha)
    args.output.mkdir(parents=True, exist_ok=False)
    files = export_prediction(image, prediction, args.output, 'prediction',
                              score_range=(args.score_min, args.score_max), threshold=args.threshold, alpha=args.alpha)
    print(json.dumps({'image_id': prediction.image_id, 'coverage': prediction.coverage,
                      'files': files}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
