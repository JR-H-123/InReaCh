"""发现原图与解析 YOLO 检测框; 不自动划分参考/测试集, 不用标签筛选建库图片."""
from pathlib import Path
import math
import warnings
from contracts import Box, ImageRecord


def discover_images(root: Path, prefix: str = 'pic1_') -> list[ImageRecord]:
    """输入含 images/labels 的目录, 返回排序后的原图记录; 缺失标注保持 None."""
    root = Path(root)
    images = sorted(p for p in (root / 'images').glob(f'{prefix}*')
                    if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'})
    if not images:
        raise FileNotFoundError(f'没有找到图片: {root / "images"}, prefix={prefix}')
    records = []
    seen = set()
    for path in images:
        if path.stem in seen:
            raise ValueError(f'重复 image_id: {path.stem}')
        seen.add(path.stem)
        label = root / 'labels' / f'{path.stem}.txt'
        records.append(ImageRecord(path.stem, path, label if label.is_file() else None))
    return records


def load_yolo_boxes(path: Path | None, image_hw: tuple[int, int]) -> tuple[Box, ...] | None:
    """归一化 YOLO cx,cy,w,h 转原图浮点 xyxy; 缺失返回 None, 空文件返回空元组.

    空标注只有在数据完整性确认后才代表正常图. 非法值报出文件与行号.
    """
    h, w = image_hw
    if h <= 0 or w <= 0:
        raise ValueError('原图高宽必须大于零')
    if path is None or not Path(path).exists():
        return None
    path = Path(path)
    boxes = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError('需要 5 列 YOLO 检测框, 不支持分割多边形')
            class_id = int(fields[0])
            cx, cy, bw, bh = map(float, fields[1:])
            if class_id < 0 or not all(math.isfinite(v) for v in (cx,cy,bw,bh)):
                raise ValueError('类别或数值非法')
            x1, y1, x2, y2 = cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2
            tolerance = 1e-6  # 六位小数 YOLO 的中心/宽高分别舍入可使边界略微越界.
            if not (bw > 0 and bh > 0 and -tolerance <= x1 < x2 <= 1+tolerance and -tolerance <= y1 < y2 <= 1+tolerance):
                raise ValueError('框宽高非正或超出原图范围')
            if min(x1,y1) < 0 or max(x2,y2) > 1:
                warnings.warn(f'{path}:{line_no}: 边界在 1e-6 舍入容差内, 已裁回 [0,1]', UserWarning, stacklevel=2)
                x1,y1,x2,y2 = (min(1.0,max(0.0,v)) for v in (x1,y1,x2,y2))
            if x1 >= x2 or y1 >= y2:
                raise ValueError('边界裁回后框面积为零')
            boxes.append(Box(class_id,x1*w,y1*h,x2*w,y2*h))
        except ValueError as exc:
            raise ValueError(f'{path}:{line_no}: {exc}') from exc
    return tuple(boxes)
