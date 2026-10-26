"""框架入口: 目前仅打通原图到切块批次, 不生成异常分数或检测指标."""
from collections.abc import Iterator
import argparse
import json
from pathlib import Path
import torch
from config import ExperimentConfig
from contracts import ImageRecord, TileBatch
from dataset import discover_images, load_yolo_boxes
from image_processing import load_image, to_model_input
from tiling import iter_tiles


def prepare_batches(record: ImageRecord, config: ExperimentConfig) -> Iterator[TileBatch]:
    """原图记录 -> 已归一化块批次; 不读取标签, 不调用整图缩放/中心裁剪."""
    image = load_image(record.image_path)
    inputs, metadata = [], []
    for tile in iter_tiles(image, record.image_id, config):
        inputs.append(to_model_input(tile.image))
        metadata.append(tile.meta)
        if len(inputs) == config.batch_size:
            yield TileBatch(torch.cat(inputs,dim=0),tuple(metadata))
            inputs, metadata = [], []
    if inputs:
        yield TileBatch(torch.cat(inputs,dim=0),tuple(metadata))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ExperimentConfig().data_root)
    parser.add_argument('--inspect',action='store_true',help='检查输入, 不运行未实现的检测算法')
    parser.add_argument('--limit',type=int,default=1,help='检查前 N 张原图')
    parser.add_argument('--batch-size',type=int,default=4)
    args = parser.parse_args()
    if not args.inspect:
        parser.error('当前仅提供 --inspect 输入检查, 检测算法尚未实现')
    if args.limit <= 0:
        parser.error('--limit 必须大于零')
    config = ExperimentConfig(data_root=args.data_root,batch_size=args.batch_size)
    records = discover_images(config.data_root)
    summaries = []
    for record in records[:args.limit]:
        image = load_image(record.image_path)
        boxes = load_yolo_boxes(record.label_path,image.shape[:2])
        count, first_shape = 0, None
        for batch in prepare_batches(record,config):
            count += len(batch.metadata)
            if first_shape is None:
                first_shape = list(batch.inputs.shape)
        summaries.append({'image_id':record.image_id,'original_hw':list(image.shape[:2]),
                          'boxes':None if boxes is None else len(boxes),'tiles':count,'first_batch_shape':first_shape})
    print(json.dumps({'mode':'input_inspection_only','dataset_images':len(records),'samples':summaries},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
