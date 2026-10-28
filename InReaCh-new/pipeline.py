"""原尺度切块建库、原图预测和框级评估；正式实验必须显式提供划分。"""
from collections.abc import Callable, Iterator
from dataclasses import asdict
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
import torch
from config import ExperimentConfig
from contracts import ImageRecord, TileBatch, DescriptorSet, TileMeta, MemoryBank, Prediction
from dataset import discover_images, load_yolo_boxes
from image_processing import load_image, to_model_input
from tiling import iter_tiles
from descriptors import build_descriptors, merge_descriptor_sets, select_reference_descriptors
from feature_extraction import extract_features, load_feature_model
from association import associate_channels
from memory_bank import build_memory_bank, load_memory_bank, save_memory_bank
from prediction import predict_tiles
from stitching import stitch_scores
from evaluation import evaluate_predictions, select_threshold
from visualization import export_prediction, select_display_range, validate_display_settings


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


def extract_record_descriptors(record: ImageRecord, model: torch.nn.Module,
                               config: ExperimentConfig) -> tuple[DescriptorSet, tuple[TileMeta, ...]]:
    """一张原图 -> 跨批汇总的完整描述子/块元数据；不读取标签或删除重叠观测。"""
    if model.training:
        raise ValueError('特征模型必须处于eval模式')
    fragments, metadata = [], []
    for batch in prepare_batches(record, config):
        fragments.extend(build_descriptors(extract_features(model, batch.inputs), batch.metadata, config))
        metadata.extend(batch.metadata)
    merged = merge_descriptor_sets(fragments)
    if len(merged) != 1 or merged[0].image_id != record.image_id:
        raise ValueError('特征提取发生原图身份混淆')
    return merged[0], tuple(metadata)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _model_cache_config(model: torch.nn.Module) -> dict:
    """模型显式声明全部非state_dict前向配置；不推测任意Python对象的语义。"""
    provider = getattr(model, 'inreach_cache_config', None)
    if not callable(provider):
        raise ValueError('模型必须提供inreach_cache_config()声明完整前向配置，才能建立或复用参考库')
    config = provider()
    if not isinstance(config, dict) or type(config.get('version')) is not int or config['version'] <= 0:
        raise ValueError('模型配置必须是包含正整数version的JSON对象')
    try:
        return json.loads(json.dumps(config, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError('模型配置必须是可序列化且数值有限的JSON对象') from error


def _model_digest(model: torch.nn.Module) -> str:
    """显式前向配置、结构和权重共同签名；未声明的自定义模型拒绝使用。"""
    declared = _model_cache_config(model)
    code = getattr(model, 'code', None)
    if code is None:
        try:
            code = inspect.getsource(type(model))
        except (OSError, TypeError):
            code = repr(model)
    identity = {'signature_version': 2, 'class': type(model).__module__+'.'+type(model).__qualname__,
                'config': declared, 'code': str(code), 'representation': repr(model)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode())
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu().contiguous()
        digest.update((name+str(tensor.shape)+str(tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _check_records(records: list[ImageRecord]) -> None:
    if not records or any(not r.image_id for r in records):
        raise ValueError('原图清单不能为空，image_id必须非空')
    if len({r.image_id for r in records}) != len(records):
        raise ValueError('清单含重复原图身份')
    if len({Path(r.image_path).resolve() for r in records}) != len(records):
        raise ValueError('清单含重复原图路径')


def validate_splits(reference: list[ImageRecord], validation: list[ImageRecord],
                    testing: list[ImageRecord]) -> None:
    """检查显式清单的身份/路径/文件内容/已知工件分组跨集合泄漏。"""
    ids, paths, contents, groups = {}, {}, {}, {}
    for split, records in (('reference', reference), ('validation', validation), ('test', testing)):
        _check_records(records)
        for r in records:
            values = ((ids, r.image_id, '原图身份'), (paths, str(Path(r.image_path).resolve()), '路径'),
                      (contents, _file_digest(r.image_path), '文件内容'))
            for registry, key, label in values:
                if registry is contents and split == 'reference' and key in registry:
                    raise ValueError('参考集文件内容重复，不能计为独立原图支持')
                if key in registry and registry[key] != split:
                    raise ValueError(f'{label}在{registry[key]}和{split}之间重叠')
                registry[key] = split
            if r.group_id is not None:
                if not r.group_id:
                    raise ValueError('工件分组不能为空字符串')
                if r.group_id in groups and groups[r.group_id] != split:
                    raise ValueError('同工件分组不能跨参考/验证/测试集合')
                groups[r.group_id] = split


def fit_reference_bank(records: list[ImageRecord], model: torch.nn.Module, config: ExperimentConfig,
                       *, cache_path: Path | None = None, progress=None) -> MemoryBank:
    """无标签原图清单 -> 正常库；逐原图提取，只保留归属后的参考行。"""
    _check_records(records)
    if len(records) <= config.min_channel_length:
        raise ValueError('参考原图数量不足以超过通道长度阈值')
    sources, seen_contents = [], set()
    for record in records:
        content = _file_digest(record.image_path)
        if content in seen_contents:
            raise ValueError('参考集文件内容重复，不能计为独立原图支持')
        seen_contents.add(content)
        sources.append({'image_id': record.image_id, 'sha256': content, 'group_id': record.group_id})
    provenance = {'reference_images': sources,
                  'model_sha256': _model_digest(model),
                  'model_config': _model_cache_config(model),
                  'preprocessing': 'RGB/ImageNet-mean-std/native-tiles-v1'}
    if cache_path is not None and Path(cache_path).exists():
        return load_memory_bank(cache_path, config, provenance)
    selected, feature_bytes = [], 0
    for i, record in enumerate(records):
        full, metadata = extract_record_descriptors(record, model, config)
        owned = select_reference_descriptors([full], metadata)[0]
        feature_bytes += owned.features.numel()*owned.features.element_size()
        if feature_bytes > config.reference_feature_limit_mb*1024**2:
            raise ValueError('参考特征超过显式内存预算；请缩小参考清单或提高预算，不自动下采样')
        selected.append(owned)
        if progress:
            progress(f'参考 {i+1}/{len(records)} {record.image_id}: {len(full.features)}观测 -> {len(owned.features)}归属行')
        del full
    bank = build_memory_bank(associate_channels(selected, config), config)
    bank.provenance = provenance
    if cache_path is not None:
        save_memory_bank(bank, cache_path)
    return bank


def predict_records(records: list[ImageRecord], bank: MemoryBank, model: torch.nn.Module,
                    config: ExperimentConfig, *, progress=None,
                    on_prediction: Callable[[str, Prediction], None] | None = None) -> dict[str, Prediction]:
    """测试原图清单 -> 按原图键控的原始分数和有效性；保留所有重叠预测。"""
    _check_records(records)
    if bank.provenance.get('model_sha256') != _model_digest(model):
        raise ValueError('预测模型与参考库权重/结构签名不一致')
    reference = bank.provenance.get('reference_images', [])
    ref_ids = {r['image_id'] for r in reference}
    ref_content = {r['sha256'] for r in reference}
    ref_groups = {r['group_id'] for r in reference if r.get('group_id') is not None}
    for r in records:
        if (r.image_id in ref_ids or _file_digest(r.image_path) in ref_content or
            (r.group_id is not None and r.group_id in ref_groups)):
            raise ValueError('预测原图与参考集身份、内容或分组重叠')
    outputs = {}
    for i, record in enumerate(records):
        descriptors, metadata = extract_record_descriptors(record, model, config)
        tiles = predict_tiles([descriptors], bank, metadata, config)
        outputs[record.image_id] = stitch_scores(tiles, descriptors.original_hw)
        if on_prediction is not None:
            on_prediction(record.image_id, outputs[record.image_id])
        if progress:
            progress(f'预测 {i+1}/{len(records)} {record.image_id}: coverage={outputs[record.image_id].coverage:.6f}')
    return outputs


def load_manifest(path: Path) -> dict:
    """JSON显式划分 -> ImageRecord清单；相对路径以清单目录为基准。"""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    for name in ('reference', 'validation', 'test'):
        records = []
        for item in manifest[name]:
            def resolved(key):
                value = item.get(key)
                return None if value is None else (path.parent/str(value)).resolve()
            image_path = resolved('image_path')
            if image_path is None:
                raise ValueError('清单缺少image_path')
            records.append(ImageRecord(item['image_id'], image_path, resolved('label_path'), item.get('group_id')))
        manifest[name] = records
    return manifest


def run_experiment(reference: list[ImageRecord], validation: list[ImageRecord], testing: list[ImageRecord],
                   model: torch.nn.Module, config: ExperimentConfig, output_dir: Path, *,
                   threshold_candidates: list[float] | None = None,
                   max_unmatched_regions_per_image: float | None = None,
                   annotations_complete: bool = False, split_confirmed: bool = False,
                   fixed_threshold: float | None = None,
                   visualization_range: tuple[float, float] | None = None,
                   overlay_alpha: float = .5, progress=None) -> dict:
    """完整清单 -> 正常库、原图NPZ、验证阈值曲线和报告；不自行生成划分。"""
    validate_splits(reference, validation, testing)
    if fixed_threshold is None:
        if not annotations_complete or not threshold_candidates or max_unmatched_regions_per_image is None:
            raise ValueError('验证选阈值需要完整标注、候选值和显式未匹配区域预算')
    elif not math.isfinite(fixed_threshold):
        raise ValueError('固定阈值必须有限')
    display_range = validate_display_settings(visualization_range, overlay_alpha)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    report = {'status': 'running',
              'config': {**asdict(config), 'data_root': str(config.data_root)},
              'threshold': None, 'test_metrics': None, 'artifacts': [],
              'visualization': {'score_range': None if display_range is None else list(display_range),
                                'range_source': 'explicit' if display_range is not None else 'validation_valid_max',
                                'overlay_alpha': overlay_alpha, 'colormap': 'opencv_turbo',
                                'invalid_rgb': [128, 128, 128]},
              'reference_provenance': None,
              'split_images': {name: [{'image_id': r.image_id, 'image_path': str(r.image_path),
                                      'group_id': r.group_id, 'image_sha256': _file_digest(r.image_path),
                                      'label_sha256': _file_digest(r.label_path) if r.label_path and r.label_path.exists() else None}
                                     for r in records]
                               for name, records in (('validation', validation), ('test', testing))},
              'seconds': {}}

    def checkpoint():
        report['seconds']['total'] = time.perf_counter()-started
        report['process_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        temporary = output_dir/'report.json.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        temporary.replace(output_dir/'report.json')

    counts = {'validation': 0, 'test': 0}
    artifacts_by_image = {}
    records_by_image = {(split, record.image_id): record
                        for split, records in (('validation', validation), ('test', testing)) for record in records}

    def render_artifact(split, image_id, prediction, threshold):
        artifact = artifacts_by_image[(split, image_id)]
        image = load_image(records_by_image[(split, image_id)].image_path)
        artifact['visualization'] = export_prediction(image, prediction, output_dir,
                    Path(artifact['file']).stem, score_range=display_range, threshold=threshold, alpha=overlay_alpha)
        checkpoint()

    def persist_prediction(split, image_id, prediction):
        # 每张原图完成就落盘；下一张读图、阈值选择或严格评估失败不能丢掉已有证据。
        filename = f'{split}_{counts[split]:04d}.npz'
        np.savez_compressed(output_dir/filename, image_id=np.asarray(image_id),
                            score_map=prediction.score_map, valid_map=prediction.valid_map)
        counts[split] += 1
        artifact = {'split': split, 'image_id': image_id, 'file': filename,
                    'image_score': prediction.image_score, 'coverage': prediction.coverage}
        report['artifacts'].append(artifact)
        artifacts_by_image[(split, image_id)] = artifact
        checkpoint()
        if split == 'test':
            # 此时验证集色标与阈值已冻结；测试图极值不反过来改变显示比例。
            render_artifact(split, image_id, prediction, report['threshold']['threshold'])

    stage = 'fit'
    checkpoint()
    try:
        bank = fit_reference_bank(reference, model, config, cache_path=output_dir/'normal_bank.npz', progress=progress)
        fitted = time.perf_counter()
        report['reference_provenance'] = bank.provenance
        report['seconds']['fit'] = fitted-started
        stage = 'validation_prediction'
        val_predictions = predict_records(validation, bank, model, config, progress=progress,
                on_prediction=lambda image_id, p: persist_prediction('validation', image_id, p))
        stage = 'validation_visualization'
        if display_range is None:
            display_range = select_display_range(list(val_predictions.values()))
        report['visualization']['score_range'] = list(display_range)
        for image_id, prediction in val_predictions.items():
            # 即使后续无法选阈值，已有显示图仍保留，JSON明确定位未就绪。
            render_artifact('validation', image_id, prediction, fixed_threshold)
        stage = 'validation_labels'
        val_boxes = {r.image_id: load_yolo_boxes(r.label_path, val_predictions[r.image_id].score_map.shape)
                     for r in validation}
        stage = 'validation_threshold'
        if fixed_threshold is None:
            threshold = select_threshold(val_predictions, val_boxes, threshold_candidates,
                                         max_unmatched_regions_per_image=max_unmatched_regions_per_image,
                                         annotations_complete=annotations_complete)
        else:
            threshold = {'threshold': float(fixed_threshold), 'objective': 'external_fixed_exploratory', 'curve': []}
        report['threshold'] = threshold
        if fixed_threshold is None:
            stage = 'validation_localization'
            for image_id, prediction in val_predictions.items():
                render_artifact('validation', image_id, prediction, threshold['threshold'])
        validated = time.perf_counter()
        report['seconds']['validation'] = validated-fitted
        checkpoint()
        stage = 'test_prediction'
        test_predictions = predict_records(testing, bank, model, config, progress=progress,
                on_prediction=lambda image_id, p: persist_prediction('test', image_id, p))
        stage = 'test_labels'
        test_boxes = {r.image_id: load_yolo_boxes(r.label_path, test_predictions[r.image_id].score_map.shape)
                      for r in testing}
        stage = 'test_evaluation'
        metrics = evaluate_predictions(test_predictions, test_boxes, threshold['threshold'],
                                       annotations_complete=annotations_complete)
        report['test_metrics'] = metrics
        confirmed = (split_confirmed and annotations_complete and fixed_threshold is None and
                     all(r.group_id for r in reference+validation+testing) and metrics['unknown_label_images'] == 0)
        report['status'] = 'confirmed_split' if confirmed else 'exploratory'
        report['seconds']['test_and_output'] = time.perf_counter()-validated
        checkpoint()
    except Exception as error:
        report.update(status='incomplete', failed_stage=stage,
                      error={'type': type(error).__name__, 'message': str(error)})
        checkpoint()
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ExperimentConfig().data_root)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--inspect',action='store_true',help='只检查输入')
    modes.add_argument('--run',action='store_true',help='执行显式清单的建库、预测和评估')
    parser.add_argument('--manifest',type=Path,help='含reference/validation/test的JSON清单')
    parser.add_argument('--output',type=Path,help='新的结果目录，不覆盖已有目录')
    parser.add_argument('--limit',type=int,default=1,help='检查前 N 张原图')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--assoc-depth',type=int,default=10)
    parser.add_argument('--min-channel-length',type=int,default=3)
    parser.add_argument('--max-channel-std',type=float,default=5.)
    parser.add_argument('--radius-px',type=float,default=8.)
    parser.add_argument('--no-spatial-gate',action='store_true')
    parser.add_argument('--reference-feature-limit-mb',type=float,default=2048.)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error('--limit 必须大于零')
    config = ExperimentConfig(data_root=args.data_root,batch_size=args.batch_size,
                              assoc_depth=args.assoc_depth, min_channel_length=args.min_channel_length,
                              max_channel_std=args.max_channel_std,
                              radius_px=None if args.no_spatial_gate else args.radius_px,
                              reference_feature_limit_mb=args.reference_feature_limit_mb)
    if args.run:
        if args.manifest is None or args.output is None:
            parser.error('--run需要--manifest和--output')
        manifest = load_manifest(args.manifest)
        visualization = manifest.get('visualization', {})
        if not isinstance(visualization, dict):
            parser.error('清单visualization必须为JSON对象')
        device = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(0)
        np.random.seed(0)
        model = load_feature_model(torch.device(device))
        report = run_experiment(manifest['reference'], manifest['validation'], manifest['test'], model, config,
                    args.output, threshold_candidates=manifest.get('threshold_candidates'),
                    max_unmatched_regions_per_image=manifest.get('max_unmatched_regions_per_image'),
                    annotations_complete=manifest.get('annotations_complete', False),
                    split_confirmed=manifest.get('split_confirmed', False), fixed_threshold=manifest.get('fixed_threshold'),
                    visualization_range=visualization.get('score_range'), overlay_alpha=visualization.get('overlay_alpha', .5),
                    progress=lambda message: print(message, flush=True))
        print(json.dumps({'status': report['status'], 'test_metrics': report['test_metrics'],
                          'report': str(args.output/'report.json')}, ensure_ascii=False, indent=2))
        return
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
