"""对照指定源码快照验证聚合数值/元数据和耗时；不会运行整图检测。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
from config import ExperimentConfig, algorithm_signature
from descriptors import _pooled_level, build_descriptors


def comparison(reference, candidate, old_seconds, new_seconds):
    difference = (reference-candidate).abs()
    return {'reference_seconds': old_seconds, 'candidate_seconds': new_seconds,
            'speedup': old_seconds/new_seconds, 'shape': list(reference.shape),
            'max_absolute_error': float(difference.max()),
            'mean_absolute_error': float(difference.mean()),
            'bitwise_equal': torch.equal(reference, candidate),
            'allclose': bool(torch.allclose(reference, candidate, atol=1e-6, rtol=1e-5))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--require-speedup', type=float, default=2.)
    parser.add_argument('--real-tile', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(101)
    spec = importlib.util.spec_from_file_location('aggregation_baseline', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    report = {'baseline': str(args.baseline),
              'baseline_sha256': hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
              'candidate_sha256': hashlib.sha256(Path('descriptors.py').read_bytes()).hexdigest(),
              'torch': torch.__version__, 'threads': args.threads,
              'tolerance': {'atol': 1e-6, 'rtol': 1e-5}, 'required_speedup': args.require_speedup}
    feature = torch.randn(2048, 7, 7)
    started = time.perf_counter(); old = baseline._pooled_level(feature, (56, 56)); old_seconds = time.perf_counter()-started
    print('random deep-layer reference seconds', old_seconds, flush=True)
    started = time.perf_counter(); new = _pooled_level(feature, (56, 56)); new_seconds = time.perf_counter()-started
    report['deep_layer'] = comparison(old, new, old_seconds, new_seconds)
    print(json.dumps(report['deep_layer']), flush=True)
    del old, new, feature
    if args.real_tile:
        from dataset import discover_images
        from feature_extraction import load_feature_model, extract_features
        from pipeline import prepare_batches
        cfg = ExperimentConfig(batch_size=1, assoc_depth=1)
        record = discover_images(cfg.data_root)[0]
        model = load_feature_model(torch.device('cuda:0'))
        batch = next(prepare_batches(record, cfg))
        features = extract_features(model, batch.inputs)
        torch.cuda.synchronize()
        started = time.perf_counter(); old = baseline.build_descriptors(features, batch.metadata, cfg)[0]; old_seconds = time.perf_counter()-started
        print('real 13-level reference seconds', old_seconds, flush=True)
        started = time.perf_counter(); new = build_descriptors(features, batch.metadata, cfg)[0]; new_seconds = time.perf_counter()-started
        result = comparison(old.features, new.features, old_seconds, new_seconds)
        result.update(image_id=record.image_id, tile_id=batch.metadata[0].tile_id, levels=len(features),
                      metadata_equal=bool(old.image_id == new.image_id and old.original_hw == new.original_hw and
                          old.grid_hw == new.grid_hw and np.array_equal(old.xy, new.xy) and
                          np.array_equal(old.tile_ids, new.tile_ids) and np.array_equal(old.valid, new.valid)),
                      algorithm_signature=algorithm_signature(cfg))
        report['real_tile'] = result
        print(json.dumps(result), flush=True)
    cases = [report['deep_layer']]+([report['real_tile']] if args.real_tile else [])
    report['passed'] = all(r['allclose'] and r.get('metadata_equal', True) and r['speedup'] >= args.require_speedup for r in cases)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    if not report['passed']:
        raise SystemExit('aggregation equivalence/performance requirement failed; see JSON')


if __name__ == '__main__':
    main()
