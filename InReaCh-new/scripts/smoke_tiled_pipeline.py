"""完整真实原图的小样本流程检查，不选择阈值或报告准确率。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time
import cv2
import numpy as np
import torch
import torchvision
from config import ExperimentConfig
from dataset import discover_images
from feature_extraction import load_feature_model
from pipeline import fit_reference_bank, predict_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', nargs='+', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    config = ExperimentConfig(batch_size=2, assoc_depth=1)
    records = {r.image_id: r for r in discover_images(config.data_root)}
    reference, query = [records[i] for i in args.reference], [records[args.query]]
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = load_feature_model(device)
    progress = lambda message: print(message, flush=True)
    bank = fit_reference_bank(reference, model, config, progress=progress)
    fitted = time.perf_counter()
    predictions = predict_records(query, bank, model, config, progress=progress)
    finished = time.perf_counter()
    result = predictions[args.query]
    np.savez_compressed(args.output/'prediction.npz', image_id=np.asarray(args.query),
                        score_map=result.score_map, valid_map=result.valid_map)
    summary = {'status': 'smoke_only_no_accuracy_claim',
               'reference': args.reference, 'query': args.query,
               'config': {**asdict(config), 'data_root': str(config.data_root)},
               'provenance': bank.provenance, 'bank_shape': list(bank.features.shape),
               'original_hw': list(result.score_map.shape), 'coverage': result.coverage,
               'image_score': result.image_score,
               'valid_scores_finite': bool(np.isfinite(result.score_map[result.valid_map]).all()),
               'invalid_scores_nan': bool(np.isnan(result.score_map[~result.valid_map]).all()),
               'seconds': {'load_and_fit': fitted-started, 'predict': finished-fitted, 'total': finished-started},
               'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
               'environment': {'python': sys.version.split()[0], 'torch': torch.__version__,
                               'torchvision': torchvision.__version__, 'numpy': np.__version__,
                               'opencv': cv2.__version__, 'device': str(device),
                               'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}}
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('provenance', 'config')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
