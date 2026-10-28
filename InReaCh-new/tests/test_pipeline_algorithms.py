"""真实读图和算法串联；小骨干只验证软件流程，不代表预训练模型准确率。"""
import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from config import ExperimentConfig
from contracts import ImageRecord
import pipeline


class SmallFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)

    def forward(self, inputs):
        return {'Level_1': F.avg_pool2d(inputs, 4)*self.scale}

    def inreach_cache_config(self):
        return {'version': 1, 'pool_kernel': 4}


class ChangingGridFeatures(SmallFeatures):
    """亮像素触发不同网格相位，产生真实的空间无候选，而不是伪造预测结果。"""
    def forward(self, inputs):
        size = 3 if inputs.max() > 2.5 else 2
        return {'Level_1': F.adaptive_avg_pool2d(inputs, (size, size))*self.scale}

    def inreach_cache_config(self):
        return {'version': 1, 'bright_threshold': 2.5, 'grid_sizes': [2, 3]}


class UnsignedScaleFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.scale = 1.  # 普通属性不会进入state_dict或默认repr。

    def forward(self, inputs):
        return {'Level_1': F.avg_pool2d(inputs, 4)*self.weight*self.scale}


class SignedScaleFeatures(UnsignedScaleFeatures):
    def inreach_cache_config(self):
        return {'version': 1, 'pool_kernel': 4, 'scale': self.scale}


class PipelineAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = ExperimentConfig(tile_size=8, stride=4, batch_size=1,
                                    assoc_depth=1, min_channel_length=1, radius_px=0)
        self.model = SmallFeatures().eval()
        self.reference = [self.record('r'+str(i)) for i in range(3)]
        self.validation = [self.record('v', (1, 1))]
        self.testing = [self.record('t', (1, 10))]

    def record(self, name, changed=None, changed_value=255):
        # 独立参考样本有不同内容，不能用三个相同文件冒充独立支持。
        gray = 128+int(name[1:]) if name.startswith('r') and name[1:].isdigit() else 128
        pixels = np.full((8, 13, 3), gray, dtype=np.uint8)
        if changed:
            pixels[changed] = changed_value
        image = self.root / (name+'.png')
        cv2.imwrite(str(image), pixels)
        label = self.root / (name+'.txt')
        # 参考图标签故意非法，建库不得读取它。
        label.write_text('not a label' if name.startswith('r') else '0 0.5 0.5 1 1\n')
        return ImageRecord(name, image, label)

    def test_reference_labels_are_ignored_and_prediction_keeps_full_original(self):
        bank = pipeline.fit_reference_bank(self.reference, self.model, self.cfg)
        output = pipeline.predict_records(self.testing, bank, self.model, self.cfg)
        self.assertEqual(set(output), {'t'})
        self.assertEqual(output['t'].score_map.shape, (8, 13))
        self.assertEqual(output['t'].coverage, 1.)
        self.assertTrue(np.isfinite(output['t'].score_map).all())
        self.assertEqual(len(bank.provenance['reference_images']), 3)
        with self.assertRaisesRegex(ValueError, '参考|重叠'):
            pipeline.predict_records(self.reference[:1], bank, self.model, self.cfg)

    def test_duplicate_reference_content_is_rejected_before_fit_or_cache_reuse(self):
        duplicate = self.record('copy')
        records = self.reference+[duplicate]
        with self.assertRaisesRegex(ValueError, '内容.*重复|重复.*内容'):
            pipeline.fit_reference_bank(records, self.model, self.cfg)
        cache = self.root/'bank.npz'
        pipeline.fit_reference_bank(self.reference, self.model, self.cfg, cache_path=cache)
        with self.assertRaisesRegex(ValueError, '内容.*重复|重复.*内容'):
            pipeline.fit_reference_bank(records, self.model, self.cfg, cache_path=cache)
        with self.assertRaisesRegex(ValueError, '内容.*重复|重复.*内容'):
            pipeline.validate_splits(records, self.validation, self.testing)

    def test_end_to_end_saves_raw_predictions_and_frozen_validation_threshold(self):
        report = pipeline.run_experiment(self.reference, self.validation, self.testing,
                   self.model, self.cfg, self.root/'output', threshold_candidates=[0., 100.],
                   max_unmatched_regions_per_image=1, annotations_complete=True)
        self.assertEqual(report['threshold']['threshold'], 0.)
        self.assertEqual(report['test_metrics']['valid_coverage'], 1.)
        self.assertEqual(report['status'], 'exploratory')
        self.assertTrue((self.root/'output'/'report.json').is_file())
        with np.load(self.root/'output'/'test_0000.npz', allow_pickle=False) as data:
            self.assertEqual(data['score_map'].shape, (8, 13))
            self.assertEqual(data['image_id'].item(), 't')
        for artifact in report['artifacts']:
            files = artifact['visualization']
            for key in ('heatmap_file', 'overlay_file', 'localization_file', 'detections_file'):
                self.assertTrue((self.root/'output'/files[key]).is_file())
            locations = json.loads((self.root/'output'/files['detections_file']).read_text())
            self.assertEqual(locations['image_id'], artifact['image_id'])
            self.assertEqual(locations['threshold'], report['threshold']['threshold'])

    def test_shared_visualization_range_comes_from_validation_not_test(self):
        self.validation = [self.record('v', (1, 1), changed_value=200)]
        output = self.root/'shared_scale'
        report = pipeline.run_experiment(self.reference, self.validation, self.testing,
                   self.model, self.cfg, output, fixed_threshold=0.)
        with np.load(output/'validation_0000.npz', allow_pickle=False) as data:
            maximum = float(data['score_map'][data['valid_map']].max())
        self.assertEqual(report['visualization']['score_range'], [0., maximum])
        self.assertEqual(report['visualization']['range_source'], 'validation_valid_max')
        for artifact in report['artifacts']:
            self.assertEqual(artifact['visualization']['score_range'], [0., maximum])

    def test_explicit_visualization_range_and_alpha_are_recorded(self):
        output = self.root/'explicit_scale'
        report = pipeline.run_experiment(self.reference, self.validation, self.testing,
                   self.model, self.cfg, output, fixed_threshold=0.,
                   visualization_range=(0., 10.), overlay_alpha=0.)
        self.assertEqual(report['visualization']['range_source'], 'explicit')
        self.assertEqual(report['visualization']['score_range'], [0., 10.])
        overlay = cv2.imread(str(output/'test_0000_overlay.png'))
        np.testing.assert_array_equal(overlay[0, 0], [128, 128, 128])

    def assert_incomplete_output(self, output, stage, filename):
        report = json.loads((output/'report.json').read_text())
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(report['failed_stage'], stage)
        self.assertIsNone(report['test_metrics'])
        artifact = next(a for a in report['artifacts'] if a['file'] == filename)
        self.assertGreater(artifact['coverage'], 0.)
        self.assertLess(artifact['coverage'], 1.)
        with np.load(output/filename, allow_pickle=False) as data:
            self.assertEqual(artifact['coverage'], float(data['valid_map'].mean()))
            self.assertTrue(np.isfinite(data['score_map'][data['valid_map']]).all())
            self.assertTrue(np.isnan(data['score_map'][~data['valid_map']]).all())
        files = artifact['visualization']
        self.assertTrue((output/files['heatmap_file']).is_file())
        self.assertTrue((output/files['overlay_file']).is_file())
        locations = json.loads((output/files['detections_file']).read_text())
        self.assertEqual(locations['coverage'], artifact['coverage'])
        if stage == 'validation_threshold':
            self.assertIsNone(locations['regions'])
            self.assertEqual(locations['localization_status'], 'threshold_not_selected')
        else:
            self.assertEqual(locations['localization_status'], 'partial_coverage')
            self.assertTrue((output/files['localization_file']).is_file())

    def test_validation_coverage_failure_preserves_raw_prediction_and_report(self):
        output = self.root/'validation_holes'
        with self.assertRaisesRegex(ValueError, '覆盖'):
            pipeline.run_experiment(self.reference, self.validation, self.testing,
                ChangingGridFeatures().eval(), self.cfg, output, threshold_candidates=[0., 100.],
                max_unmatched_regions_per_image=1, annotations_complete=True)
        self.assert_incomplete_output(output, 'validation_threshold', 'validation_0000.npz')

    def test_test_coverage_failure_preserves_both_splits_and_frozen_threshold(self):
        self.validation = [self.record('v', (1, 1), changed_value=200)]
        output = self.root/'test_holes'
        with self.assertRaisesRegex(ValueError, '覆盖'):
            pipeline.run_experiment(self.reference, self.validation, self.testing,
                ChangingGridFeatures().eval(), self.cfg, output, threshold_candidates=[0., 100.],
                max_unmatched_regions_per_image=1, annotations_complete=True)
        self.assert_incomplete_output(output, 'test_evaluation', 'test_0000.npz')
        self.assertTrue((output/'validation_0000.npz').is_file())
        report = json.loads((output/'report.json').read_text())
        self.assertEqual(report['threshold']['threshold'], 0.)

    def test_later_image_failure_preserves_completed_prediction(self):
        broken = self.root/'broken.png'
        broken.write_bytes(b'not an image')
        records = self.validation+[ImageRecord('broken', broken, None)]
        output = self.root/'broken_image'
        with self.assertRaises(FileNotFoundError):
            pipeline.run_experiment(self.reference, records, self.testing, self.model, self.cfg,
                                   output, fixed_threshold=0.)
        with np.load(output/'validation_0000.npz', allow_pickle=False) as data:
            self.assertEqual(data['image_id'].item(), 'v')
        report = json.loads((output/'report.json').read_text())
        self.assertEqual(report['failed_stage'], 'validation_prediction')

    def test_split_checks_image_content_and_group_leakage(self):
        copy = self.record('copy')  # 与参考图同像素、同文件内容但不同id/路径。
        with self.assertRaisesRegex(ValueError, '内容|重叠'):
            pipeline.validate_splits(self.reference, self.validation, [copy])
        shared_group = ImageRecord('v', self.validation[0].image_path, None, group_id='same')
        reference = [ImageRecord('r0', self.reference[0].image_path, None, group_id='same')]
        with self.assertRaisesRegex(ValueError, '分组|工件'):
            pipeline.validate_splits(reference, [shared_group], self.testing)

    def test_reference_feature_limit_refuses_instead_of_subsampling(self):
        with self.assertRaisesRegex(ValueError, '内存|预算'):
            pipeline.fit_reference_bank(self.reference, self.model,
                                        replace(self.cfg, reference_feature_limit_mb=.001))

    def test_manifest_resolves_paths_relative_to_file(self):
        path = self.root/'manifest.json'
        entries = {'reference': [{'image_id': 'r0', 'image_path': 'r0.png', 'group_id': 'g'}],
                   'validation': [{'image_id': 'v', 'image_path': 'v.png', 'label_path': 'v.txt'}],
                   'test': [{'image_id': 't', 'image_path': 't.png'}]}
        path.write_text(json.dumps(entries))
        loaded = pipeline.load_manifest(path)
        self.assertEqual(loaded['reference'][0].image_path, self.root/'r0.png')
        self.assertEqual(loaded['reference'][0].group_id, 'g')
        self.assertIsNone(loaded['test'][0].label_path)

    def test_unsigned_custom_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '配置|声明|签名'):
            pipeline.fit_reference_bank(self.reference, UnsignedScaleFeatures().eval(), self.cfg)

    def test_custom_forward_config_change_rejects_cache_and_prediction(self):
        model = SignedScaleFeatures().eval()
        cache = self.root/'configured_bank.npz'
        bank = pipeline.fit_reference_bank(self.reference, model, self.cfg, cache_path=cache)
        inputs = torch.ones(1, 3, 8, 8)
        self.assertEqual(model(inputs)['Level_1'][0, 0, 0, 0], 1.)
        model.scale = 2.
        self.assertEqual(model(inputs)['Level_1'][0, 0, 0, 0], 2.)
        with self.assertRaisesRegex(ValueError, '来源|签名'):
            pipeline.fit_reference_bank(self.reference, model, self.cfg, cache_path=cache)
        with self.assertRaisesRegex(ValueError, '签名'):
            pipeline.predict_records(self.testing, bank, model, self.cfg)

    def test_nonfinite_declared_model_config_is_rejected(self):
        class InvalidConfigFeatures(SignedScaleFeatures):
            def inreach_cache_config(self):
                return {'version': 1, 'scale': float('nan')}
        with self.assertRaisesRegex(ValueError, '配置|有限|JSON'):
            pipeline.fit_reference_bank(self.reference, InvalidConfigFeatures().eval(), self.cfg)


if __name__ == '__main__': unittest.main()
