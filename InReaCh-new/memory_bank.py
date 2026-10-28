"""筛选通道建立正常参考库; 数组缓存记录配置、数据与模型来源。"""
import json
from pathlib import Path
import numpy as np
import torch
from config import ExperimentConfig, algorithm_signature
from contracts import Channel, MemoryBank


def build_memory_bank(channels: list[Channel], config: ExperimentConfig) -> MemoryBank:
    """候选通道 -> 正常描述子、位置、来源与配置签名。

    依据独立原图支持数与通道离散程度筛选. 空结果应报错, 不能用全量特征回退.
    """
    retained = {}
    dimension = None
    for channel in channels:
        if len({m.image_id for m in channel.members}) != len(channel.members):
            raise ValueError('同一通道含重复原图，不能重复计支持数')
        for m in channel.members:
            if (not m.image_id or m.descriptor_index < 0 or m.feature.ndim != 1 or
                not m.feature.numel() or m.feature.device.type != 'cpu' or
                not torch.isfinite(m.feature).all() or np.asarray(m.xy).shape != (2,) or
                not np.isfinite(m.xy).all()):
                raise ValueError('通道成员特征、原图身份或坐标无效')
            if dimension is not None and dimension != len(m.feature):
                raise ValueError('通道描述子维度不一致')
            dimension = len(m.feature)
        if len(channel.members) <= config.min_channel_length:
            continue
        features = torch.stack([m.feature for m in channel.members]).double()
        radii = torch.linalg.vector_norm(features - features.mean(dim=0), dim=1)
        # 旧版np.std默认ddof=0，度量是距离的标准差而非各特征维的标准差。
        dispersion = radii.std(correction=0)
        keep = radii < config.max_channel_std * dispersion
        if torch.all(radii == 0):
            keep = torch.ones_like(keep)  # 完全相同的正常证据不应因0<0被删空。
        survivors = [m for m, take in zip(channel.members, keep) if bool(take)]
        if len(survivors) <= config.min_channel_length:
            continue
        for m in survivors:
            key = (m.image_id, m.descriptor_index)
            if key in retained:
                old = retained[key]
                if old.xy != m.xy or old.original_hw != m.original_hw or not torch.equal(old.feature, m.feature):
                    raise ValueError('同一原图描述子的多通道来源冲突')
            retained[key] = m
    if not retained:
        raise ValueError('没有合格通道，正常参考库为空；不回退到全量特征')
    members = list(retained.values())
    bank = MemoryBank(torch.stack([m.feature for m in members]).detach().float(),
                      np.asarray([m.xy for m in members], dtype=np.float64),
                      tuple(m.image_id for m in members),
                      np.asarray([m.descriptor_index for m in members], dtype=np.int64),
                      tuple(m.original_hw for m in members), algorithm_signature(config))
    validate_memory_bank(bank, config)
    return bank


def validate_memory_bank(bank: MemoryBank, config: ExperimentConfig) -> None:
    """拒绝空库、字段错位、坐标无效或建库/预测配置失配。"""
    if bank.algorithm != algorithm_signature(config):
        raise ValueError('参考库算法配置不一致')
    if bank.features.ndim != 2 or min(bank.features.shape) <= 0:
        raise ValueError('参考库为空或特征形状无效')
    n = len(bank.features)
    if (bank.features.device.type != 'cpu' or not bank.features.is_floating_point() or
        not torch.isfinite(bank.features).all() or bank.xy.shape != (n, 2) or
        not np.isfinite(bank.xy).all() or len(bank.image_ids) != n or
        len(bank.source_hw) != n or bank.descriptor_indices.shape != (n,) or
        not np.issubdtype(bank.descriptor_indices.dtype, np.integer) or
        np.any(bank.descriptor_indices < 0) or not all(bank.image_ids)):
        raise ValueError('参考库特征、坐标或来源字段无效')
    if len(set(zip(bank.image_ids, bank.descriptor_indices.tolist()))) != n:
        raise ValueError('参考库含重复来源')
    for xy, hw in zip(bank.xy, bank.source_hw):
        if hw is not None and (len(hw) != 2 or min(hw) <= 0 or np.any(xy < 0) or np.any(xy >= (hw[1], hw[0]))):
            raise ValueError('参考库坐标超出原图尺寸')
    if config.radius_px is not None and (None in bank.source_hw or len(set(bank.source_hw)) != 1):
        raise ValueError('空间门控参考库必须使用一致且已知的原图尺寸')


def _canonical(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def save_memory_bank(bank: MemoryBank, path: Path) -> None:
    """正常库 -> NPZ数组和JSON头；不保存pickle，不接受缺少来源签名的缓存。"""
    if not bank.provenance.get('reference_images') or not bank.provenance.get('model_sha256'):
        raise ValueError('缓存必须包含参考原图清单/哈希与模型签名')
    header = {'schema': 1, 'algorithm': bank.algorithm, 'provenance': bank.provenance,
              'source_hw': bank.source_hw}
    with Path(path).open('wb') as handle:
        np.savez_compressed(handle, features=bank.features.detach().cpu().numpy(), xy=bank.xy,
                            image_ids=np.asarray(bank.image_ids), descriptor_indices=bank.descriptor_indices,
                            header=np.asarray(_canonical(header)))


def load_memory_bank(path: Path, config: ExperimentConfig, expected_provenance: dict) -> MemoryBank:
    """缓存和本次期望的数据/模型来源 -> 经配置与签名检查的正常库。"""
    with np.load(path, allow_pickle=False) as payload:
        header = json.loads(str(payload['header'].item()))
        if header.get('schema') != 1:
            raise ValueError('不支持的参考库缓存版本')
        if not expected_provenance or _canonical(header['provenance']) != _canonical(expected_provenance):
            raise ValueError('参考库数据或模型来源签名不一致')
        bank = MemoryBank(torch.from_numpy(payload['features'].copy()), payload['xy'].copy(),
                          tuple(payload['image_ids'].tolist()), payload['descriptor_indices'].copy(),
                          tuple(None if hw is None else tuple(hw) for hw in header['source_hw']),
                          header['algorithm'], header['provenance'])
    validate_memory_bank(bank, config)
    return bank
