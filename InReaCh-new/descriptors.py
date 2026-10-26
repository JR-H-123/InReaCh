"""将多层局部特征汇成描述子, 同时保留每个描述子的原图坐标。"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from config import ExperimentConfig
from contracts import DescriptorSet, TileMeta


PRETRAIN_DIM = 1024
TARGET_DIM = 1024
PATCH_SIZE = 3


def _pooled_level(feature: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """单块 (C,Hf,Wf) -> (目标网格块数,1024), 对应旧版3×3邻域与一维池化。"""
    _, h, w = feature.shape
    # 在原层先取3×3邻域; 边缘补零与旧版 Unfold(padding=1) 一致。
    patches = F.unfold(feature.unsqueeze(0), kernel_size=PATCH_SIZE, padding=1)
    patches = patches.reshape(1, -1, h, w)
    # 旧版先展开邻域再将每个分量双线性插值到最浅层网格。
    if (h, w) != target_hw:
        patches = F.interpolate(patches, size=target_hw, mode='bilinear', align_corners=False)
    vectors = patches.squeeze(0).permute(1, 2, 0).reshape(-1, 1, patches.shape[1])
    return F.adaptive_avg_pool1d(vectors, PRETRAIN_DIM).squeeze(1)


def build_descriptors(features: dict[str, torch.Tensor], metadata: tuple[TileMeta, ...],
                      config: ExperimentConfig) -> list[DescriptorSet]:
    """批次多层特征 -> 按 image_id 分组的 (P,1024) 描述子和原图 (x,y) 坐标。

    使用输入字典的特征层顺序, 即现有骨干的 Level_1..Level_13 顺序。
    以第一层真实 (Hf,Wf) 为公共网格, 不用 sqrt 猜形状。
    坐标是网格单元中心, 不是卷积感受野的精确中心。
    补边位置保留于输出但 valid=False, 方便预测阶段回填网格。
    每批单独返回; 跨批汇总及仅建库的位置归属留给后续步骤。
    """
    if not features or not metadata:
        raise ValueError('特征和图块元数据不能为空')
    batch_size = len(metadata)
    for name, value in features.items():
        if value.ndim != 4 or value.shape[0] != batch_size or min(value.shape[1:]) <= 0:
            raise ValueError(f'{name} 应为 (B,C,Hf,Wf), B={batch_size}')
    names = list(features)
    ref_hw = tuple(features[names[0]].shape[-2:])
    if any(meta.valid_hw[0] <= 0 or meta.valid_hw[1] <= 0 or
           meta.valid_hw[0] > config.tile_size or meta.valid_hw[1] > config.tile_size
           for meta in metadata):
        raise ValueError('图块有效区域与 tile_size 不一致')
    grouped: dict[str, list[tuple[torch.Tensor,np.ndarray,np.ndarray,np.ndarray,int]]] = OrderedDict()
    h_ref, w_ref = ref_hw
    local_x = (np.arange(w_ref, dtype=np.float64) + .5) * config.tile_size / w_ref
    local_y = (np.arange(h_ref, dtype=np.float64) + .5) * config.tile_size / h_ref
    xx, yy = np.meshgrid(local_x,local_y,indexing='xy')
    for index, meta in enumerate(metadata):
        pooled = []
        for name in names:
            # 骨干可以留在 GPU; 单块特征移到 CPU 做局部聚合, 控制显存峰值。
            pooled.append(_pooled_level(features[name][index].detach().float().cpu(), ref_hw))
        joined = torch.stack(pooled,dim=1).reshape(h_ref*w_ref,1,-1)
        embedding = F.adaptive_avg_pool1d(joined,TARGET_DIM).squeeze(1)
        xy = np.column_stack((xx.ravel()+meta.origin_xy[0],yy.ravel()+meta.origin_xy[1]))
        valid = ((xx.ravel() < meta.valid_hw[1]) & (yy.ravel() < meta.valid_hw[0]) &
                 (xy[:,0] < meta.original_hw[1]) & (xy[:,1] < meta.original_hw[0]))
        ids = np.full(h_ref*w_ref,meta.tile_id,dtype=np.int64)
        grouped.setdefault(meta.image_id,[]).append((embedding,xy,ids,valid,meta.tile_id))
    return [DescriptorSet(torch.cat([part[0] for part in parts],dim=0),
                          np.concatenate([part[1] for part in parts]),image_id,
                          np.concatenate([part[2] for part in parts]),
                          {part[4]:ref_hw for part in parts},
                          np.concatenate([part[3] for part in parts]))
            for image_id,parts in grouped.items()]


def _check_descriptor_set(item: DescriptorSet) -> None:
    """检查逐行字段与每块网格, 避免汇总时悄悄错位。"""
    n = item.features.shape[0]
    if (item.features.ndim != 2 or n == 0 or
        item.xy.shape != (n, 2) or item.tile_ids.shape != (n,) or
        item.valid.shape != (n,) or item.valid.dtype != np.bool_ or
        not np.isfinite(item.xy).all() or not torch.isfinite(item.features).all()):
        raise ValueError(f'{item.image_id}: 描述子逐行字段无效或错位')
    actual = set(int(t) for t in np.unique(item.tile_ids))
    if actual != set(item.grid_hw):
        raise ValueError(f'{item.image_id}: tile_ids 与 grid_hw 不一致')
    for tile_id, (h, w) in item.grid_hw.items():
        if h <= 0 or w <= 0 or np.count_nonzero(item.tile_ids == tile_id) != h*w:
            raise ValueError(f'{item.image_id}: tile {tile_id} 描述子数与网格不一致')


def merge_descriptor_sets(fragments: list[DescriptorSet]) -> list[DescriptorSet]:
    """将多个批次片段按原图合并, 保留输入批次/块/网格的原顺序。

    不合并不同图像, 不把同一图块的重复片段计为两个独立样本。
    """
    if not fragments:
        raise ValueError('没有可汇总的描述子片段')
    groups: dict[str, list[DescriptorSet]] = OrderedDict()
    dims: dict[str, int] = {}
    seen: dict[str, set[int]] = {}
    for item in fragments:
        _check_descriptor_set(item)
        if item.image_id in dims and item.features.shape[1] != dims[item.image_id]:
            raise ValueError(f'{item.image_id}: 描述子维度不一致')
        dims[item.image_id] = item.features.shape[1]
        ids = set(item.grid_hw)
        if ids & seen.setdefault(item.image_id, set()):
            raise ValueError(f'{item.image_id}: 重复 tile_id')
        seen[item.image_id].update(ids)
        groups.setdefault(item.image_id, []).append(item)
    return [DescriptorSet(torch.cat([p.features for p in parts],dim=0),
                          np.concatenate([p.xy for p in parts]),image_id,
                          np.concatenate([p.tile_ids for p in parts]),
                          {key:hw for p in parts for key,hw in p.grid_hw.items()},
                          np.concatenate([p.valid for p in parts]))
            for image_id,parts in groups.items()]


def select_reference_descriptors(descriptors: list[DescriptorSet], metadata: tuple[TileMeta, ...]) -> list[DescriptorSet]:
    """只为建库保留所属块的有效描述子, 预测仍使用未筛选的输入。

    每个原图坐标可由多个重叠块覆盖。选择该坐标离有效块边缘
    最远的覆盖块, 平局由较小 tile_id 决定。使用真实起点处理贴边块。
    """
    if not descriptors or not metadata:
        raise ValueError('描述子和完整图块元数据不能为空')
    metas: dict[str, dict[int, TileMeta]] = {}
    for meta in metadata:
        group = metas.setdefault(meta.image_id,{})
        if meta.tile_id in group:
            raise ValueError(f'{meta.image_id}: 重复 tile_id 元数据')
        group[meta.tile_id] = meta
    selected = []
    seen_images = set()
    for item in descriptors:
        _check_descriptor_set(item)
        if item.image_id in seen_images:
            raise ValueError(f'{item.image_id}: 请先跨批汇总, 不能分别筛选')
        seen_images.add(item.image_id)
        group = metas.get(item.image_id)
        if group is None or set(group) != set(item.grid_hw):
            raise ValueError(f'{item.image_id}: 必须提供全部且仅有本图块的元数据')
        shapes = {m.original_hw for m in group.values()}
        if len(shapes) != 1:
            raise ValueError(f'{item.image_id}: 原图尺寸冲突')
        # 浮点网格相位可能不同; 逐坐标比较覆盖块, 不用展平索引去重。
        best_id = np.full(len(item.xy),-1,dtype=np.int64)
        best_margin = np.full(len(item.xy),-np.inf,dtype=np.float64)
        for tile_id in sorted(group):
            meta = group[tile_id]
            x0,y0 = meta.origin_xy
            width,height = meta.valid_hw[1],meta.valid_hw[0]
            dx,dy = item.xy[:,0]-x0,item.xy[:,1]-y0
            covered = (dx >= 0) & (dy >= 0) & (dx < width) & (dy < height)
            margin = np.minimum.reduce((dx,dy,width-dx,height-dy))
            better = covered & (margin > best_margin)
            best_margin[better] = margin[better]
            best_id[better] = tile_id
        keep = item.valid & (item.tile_ids == best_id)
        if not np.any(keep):
            raise ValueError(f'{item.image_id}: 没有归属到有效原图区域的参考描述子')
        retained_ids = set(int(t) for t in np.unique(item.tile_ids[keep]))
        selected.append(DescriptorSet(item.features[keep],item.xy[keep],item.image_id,
                                      item.tile_ids[keep],
                                      {k:v for k,v in item.grid_hw.items() if k in retained_ids},
                                      item.valid[keep]))
    if set(metas) != seen_images:
        raise ValueError('元数据含未提供描述子的图像')
    return selected
