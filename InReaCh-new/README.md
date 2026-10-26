# InReaCh-new

本目录用于按 ICCV 2023 论文逐步重写 InReaCh。原项目代码位于 `../InReaCh-main/`。

## 当前进度

当前实现了**论文实验中的图像尺寸处理和原版骨干特征提取**：读取任意图像路径，将 OpenCV 读出的 BGR 转为 RGB，缩放到 `256×256`，中心裁剪到 `224×224`，再按 ImageNet RGB 均值和标准差转换为 `(1, 3, 224, 224)` 张量。使用 Wide ResNet-50-2 的 ImageNet V1 预训练权重，提取原版实际运行配置中的 13 个节点。`pic1` 目前仅用来检查单图流程；代码没有数据集专用路径或 YOLO 框处理。

| 文件 | 用途 |
| --- | --- |
| `image_processing.py` | 图像读取、缩放裁剪、模型张量转换。 |
| `feature_extraction.py` | 加载预训练骨干并返回 13 层特征图。 |
| `tests/test_image_processing.py` | 检查裁剪位置、张量形状、通道归一化和路径读取。 |
| `tests/test_feature_extraction.py` | 检查原版节点顺序、特征形状和推理模式。 |
| `docs/REFACTOR_LOG.md` | 记录每次实际修改与验证结果。 |
| `docs/ROADMAP.md` | 后续重构步骤。 |

## 使用

```python
from image_processing import load_image, preprocess_image, to_model_input
from feature_extraction import load_feature_model, extract_features
import torch

image = load_image("/path/to/image")       # NumPy RGB 图像，(H, W, 3)
image = preprocess_image(image)             # (224, 224, 3)
model_input = to_model_input(image)         # PyTorch 张量，(1, 3, 224, 224)
model = load_feature_model(torch.device("cpu"))
features = extract_features(model, model_input)  # Level_1 至 Level_13 的特征图
```

当前使用 RGB 顺序的 ImageNet 均值 `[0.485, 0.456, 0.406]` 和标准差 `[0.229, 0.224, 0.225]`。原项目直接处理 BGR 并使用不同的均值顺序，因此本版特征值会与原版不同；参考图和测试图应统一使用当前流程。预训练权重首次使用时由 torchvision 下载，已有缓存时直接读取。当前仅实现骨干特征提取，尚无描述子、通道关联或预测代码。

在本目录运行测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v
```

## 原尺度重叠切块框架（2026-10-23）

新增方案 A：原图切成 `224×224`、步长 `112`，不缩放、不中心裁剪。
已有整图预处理与骨干实现保持原样，可作为后续对照。

已实现：pic1 图片发现、YOLO 框解析、原尺度切块、贴边完整覆盖、小图补边、RGB 归一化批次与输入检查入口。
待实现：多层描述子、建库位置归属、跨图通道筛选、正常参考库、异常预测、热图融合及评估。
待实现入口均抛出 `NotImplementedError`；当前不能输出检测结果，也未验证小缺陷检测改善。

### 数据

按用户要求从 `/home/g2080-2/Project/data/all` 原样复制 pic1：

```text
data/pic1/
  images/               # 158 张图
  labels/               # 158 个 YOLO 检测框 txt
  copy_manifest.json    # 来源、相对路径、字节数与 SHA-256
```

300 个框全部表示缺陷（用户确认），158 张图均有框。因此当前 pic1 的图片级污染率是 100%，并非之前估计的 40%；不能仅用该子集计算正常/异常图像级 AUROC。
尚未创建正式参考/验证/测试划分。按原图（同工件近重复图还应按工件）划分后才能正式评估。
8 个边界框存在六位小数舍入误差，解析时在 `1e-6` 容差内告警并裁回原图边界；原 txt 不变，更大越界仍报错。
缺失标签表示未知，空标签只有在确认标注完整后才表示正常。

### 模块阅读顺序

| 模块 | 用途与状态 |
| --- | --- |
| `config.py` | 已实现：数据目录、块尺寸、步长、批大小、未来空间门控半径 |
| `contracts.py` | 框、图块元数据、特征、通道、参考库、预测的类型契约 |
| `dataset.py` | 已实现：原图发现与原图坐标系 YOLO 框读取 |
| `tiling.py` | 已实现：完整覆盖的原尺度切块及有效区域 |
| `pipeline.py` | 已实现：`prepare_batches` 与输入检查 CLI |
| `feature_extraction.py` | 复用：现有骨干与 13 层特征提取 |
| `descriptors.py` | 接口：提取描述子；单独选择建库描述子，推理保留重叠观测 |
| `association.py` | 接口：不同原图的互近邻与候选通道 |
| `memory_bank.py` | 接口：筛选正常通道、建索引 |
| `prediction.py` | 接口：局部异常分数与无候选有效性 |
| `stitching.py` | 接口：原图分数融合 |
| `evaluation.py` | 接口：基于原图框的定位评估 |

原图坐标为 `(x,y)`，尺寸为 `(H,W)`，补边为 `(left,top,right,bottom)`。
`prepare_batches` 不读取标签，且不调用整图基线的 `preprocess_image`。
数据结构是框架契约，算法阶段还需要补充阈值配置、缓存与契约校验，不能直接视为完整检测器。

### 运行输入检查

在本目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/g2080-2/miniforge3/envs/inreach/bin/python pipeline.py --inspect --limit 1
```

对当前 `847×711` 图像产生 42 块；默认每批 4 块，张量为 `(4,3,224,224)`，末批保留不足 4 块的剩余输入。
可通过 `--data-root`、`--batch-size`、`--limit` 调整输入检查。此命令不加载骨干或参考库。

```python
from config import ExperimentConfig
from dataset import discover_images
from pipeline import prepare_batches

config = ExperimentConfig()
record = discover_images(config.data_root)[0]
for batch in prepare_batches(record, config):
    print(batch.inputs.shape, batch.metadata[0].origin_xy)
    # 下一步可传给已有 extract_features; 描述子与检测接口尚未实现。
```

设计见 `docs/superpowers/specs/2026-10-23-overlapping-tiles-design.md`，实施计划见 `docs/superpowers/plans/2026-10-23-overlapping-tiles-framework.md`。

### 多层描述子（2026-10-23）

`build_descriptors(features, metadata, config)` 已实现：按骨干返回顺序从每层特征取 3×3 邻域，对齐到最浅层真实网格；每层池化到 1024 维，再把所有层池化为 1024 维描述子。输出 `DescriptorSet`，保存每个描述子的原图 `(x,y)` 浮点坐标、tile_id、真实网格和 valid 标志。补边网格位置保留在输出以便以后恢复热图，但标记为无效。

```python
from descriptors import build_descriptors
from feature_extraction import extract_features

for batch in prepare_batches(record, config):
    layer_maps = extract_features(model, batch.inputs)
    descriptor_sets = build_descriptors(layer_maps, batch.metadata, config)
```

这里的 `(x,y)` 是网格单元中心约定，不是卷积感受野的精确中心。当前输出不附加旧版按块内坐标计算的两维位置编码；后续空间门控应使用保存的原图坐标。`build_descriptors` 按批次输出，跨批汇总和仅用于建库的位置归属仍待实现，不能直接交给通道构建。

### 跨批汇总与建库归属

```python
from descriptors import merge_descriptor_sets, select_reference_descriptors

parts = []
all_metadata = []
for batch in prepare_batches(record, config):
    parts.extend(build_descriptors(extract_features(model, batch.inputs), batch.metadata, config))
    all_metadata.extend(batch.metadata)
whole_image = merge_descriptor_sets(parts)
reference_descriptors = select_reference_descriptors(whole_image, tuple(all_metadata))
# 预测阶段使用 whole_image，保留重叠观测供分数融合。
```

`select_reference_descriptors` 要求传入该原图的全部块元数据，按真实原图坐标选择所属块，不把贴边块当成规则网格。输出为稀疏参考描述子，`grid_hw` 仍记录原块网格尺寸；不能把筛选后的行数当作完整块网格恢复热图。
