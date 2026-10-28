# InReaCh-new

重构 ICCV 2023 InReaCh，采用已批准的原尺度224×224、步长112重叠切块方案。旧版在 `../InReaCh-main/`，本目录继续保留整图256缩放/224中心裁剪预处理作为后续对照。

## 当前实现

**2026-10-28本阶段已完成并停下等待下一步确认。** 导出已实现；真实长跑按用户选择暂停，随后完成聚合等价优化，完整67项测试通过。真实完整原图覆盖率、导出验证及准确率对照仍未完成。详情见REFACTOR_LOG末节。

已接通原图切块、13层骨干、1024维描述子、跨批汇总、仅建库的位置归属、跨原图互近邻通道、通道筛选、正常库缓存、欧氏最近邻预测、原图融合、框级评估和显式清单入口。已验证的运行规模与结果见 `docs/REFACTOR_LOG.md`；实现完整链路不代表小缺陷准确率已改善。

| 模块 | 输入 → 输出与职责 |
| --- | --- |
| dataset / tiling | 原图清单与YOLO → 原图框、原尺度块与真实起点/有效区域 |
| feature_extraction | RGB归一化批次 → Wide ResNet-50-2 V1的13层特征 |
| descriptors | 多层特征 → 带原图身份、尺寸、坐标、有效性的描述子；跨批汇总及仅建库归属 |
| matching / association | 不同原图描述子 → 空间候选内的互近邻通道，每图每通道至多一项 |
| memory_bank | 候选通道 → 筛选后的正常库、来源与配置/数据/模型签名缓存 |
| prediction / stitching | 完整测试描述子 → 块分数/有效性 → 原图分数/有效性/最大分数 |
| localization / evaluation | 共用原图区域/峰值定位 → 一对一匹配、阈值曲线与指标 |
| visualization | 原图与分数/有效性 → 热力图、叠加图、定位标注PNG与JSON |
| pipeline | 明确的参考/验证/测试原图清单 → 库、原始NPZ、JSON报告 |

原图坐标为 `(x,y)`，尺寸/网格为 `(H,W)`，补边为 `(left,top,right,bottom)`。网格单元中心不是精确感受野中心。贴边块保存真实起点，不以网格下标代替原图坐标。

## 数学约定与旧版差异

- 输入为RGB/ImageNet归一化；切块路径不缩放。旧版BGR与整图预处理的差异仍存在。
- 描述子使用原层3×3局部邻域，先各层通道池化到1024，再双线性插值到同一公共网格，最后做层间1024维池化；不拼接旧版块内位置编码，空间规则使用独立的原图xy。
- 位置归属只供建库：离有效块边缘最远者拥有位置，平局取较小tile_id。预测保留全部重叠观测。
- 按参考清单顺序取前assoc_depth张为种子，仅与后续原图建立互近邻；默认10。半径默认8原图像素，要求两轴距离均不超过半径；None为无门控对照。空间门控要求已知且相同的原图尺寸；图像对齐本身仍依赖数据准备。
- 通道成员数在筛选前后均须严格大于min_channel_length，默认3。离散度沿用旧版到通道均值的欧氏距离的总体标准差，保留距离严格小于max_channel_std×标准差者，默认倍数5。完全一致的零半径通道保留；等距但不相同的通道仍可能被删空。
- 关联与预测统一欧氏距离，旧版FAISS预测为平方L2；旧阈值不能复用。无候选输出NaN/False，空正常库报错。
- 融合对有效分子/支持分别双线性插值（align_corners=False），使用严格正的二维三角中心窗，排除补边。默认不高斯平滑、不逐块min-max。
- 框定位：分数>=阈值，8连通区域各取一个峰值，平局行优先；定位点为像素中心(x+.5,y+.5)，落框采用半开边界。以最大基数一对一匹配计算召回，保留未匹配区域数和异常面积，不能解释为像素分割准确率。

以上阈值/门控参数尚未在正式划分上调优。无合格通道、无完整预测覆盖时会明确报告，不能以全量特征或正常零分回退。

## 运行

本机Python：`/home/g2080-2/miniforge3/envs/inreach/bin/python`。已有权重缓存；首次在其他机器加载骨干可能需要下载。

```bash
# 输入检查，不提取骨干
PYTHONDONTWRITEBYTECODE=1 python pipeline.py --inspect --limit 1

# 完整测试（本机建议限制CPU线程，减少小张量线程调度开销）
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v

# 用明确清单运行；输出目录必须尚不存在
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python pipeline.py --run --manifest my_split.json --output result/my_run --device cuda
```

清单是JSON对象，`reference`、`validation`、`test`各为原图记录数组。记录包含 `image_id`、`image_path`，可含 `label_path`、`group_id`；相对路径从清单所在目录解析。参考集至少需要 `min_channel_length+1` 张独立原图，标签不进入建库。参考集内部还会拒绝文件内容SHA-256相同的副本；三个集合间身份、路径、完全相同文件内容以及已知工件分组不能重叠；衍生图或近重复图需要提供真实group_id，代码无法从像素自动证明它们相互独立。

验证选阈值还需顶层字段 `annotations_complete: true`、`threshold_candidates: [数值...]` 与 `max_unmatched_regions_per_image: 数值`。在预算内最大化框召回，平局选更小异常面积，再选更高阈值。`split_confirmed: true`、完整工件分组和完整标注共同决定报告能否标为confirmed_split。未具备这些条件时报告为exploratory。

也可传入顶层 `fixed_threshold` 做探索性流程运行，此时不自动选阈值，始终标exploratory。这不是正式阈值来源。评估默认要求原图100%有效覆盖；未知标注不当正常图，pic1缺少正常图时图像AUROC为null。

每张原图预测完成后立即保存NPZ和覆盖报告。后续读图、验证阈值或严格评估失败时，异常继续抛出，`report.json`保留`status: incomplete`、`failed_stage`、错误和已完成产物；完整指标不成立时`test_metrics`为null。

模型必须提供`inreach_cache_config()`，返回包含正整数`version`的有限JSON对象。默认骨干工厂已提供；自定义模型需声明影响forward的全部非state_dict配置，例如普通`scale`、池化设置和外部依赖，子类改变前向时也应更新声明。框架验证格式并把声明加入权重/结构摘要，不自动证明声明完整。未声明的模型直接拒绝建库和预测。新签名版本会拒绝旧缓存，需要在新结果目录重建。

程序接口：`extract_record_descriptors`、`fit_reference_bank`、`predict_records`、`run_experiment`。正常库缓存用NPZ+JSON，加载必须传入本次期望的配置与数据/模型签名。原始预测NPZ保留score_map、valid_map与image_id，JSON报告含参数、阈值曲线、清单哈希、耗时和内存统计。

逐原图提取后仅保留建库归属行；默认参考特征预算2048MiB，超出则拒绝，不隐式降采样。这个预算仅约束累计参考特征张量，不是进程RSS硬上限；通道和最终库还需要额外内存。当前实现适于小样本验证，全量158张的流式库/性能优化仍待完成。`--no-spatial-gate`为全局搜索对照，计算量可能明显增加。


## 热力图与定位输出

每个完成的测试图在原始NPZ之外导出：

| 文件后缀 | 内容 |
| --- | --- |
| `_heatmap.png` | 与原图同尺寸的TURBO彩色分数图 |
| `_overlay.png` | 热力图和RGB原图叠加，默认alpha=0.5 |
| `_localization.png` | 原图上标注阈值区域红框和峰值黄十字，仅在阈值已确定时生成 |
| `_detections.json` | 原图身份/尺寸、覆盖率、阈值、区域峰值坐标/原始分数/面积/半开外接框及显示设置 |

无有效预测区域统一显示灰色(RGB128)，不能当正常低分。区域框表示阈值化区域，不是真实缺陷框；峰值距离不是概率。阈值未选出时JSON的regions为null，状态为threshold_not_selected。部分覆盖仍可导出已有定位证据，完整指标继续拒绝计算。

运行共用一个显示范围，默认[0,验证集有效分数最大值]，无正有效最大值时[0,1]。测试图不参与确定色标。也可在清单中显式设置：

```json
"visualization": {"score_range": [0, 10], "overlay_alpha": 0.5}
```

这里的0到10只演示配置格式，不是已验证参数。显示范围的裁剪和颜色变换仅作用于PNG，NPZ原始分数、阈值和评估保持原值。范围及来源记录在report.json，避免把不同色标的图片直接作效果比较。

已保存NPZ可离线导出，无需重新提取骨干：

```bash
PYTHONDONTWRITEBYTECODE=1 python visualization.py \
  --image /path/to/original.png \
  --prediction /path/to/test_0000.npz \
  --output result/new_export \
  --threshold 4 --score-min 0 --score-max 10 --alpha 0.5
```

阈值4也是格式示例，实际应使用本次验证确定的阈值。省略--threshold仅导出显示图和“定位未就绪”JSON。原图文件stem应与NPZ的image_id一致，若原图使用别名，可通过--image-id显式核对。输出目录必须尚不存在。

验证集所有预测完成后会先导出显示图，阈值选择成功后补定位；阈值失败也保留显示诊断。若验证预测过程的后续图读入失败，共用色标尚未确定，已经完成的NPZ仍保留，PNG可能尚未生成。测试图则逐张保存完整产物。

直接调用`export_prediction`时，若要把已设置的阈值撤回为未设置，请换结果目录或文件前缀；当前这种状态回退不会删除旧定位PNG。标准pipeline和CLI没有这一路径。

`result/export-preview-20261028`是离线合成预览，图片明确标注SYNTHETIC DEMO，仅验证显示与坐标，不是真实检测结果。

## 聚合优化的验证边界

2026-10-28已将原层通道池化提前到空间插值之前，两步分别作用于不同轴，在实数下等价。测试固定atol=1e-6、rtol=1e-5，浮点结果不承诺逐位一致。真实骨干单块13层本次79.40秒→10.82秒，最大绝对误差约8.94e-8，坐标/网格/valid完全一致；这是单次单块计时，未做预热重复统计，不能当完整运行加速倍率或检测准确率。

描述子签名已升级v2，旧聚合正常库/描述子缓存应在新目录重建。当前67项测试通过；小型通道压缩/下采样专项回归及旧descriptor签名专项失配用例仍待完善。此前真实6图长跑已按用户选择停止，未产生真实覆盖率/检测图；后续完整原图重跑需再次确认。

## 数据与未完成项

本地 `data/pic1` 按用户授权复制158张图、158个YOLO文件、300个缺陷框，源文件不变。全部图均有缺陷框，不能用此子集算正常/异常两类图像AUROC。8个框的六位小数边界误差在1e-6归一化容差内告警裁回，较大越界报错。缺标签表示unknown；空标签仅在确认完整标注后代表正常。

用户于2026-10-26确认目前没有正式参考/验证/测试清单、工件分组及标注完整性约定。尚未完成正式固定划分对照、整图基线检测器对照、小缺陷指标改善验证，以及全量158张吞吐/峰值资源验证。

设计与选择：`docs/superpowers/specs/2026-10-23-overlapping-tiles-design.md`、`docs/superpowers/plans/2026-10-26-algorithm-completion.md`。后续交接以 `docs/AI_HANDOFF.md` 和最新 `docs/REFACTOR_LOG.md` 为准。
