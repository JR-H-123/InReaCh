# InReaCh 重构：交给下一位 AI 的实施说明

更新时间：2026-10-23。本文记录交接状态，不代表完整检测器已经完成。后续进度以本文末尾增量记录及最新REFACTOR_LOG为准。

## 先读什么

1. `AGENTS.md`：项目范围、初学者讲解方式和验证要求。
2. 本文件：当前完成状态、下一步与尚未冻结的细节。
3. `docs/superpowers/specs/2026-10-23-overlapping-tiles-design.md`：用户已认可的完整方向；历史40%估计以文末数据补充为准。
4. `contracts.py`、`config.py` 与相关模块：实际接口。
5. `docs/REFACTOR_LOG.md`：按时间追加的实测记录，后面的记录优先说明当前状态。

`docs/superpowers/plans/2026-10-23-overlapping-tiles-framework.md` 是已完成的框架阶段计划，不是剩余算法的完整实施计划。README/ROADMAP中保留了历史阶段说明，不能将早期“尚未实现描述子”误当作当前状态。

## 用户目标与已确认条件

- 重构 ICCV 2023 InReaCh，改善原图约4×7像素、最窄约3像素缺陷的漏检。
- 图像已对齐，估计残余偏移2–4像素；这一偏移的测量口径尚未确认。
- 原尺度224×224重叠切块、步长112，不先缩小整图。
- 参考数据允许混有缺陷；YOLO标签只用于验证，不参与正常库筛选。
- 当前数据为 `data/pic1/images` 与 `labels`：158张847×711图片，158个txt，300个框。
- 用户已确认全部框表示缺陷，且每张图都有框。因此该子集图片级缺陷率为100%，取代早期40%估计；不能计算正常/异常两类图像级AUROC。
- 尚未指定正式参考、验证、测试清单及同工件分组。实现算法可以继续，不能自行把全部图片同时建库、测试后声称泛化效果。

## 已实现与待实现

| 模块 | 当前状态 |
| --- | --- |
| image_processing.py | 已实现RGB读取、整图基线缩放裁剪、ImageNet归一化 |
| feature_extraction.py | 已实现Wide ResNet-50-2 V1与13个指定节点 |
| dataset.py | 已实现pic1发现、YOLO框转换；缺失标签为unknown |
| tiling.py | 已实现原尺度切块、末块贴边、小图补边 |
| config.py / contracts.py | 基础参数与数据类型已建立；后续仍需补配置和验证 |
| pipeline.py | 仅实现prepare_batches和--inspect输入检查 |
| descriptors.py:build_descriptors | 已实现3×3局部邻域、网格对齐、两次1024维池化、原图坐标与有效性 |
| descriptors.py:select_reference_descriptors | 占位，尚未实现 |
| association.py | 跨图关联占位 |
| memory_bank.py | 通道筛选与正常索引占位 |
| prediction.py | 描述子异常分数占位 |
| stitching.py | 原图热图融合占位 |
| evaluation.py | 框级评估占位 |

占位入口均抛出NotImplementedError，不得用常量、随机分数或所有输入特征代替真实算法。

## 必须保持的接口语义

- 图像(H,W,C)，张量(B,C,H,W)，网格(Hf,Wf)，坐标(x,y)。不能用sqrt(块数)恢复网格。
- `TileMeta.origin_xy` 是真实切块起点。末块贴边可能使网格相位不同，不得假定所有块坐标是步长整数倍。
- 当前tiling只在右/下补边；`valid_hw`保存有效区域，`padding`顺序为left,top,right,bottom。
- `DescriptorSet.features`为(P,1024) CPU张量，xy为(P,2)原图浮点坐标；tile_ids、valid与行一一对应。
- 每个块按行优先展开。网格单元中心公式：x=origin_x+(col+0.5)*tile_size/Wf，y同理；这不是精确感受野中心。
- `build_descriptors`使用字典的层顺序，现有骨干返回Level_1至Level_13；第一层是真实参考网格。
- 当前描述子不拼接旧版块内两维位置编码，后续坐标约束使用独立xy；这一差异必须记录。
- `build_descriptors`每次只处理一个批次。返回同一image_id的分组片段，不代表整图已经汇总。
- 构建参考库时才应用重叠位置归属；预测必须保留全部有效重叠观测供融合。
- 无参考候选返回无效标志，不能作为正常零分；空正常库明确报错。
- 标签原文件不改。8个框有六位小数舍入越界，现有读取器仅在1e-6归一化容差内告警裁回。

## 推荐实施顺序与验收

1. **跨批汇总**：增加按image_id合并DescriptorSet的函数。保持features/xy/tile_ids/valid同步，合并grid_hw；重复tile_id、维度不一致、元数据冲突必须报错。用不同批次拆分验证合并结果与一次处理一致。
2. **建库位置归属**：实现select_reference_descriptors。输入必须包含同一原图的全部块元数据；对于每个描述子坐标，在覆盖它的块中选离最近边缘最远者，平局按tile_id。仅保留属于当前块且valid的参考描述子。用非规则末块起点、交界平局及补边做测试。
3. **跨图关联**：先核对旧版gen_assoc/gen_channels；以独立原图为单位计算候选和互近邻，排除同图，每通道每图最多贡献一次。空间门控为可选，两轴坐标差均不超过radius_px；没有候选不放宽约束。先用可手算小特征验证，再做真实小样本。
4. **通道筛选和正常库**：按独立原图支持数与离散程度筛选。保持旧版球面距离离散度的语义，明确阈值边界；空通道不能退回全部特征。保留坐标及足够的缓存元信息。
5. **预测与热图**：检索与建库采用一致的空间规则和距离单位。恢复真实特征网格，再用align_corners=False对应的坐标约定映射回图块；正权重融合回原图，排除补边，输出覆盖有效性。不得逐块min-max归一化再融合。
6. **评估与流程入口**：按设计文档实现8连通区域峰值落框的一对一匹配、框召回、未匹配区域/图及异常覆盖面积；阈值由独立验证集确定。未确认完整标注时称“未匹配”而不是确定误报。接通建库、预测、评估入口，再运行固定清单对照。

## 尚未完全冻结的细节

整体框架已定，但以下不能声称已有最终答案。下一位AI应核对旧版，给出有依据的实现选择、加配置和测试，并记录差异：

- 关联种子选择与assoc_depth、通道长度阈值、max_channel_std；现有ExperimentConfig还没有这些字段。
- 阈值使用严格大于还是大于等于、标准差为零时的处理、多个种子冲突规则。
- 统一使用欧氏还是平方L2距离，需同时核对筛选与预测，不能直接混用FAISS输出。
- MemoryBank缓存格式、数据/权重/配置签名与失配拒绝规则；现有类型仅有features和xy。
- 热图中心权重的具体函数、无效网格插值规则；需要保证无效分数不污染有效区域。
- 性能方案：目前逐块CPU聚合只做过真实单块检查，尚未测全图42块。按原图处理或流式缓存，避免一次把158张全部描述子堆入内存。优化应验证数值等价，不默默下采样。
- 参考/验证/测试清单、同工件分组、标注完整性、阈值选取目标，需要真实数据背景；不要用未经确认的假设声称正式准确率。

这些项不阻塞跨批汇总和位置归属实现，但完整端到端交付前必须落实。

## 原版参考与环境

旧版在 `../InReaCh-main/`，重点读取 `FeatureDescriptors.py`、`InReaCh.py`、`model.py`、`utils.py`、`mvtec_loader.py`。不要修改旧版或把Online-InReaCh混进来。
当前可用Python：`/home/g2080-2/miniforge3/envs/inreach/bin/python`。不要按旧requirements/conda文件直接重装环境；git命令此前不可用。

已有验证：16项unittest通过；真实pic1_00006首块经13层骨干得到(3136,1024)描述子、56×56网格，坐标从(2,2)到(222,222)，全部有限。这些不是检测精度验证，也未证明聚合与原版逐元素数值等价。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 /home/g2080-2/miniforge3/envs/inreach/bin/python pipeline.py --inspect --limit 1
```

## 可以直接发给下一位AI的指令

请在InReaCh-new内继续实现已批准的重构方案。先阅读AGENTS.md、docs/AI_HANDOFF.md、设计文档和最新REFACTOR_LOG，再核对实际代码。先完成跨批描述子汇总与仅建库的位置归属，按上文顺序逐步完成其余算法。每一步先用中文说明输入、输出和目的，再编写与运行必要测试，记录实际结果。保持原图身份、坐标与有效性贯穿全流程，不把重叠块当成独立图像。算法细节未定处对照旧版并记录选择；不要声称占位实现已经完成或用热图外观代替准确率。完成后明确报告已经实现、已验证和尚未完成的内容。

若交给另一台机器上的AI，须提供整个InReaCh-new源码/测试/文档，以及旧版参考文件；本机绝对路径不能供远程AI访问。数据与权重可单独提供，但真实数据验证需要能访问它们。

## 2026-10-23 增量进度：跨批汇总与建库位置归属

`descriptors.py` 现已实现 `merge_descriptor_sets(fragments)` 与 `select_reference_descriptors(descriptors, metadata)`。前者按image_id汇总各批，保持逐行字段同步，拒绝重复tile_id、维度不一致与字段错位；后者只用于建库，根据真实图块位置与有效区域选择离最近块边缘最远的覆盖块，平局取较小tile_id。预测仍应使用未筛选的描述子。

原表格中 `select_reference_descriptors` “尚未实现”和推荐顺序的第1、2步已由本节取代。下一步从第3步跨图关联开始；通道、参考库、预测、融合和评估仍未实现。具体测试与结果见最新REFACTOR_LOG。
