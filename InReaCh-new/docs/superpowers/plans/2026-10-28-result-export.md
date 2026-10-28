# 原图热力图与定位导出实施计划

> **For agentic workers:** 使用executing-plans逐项实现，测试先失败再通过。本阶段结束后按用户约定停止，更新说明并询问下一步。

**Goal:** 完成用户已确认的热力图、原图叠加图、定位点清单和原图定位标注导出。
**Architecture:** localization.py统一8连通区域/峰值提取供评估与导出复用；visualization.py只映射显示颜色、写PNG/JSON并提供离线NPZ导出CLI；pipeline.py接入逐图输出和共享显示范围。
**Tech Stack:** 现有NumPy、OpenCV、unittest，不安装依赖。
**Spec:** ../specs/2026-10-23-overlapping-tiles-design.md；本轮用户在明确“先补导出、测试后停下，真实原图运行另行确认”后回复“继续”。

## Global Constraints

- 坐标(x,y)，尺寸(H,W)，输出保持原图尺寸/身份与valid；补边不构成证据。
- score>=threshold、8连通、峰值平局行优先、像素中心(x+.5,y+.5)，与已实现评估一致。
- 显示颜色不改变原始分数、阈值或指标，不按块归一化。
- 只写新运行的结果目录，不修改原图、旧结果；本轮不运行真实全图模型或正式准确率实验。

## Implementation choices

- PNG使用TURBO颜色表，无效区域RGB(128,128,128)，叠加alpha默认0.5；颜色界限裁剪仅影响显示。
- 显示范围可显式给定；默认使用[0,验证集有效分数最大值]并供两集合共用。没有正的有效最大值时使用[0,1]，记录来源，不据此判正常。
- 定位JSON保存原图身份、尺寸、覆盖率、阈值、原始峰值分数、区域面积、区域外接框（半开xyxy）和峰值坐标。外接框表示阈值区域，不解释为真实缺陷框；不提供概率置信度。
- 验证集预测完成后先保存可视化；阈值选择成功后补定位标注。阈值未选定时regions为null，不伪装成“没有缺陷”。测试图逐图保存全部产物；缺覆盖仍保留诊断输出并拒绝完整指标。

## Review Focus

- 非正方形尺寸、像素中心与半开框不能偏移。
- 无效NaN不得污染色标或参与区域，常量/全无效图不得除零。
- PNG通道必须RGB/BGR一致；原图/原分数不得修改。
- 跨图片显示比例必须共用，测试极大值不得反向改变验证色标。
- 阈值失败、部分覆盖时已有NPZ/PNG/报告仍保留；定位未就绪状态明确。

### Task 1: 公共区域提取

- [x] tests/test_localization.py：手算对角连通、峰值平局、边界、面积/框、无效高分；运行确认缺模块失败。
- [x] localization.py提供extract_regions(Prediction, threshold)->list[dict]；evaluation复用，运行原有评估测试。

### Task 2: 图像与清单导出

- [x] tests/test_visualization.py：固定色标、常量/全无效、RGB写读、显示裁剪不影响定位、无阈值与坏参数；先失败。
- [x] visualization.py提供export_prediction和离线CLI，保存heatmap/overlay/localization PNG及detections JSON，运行上述测试。

### Task 3: 流程接入与验收

- [x] 扩展流程测试，验证完整输出、共享色标、阈值/覆盖失败产物保留，先失败。
- [x] 接入run_experiment和JSON清单visualization设置，保留原始NPZ与错误报告。
- [x] 全套unittest、离线合成预览CLI、补丁检查；记录实际结果，更新README/AI_HANDOFF/REFACTOR_LOG后停止。

实施验证：66项unittest通过12.199秒；离线合成CLI退出0，预览人工查看通过。最终只读审阅结果见REFACTOR_LOG后续补记。真实完整图像模型运行未启动。

最终审阅：无Critical/Important，独立28项相关测试通过；1项仅直接API阈值回退的旧PNG残留Minor延期，详情见日志。阶段停止，下一步等待用户确认。
