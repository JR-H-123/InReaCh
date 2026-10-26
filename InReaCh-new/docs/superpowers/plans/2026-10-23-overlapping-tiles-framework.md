# Overlapping Tiles Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 建立可运行的 pic1 数据准备入口和有明确输入输出的 InReaCh 算法框架；本轮不实现完整检测器。
**Architecture:** 复用现有 RGB 读取与骨干；新增配置、数据契约、YOLO 解析、切块和批量准备，算法阶段显式抛出 NotImplementedError。
**Tech Stack:** Python 3.12、NumPy、OpenCV、PyTorch、unittest；不安装新依赖。
**Spec:** ../specs/2026-10-23-overlapping-tiles-design.md（用户已确认）

## Global Constraints

- 原尺度 224×224 切块、步长 112；坐标 (x,y)、尺寸 (H,W)。
- 数据按用户最新指示复制到 data/pic1；不修改源数据。
- 建库接口不接收标签；未知标注不作为正常标注。
- 原版代码与已实现的预处理、骨干行为保持不变。
- 本轮交付框架与数据输入，通道、描述子、预测融合、评估只定义接口，后续逐步实现。

## Review Focus

- 已存在目标文件内容不同：复制前检查并拒绝覆盖。
- 贴边块不在规则网格上：保存真实起点，覆盖非整除图像。
- 小图补边：有效区域只包含原图，元数据明确。
- 缺失/非法 YOLO：缺失返回 unknown，非法报路径与行号。
- 未完成检测接口：必须抛出 NotImplementedError，不生成伪结果。

### Task 1: 复制与数据契约

**Files:** data/pic1/{images,labels,copy_manifest.json}, config.py, contracts.py, dataset.py, tests/test_tiled_input.py。
**Interfaces:** ExperimentConfig(data_root, tile_size=224, stride=112, batch_size=4, radius_px=8); ImageRecord(image_id,image_path,label_path); Box(class_id,x1,y1,x2,y2); discover_images(root,prefix='pic1_')->list[ImageRecord]; load_yolo_boxes(path,image_hw)->tuple[Box,...]|None。
- [x] 复制 158 对文件并逐项 SHA-256 核验，不覆盖不同内容目标。
- [x] 写缺失标注、正常坐标转换、非法值、非法配置测试并运行，确认缺少新模块导致失败。
- [x] 实现配置、数据结构及解析器，含清晰中文输入输出说明。
- [x] 运行 `python -m unittest discover -s tests -p test_tiled_input.py -v`，预期相关测试通过。

### Task 2: 切块与可运行准备入口

**Files:** tiling.py, pipeline.py, tests/test_tiled_input.py。
**Interfaces:** TileMeta(image_id,tile_id,origin_xy,original_hw,valid_hw,padding); iter_tiles(image,image_id,config)->Iterator[Tile]; prepare_batches(record,config)->Iterator[TileBatch]。
- [x] 写非整除覆盖、小图边缘复制、真实坐标像素一致性、批大小及原尺度张量测试，运行确认失败。
- [x] 实现切块、RGB 归一化批量输入；CLI `python pipeline.py --inspect --limit 1` 读取 pic1 并报告图片/标签/切块信息，不加载骨干、不建库。
- [x] 运行上述测试及真实数据检查，预期 847×711 原图产生 7×6=42 块，模型输入 (B,3,224,224)。

### Task 3: 算法接口与文档

**Files:** descriptors.py, association.py, memory_bank.py, prediction.py, stitching.py, evaluation.py, contracts.py, README.md, docs/ROADMAP.md, docs/REFACTOR_LOG.md。
**Interfaces:** DescriptorSet(features,xy,image_id,tile_ids,grid_hw,valid); Channel(members); MemoryBank(features,xy); TileScores(meta,scores,valid); Prediction(score_map,valid_map); build_descriptors(features,metadata,config); associate_channels(descriptors,config); build_memory_bank(channels,config); predict_tiles(descriptors,bank,metadata,config); stitch_scores(scores,original_hw); evaluate_predictions(predictions,boxes,threshold)。
- [x] 定义 dataclass 和带完整类型、中文语义说明的函数签名；上述六个算法入口明确抛出 NotImplementedError。
- [x] 文档说明每个模块的责任、入口命令、已实现/待实现边界以及正式实验仍需划分清单。
- [x] 全套 unittest、真实数据遍历、CLI 帮助/检查及模块语法检查通过；不能声称检测效果已验证。
- [x] 最后一次独立代码审阅，修复实质问题并记录证据。

## 实施记录

- 用户明确指定 InReaCh-new 目录，原位实现。当前环境 git 不可用，不建 worktree、不提交；使用本计划与 REFACTOR_LOG 记录实际改动。
- 未指定参考/验证/测试比例，不擅自生成正式划分；pic1 清单仅用于输入检查，不宣称为独立测试集。

任务1、2已验证完成；任务3实现与验证已完成，最终独立审阅结果追加至 REFACTOR_LOG。
