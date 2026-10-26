# 重构过程记录

## 2026-10-20 · 步骤 0：建立独立工作区

**目标**：在不改动旧项目的情况下，为从零重写建立可追溯的起点，并说明输入尺寸如何处理。

**已做**：

- 创建 `InReaCh-new/`、`baseline/` 和 `docs/`。
- 从 `../InReaCh-main/` 复制当时的 7 个 Python 文件、原 `README.md`、`requirements.txt` 和 `conda_inreach.yml` 到 `baseline/`；在 `baseline/SHA256SUMS` 记录校验值。`test_pic1_heatmaps.py` 是之前修改过的测试脚本，所以快照代表“当前旧项目”，不是纯净上游版本。
- 编写本目录的 `README.md`、`AGENTS.md`、`DESIGN.md` 和 `ROADMAP.md`。
- 约定推荐方向：代码允许配置高、宽；同一次实验采用固定尺寸。此处还没有编写或运行新版模型。

**创建时检查到的环境**：Python 3.12.15、PyTorch 2.6.0+cu118、torchvision 0.21.0+cu118、NumPy 1.26.4、SciPy 1.15.2、scikit-learn 1.6.1、OpenCV 4.11.0、FAISS 1.8.0；PyTorch 报告可用 GPU 数量为 8。这里只是环境记录，不表示新版代码已通过 GPU 测试。

**验证**：在 `baseline/` 执行 `sha256sum -c SHA256SUMS`，10 个文件均显示 `OK`、退出码为 0；逐个用 `cmp` 与旧目录对应文件比较，全部一致、退出码为 0。文档只描述已建立的第 0 步，新模型尚无运行测试。

**已知问题**：旧版对正方形特征图有多处假设；此前 pic1 的全量可视化产生了 158 张文件，但用户反馈检测效果很差，文件成功生成不能代表指标合格。后续应先确认数据与标签含义，再比较输入尺寸。

**下一步**：先解释并实现尺寸配置与图像形状检查，给出一张正方形图和一张长方形图的可复现测试。

## 2026-10-21 · 步骤 1 与部分步骤 2：pic1 图像输入

**目标**：参照原代码，先把图像从文件转换成固定尺寸的模型输入；中心裁剪时同步变换 pic1 检测框。

**文件**：新增 `image_processing.py`、`tests/test_image_processing.py`；更新 `README.md`、`docs/ROADMAP.md`、本日志。

**与基线的区别**：保留 `cv2.imread` 的 BGR 输出、`cv2.resize` 的线性插值、中心裁剪和 `FeatureDescriptors.image_net_norm` 的 BGR 通道运算。将原版散落的固定尺寸改为 `ImageTransform(resize_height, resize_width, crop_height, crop_width)`，支持方形和长方形；增加 pic1 的 YOLO 框读取和裁剪坐标变换。未修改旧项目或数据。原版训练图默认直接缩放到 `224×224`，测试图默认先缩放到 `256×256` 再裁剪到 `224×224`，新配置可分别表达这两种行为。

**输入数据与固定参数**：`/home/g2080-2/Project/data/all/images/pic1_*.jpg` 与同名 `labels/*.txt`。真实单图检查使用 `pic1_00006.jpg`：原图 `(847, 711, 3)`、3 个框；检查 `ImageTransform(256, 256, 224, 224)` 和 `ImageTransform(847, 847)`。当前机器的 `inreach` 环境：Python 3.12.15、PyTorch 2.6.0+cu118、torchvision 0.21.0+cu118、NumPy 1.26.4、OpenCV 4.11.0、FAISS 1.8.0；GPU 为 RTX 2080。

**验证命令与实际输出**：

- `PYTHONPATH=. /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v`：6 个测试通过，退出码 0。测试覆盖尺寸顺序、中心裁剪、框裁剪、旧版通道归一化，以及 `.jpg` 文件名但 BMP 内容的图像读取。
- 逐张调用 `list_pic1_images`、`load_bgr_image`、`load_yolo_boxes` 检查完整 pic1 集合：158 张均可读取且形状都是 `(847, 711, 3)`，共解析 300 个框，退出码 0。
- 用模块读取真实 `pic1_00006.jpg` 后：论文配置输出图像 `(224, 224, 3)`、张量 `(1, 3, 224, 224)`、剩余 2 个框；方形配置输出图像 `(847, 847, 3)`、张量 `(1, 3, 847, 847)`、保留 3 个框；两种张量均为有限值。命令退出码 0。
- `/home/g2080-2/miniforge3/envs/inreach/bin/python -m py_compile image_processing.py tests/test_image_processing.py`：退出码 0。

**已知限制**：旧版 BGR 归一化并非标准 RGB ImageNet 预处理，这里仅为行为对照而保留。尚未实现 MVTec 掩码、数据划分、模型特征、通道或检测指标；单图张量成功不代表检测有效。`847×847` 的完整特征匹配量很大，尚未运行模型。

**下一步**：加载与旧版相同的骨干和特征节点，在单张图上核对特征图真实高宽，再决定大尺寸的计算控制方案。

## 2026-10-21 · 图像输入代码增加中文说明

**目标**：让初学者能从函数说明直接看懂名称、参数类型、返回类型和关键形状变换。

**文件**：更新 `image_processing.py`、`tests/test_image_processing.py`、本日志。仅增加中文模块/类/函数说明及关键步骤注释；输入、输出与旧版对照行为未改。

**输入数据与固定参数**：沿用上一节的 pic1 路径与尺寸配置；本次测试使用原有合成图和临时标签样本，未加载模型。

**验证**：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v`：6 个测试通过，退出码 0。用 `ast.parse` 检查两个 Python 文件：均输出 `syntax ok`，退出码 0。

**已知限制与下一步**：本次只改善代码说明，未新增模型或检测指标；继续按上一节计划核对模型特征形状。

## 2026-10-21 · 按论文流程重新编写最小图像输入

**目标**：按用户最新要求，先说明输入输出，再把图片处理简化为可用于不同数据集的论文基线。此前的 `ImageTransform`、pic1 文件筛选和 YOLO 框处理代码已由用户删除，本节取代前两节所述代码的当前状态；历史验证记录保留。

**文件**：重新新增 `image_processing.py`、`tests/test_image_processing.py`；更新 `README.md`、`AGENTS.md`、`docs/DESIGN.md`、`docs/ROADMAP.md` 和本日志。

**与论文和旧代码的关系**：采用论文实验描述的 `256×256` 缩放后中心裁剪到 `224×224`；使用原项目 `mvtec_loader.py` 中的 OpenCV 线性缩放、BGR 读取方式，以及 `FeatureDescriptors.image_net_norm` 的原始归一化数值。旧代码的训练读取函数默认直接缩放到 `224×224`，与论文文字和旧测试读取函数不同；此处统一按论文尺寸流程实现。没有数据集路径、标签或 YOLO 处理代码。

**输入与固定参数**：函数接受任意图像路径或 BGR NumPy 图像。固定缩放边长 256、裁剪边长 224；输出 float32 张量 `(1, 3, 224, 224)`。实际单图检查使用 `/home/g2080-2/Project/data/all/images/pic1_00006.jpg`，只读图像，不读标签。

**验证命令与结果**：先运行新测试时因模块尚不存在报 `ModuleNotFoundError`、退出码 1；实现后运行 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v`，3 个测试通过、退出码 0。真实 pic1 单图检查输出原图 `(847, 711, 3)`、处理后 `(224, 224, 3)`、张量 `(1, 3, 224, 224)`、类型 `torch.float32`、数值均有限，退出码 0。

**已知限制与下一步**：原项目 BGR 归一化不同于标准 RGB ImageNet 预处理，这里为保持代码对照而保留。当前只验证输入，未加载模型，也未评估检测质量。下一步验证原版骨干与特征节点在单图上的输出形状。

## 2026-10-21 · 消除注释中的 Unicode 提示

**目标与原因**：VS Code 将 Python 文档字符串识别为字符串，报告全角右括号 `U+FF09` 可能与 ASCII 右括号混淆。仅关闭普通注释高亮无法消除文档字符串中的提示。

**文件与改动**：新增 `.vscode/settings.json`，只关闭普通注释的 Unicode 高亮；将 `image_processing.py` 和 `tests/test_image_processing.py` 文档字符串中的全角标点与易混淆符号改为 ASCII。保留中文说明及代码字符串的检查，不改图像处理逻辑或输入输出。

**验证**：`/home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v`：3 个测试通过，退出码 0；搜索两个 Python 文件，易混淆符号仅剩普通注释和运行时错误消息中的全角冒号。普通注释由项目设置排除，运行时字符串仍由编辑器检查。

**限制与下一步**：没有直接访问用户的 VS Code 图形界面，黄色提示是否消失需要重新打开文件后确认。模型特征形状验证仍是下一步。

## 2026-10-21 · 单图骨干特征验证

**目标**：把现有 `(1, 3, 224, 224)` 输入交给原版 Wide ResNet-50-2 骨干，确认实际使用的 13 个节点和每层空间尺寸。

**文件**：新增 `feature_extraction.py` 与 `tests/test_feature_extraction.py`；更新 `README.md`、`docs/ROADMAP.md` 和本日志。

**与原版的对应关系**：节点映射逐项采用 `../InReaCh-main/InReaCh.py` 的实际运行配置，而非 `model.py` 底部的五节点示例。使用 torchvision 当前环境的 `Wide_ResNet50_2_Weights.IMAGENET1K_V1` 加载相同的预训练权重，替代原版每次 `force_reload=True` 的 torch.hub 加载；模型设为 `eval()`，推理使用 `torch.inference_mode()`。原版 BGR 输入归一化仍沿用现有模块。本步只返回骨干特征，不生成描述子或预测分数。

**输入与固定参数**：单图检查使用 `/home/g2080-2/Project/data/all/images/pic1_00006.jpg`，原图 `(847, 711, 3)`，模型张量 `(1, 3, 224, 224)`，类型 `torch.float32`。本次在 CPU 上验证；本地已有 `wide_resnet50_2-95faca4d.pth` 权重缓存。代码本身接受任意同规格图像张量，输入不含标签。

**验证命令与结果**：新测试先因 `feature_extraction` 模块不存在报 `ModuleNotFoundError`、退出码 1；实现后运行 `/home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v`，4 个测试通过、退出码 0。真实单图运行退出码 0，`Level_1` 到 `Level_3` 为 `(1, 256, 56, 56)`，`Level_4` 到 `Level_7` 为 `(1, 512, 28, 28)`，`Level_8` 到 `Level_12` 为 `(1, 1024, 14, 14)`，`Level_13` 为 `(1, 2048, 7, 7)`；全部特征值有限。

**已知限制与下一步**：首次在其他环境运行可能需要下载预训练权重；目前只检查特征形状，尚未比较旧版与新版的逐元素特征值或检测指标。下一步先确定参考图与测试图名单，再实现多层特征描述子。

## 2026-10-21 · 说明图像尺寸常量

**目标与文件**：在 `image_processing.py` 中说明 `RESIZE_SIZE` 是缩放后的正方形边长，`CROP_SIZE` 是中心裁剪后的正方形边长，两者单位均为像素；本日志记录变更。

**与基线及输入输出的关系**：仅补充中文注释，仍对任意输入图像执行 `256×256` 缩放和 `224×224` 中心裁剪，模型输入形状不变。

**验证**：用 `ast.parse` 解析 `image_processing.py`，输出 `image_processing.py syntax ok`，退出码 0。

**已知限制与下一步**：未改变处理逻辑；继续按路线图完成数据组织和描述子阶段。

## 2026-10-21 · 将尺寸常量说明改为悬停文档

**目标与原因**：用户需要在其他代码位置悬停 `RESIZE_SIZE`、`CROP_SIZE` 时看到含义；上一节的普通注释不能作为变量文档显示。

**文件与改动**：在 `image_processing.py` 中将两个普通注释改为紧跟赋值的字符串文档，分别说明缩放边长与中心裁剪边长及像素单位；本日志记录修正。数值和图像处理输入输出保持不变。

**验证**：用 `ast.parse` 检查两个常量的值仍为 256、224，且每个赋值后紧跟字符串文档，退出码 0；运行 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v`，4 个测试通过，退出码 0。

**限制与下一步**：本地没有直接检查 VS Code 悬停界面；该格式依据 Pylance 官方的变量文档解析规则。继续按路线图推进数据组织和描述子。

## 2026-10-21 · 将模型输入改为 RGB

**目标**：按用户要求把 OpenCV 读取的 BGR 图像转为 RGB，并使用当前 Wide ResNet-50-2 ImageNet V1 权重对应的 RGB 归一化参数。

**文件**：修改 `image_processing.py`、`tests/test_image_processing.py`、`README.md`、`docs/DESIGN.md`、`docs/ROADMAP.md` 和本日志。

**与论文和原版的区别**：保留论文描述的 `256×256` 缩放及 `224×224` 中心裁剪。原版 `mvtec_loader.py` 使用 `cv2.imread` 后不交换通道，`FeatureDescriptors.image_net_norm` 使用均值 `[0.456, 0.406, 0.485]`。当前版本在读取时用 `cv2.COLOR_BGR2RGB` 交换通道，再使用均值 `[0.485, 0.456, 0.406]`、标准差 `[0.229, 0.224, 0.225]`。论文未说明通道转换步骤；这次变更是用户指定的预处理差异，不能把新特征值直接当作原版复现结果。

**输入与固定参数**：接受任意图像路径，`load_image` 返回 RGB NumPy 图像；缩放边长 256，裁剪边长 224，模型输入为 RGB 顺序的 `(1, 3, 224, 224)` float32 张量。真实单图检查使用 `/home/g2080-2/Project/data/all/images/pic1_00006.jpg`，不读取标签。

**验证命令与结果**：修改实现前先运行 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -p 'test_image_processing.py' -v`，RGB 读取与归一化两项测试按预期失败，退出码 1；实现后同一命令 3 项测试通过，退出码 0。完整测试命令 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v` 的 4 项测试全部通过，退出码 0。真实单图运行得到 `(1, 3, 224, 224)` 输入、13 层有限值特征，首层 `(1, 256, 56, 56)`、末层 `(1, 2048, 7, 7)`，退出码 0。

**已知限制与下一步**：尚未比较 RGB 与原版 BGR 在同一数据划分上的检测指标。后续构建参考集和测试集时，两者必须统一采用 RGB 流程。

## 2026-10-21 · 保存单图缩放效果

**目标与文件**：将指定测试图的缩放图和最终中心裁剪图保存到 `result/`，便于人工查看；新增 `result/pic1_00036_resized_256.png`、`result/pic1_00036_preprocessed_224.png` 和本日志记录，未修改处理代码。

**输入与参数**：读取 `/home/g2080-2/Project/data/all/images/pic1_00036.jpg`，原始尺寸 `(847, 711, 3)`；沿用当前 RGB 读取、OpenCV 线性插值、`256×256` 缩放与 `224×224` 中心裁剪。保存时转回 BGR 供 `cv2.imwrite` 写入 PNG。

**验证与结果**：一次性 Python 命令成功写入两张图片，输出尺寸分别为 `(256, 256, 3)` 和 `(224, 224, 3)`，退出码 0；使用本地图像查看工具检查两张图片均能显示。

**限制与下一步**：此处只检查几何变化和颜色显示，不代表异常检测效果；后续按路线图推进。

## 2026-10-21 · 保存原尺寸灰度图

**目标与文件**：将指定测试图直接转为灰度，不缩放或裁剪；新增 `result/pic1_00036_grayscale_original.png` 和本日志记录，未修改处理代码。

**输入与参数**：输入 `/home/g2080-2/Project/data/all/images/pic1_00036.jpg`，使用 OpenCV 的 `IMREAD_GRAYSCALE` 读取并保存为 PNG。

**验证与结果**：命令退出码 0；读取及保存后的灰度图形状均为 `(847, 711)`，并通过本地图像查看工具确认可显示。

**限制与下一步**：该图只用于查看灰度外观，不代表模型当前改用灰度输入；后续仍按路线图推进。

## 2026-10-21 · 灰度图双边滤波试验

**目标与文件**：以上一步原尺寸灰度图为输入，生成双边滤波预览；新增 `result/pic1_00036_grayscale_bilateral_d9_sc75_ss75.png` 和本日志记录，未修改模型预处理代码。

**输入与参数**：输入 `result/pic1_00036_grayscale_original.png`，调用 OpenCV `bilateralFilter`，参数为 `d=9`、`sigmaColor=75`、`sigmaSpace=75`。

**验证与结果**：命令退出码 0；输入和保存后的输出形状均为 `(847, 711)`；两图平均绝对像素差约为 `9.48`，结果通过本地图像查看工具确认可显示。

**限制与下一步**：此图只用于比较滤波外观，尚未用检测指标评估滤波对模型的影响。

## 2026-10-21 · 灰度图高斯滤波试验

**目标与文件**：对原尺寸、未滤波的灰度图生成高斯滤波预览；新增 `result/pic1_00036_grayscale_gaussian_k5_sigma1p5.png` 和本日志记录，未修改模型预处理代码。

**输入与参数**：输入 `result/pic1_00036_grayscale_original.png`，调用 OpenCV `GaussianBlur`，核大小 `(5, 5)`、`sigmaX=1.5`。

**验证与结果**：命令退出码 0；输入和保存后的输出形状均为 `(847, 711)`；两图平均绝对像素差约为 `6.81`，结果通过本地图像查看工具确认可显示。

**限制与下一步**：此图用于与原始灰度图及双边滤波图比较；尚未用检测指标评估滤波对模型的影响。

## 2026-10-23 · 重叠切块设计草案（待确认，未实现）

**目标**：根据用户选定的方案 A，整理改善约 4×7 像素缺陷漏检的重构设计。

**文件**：新增 `docs/superpowers/specs/2026-10-23-overlapping-tiles-design.md`；更新本日志。现有实现与旧项目未修改。

**输入与固定参数**：用户描述的已对齐图片，残余偏移约 2–4 像素，约 40% 图片有缺陷，YOLO txt 框；选定 224×224 原尺度切块、步长 112。本次没有读取数据集或运行模型。

**与基线区别**：设计提出切块坐标追踪、跨原图支持计数、重叠描述子归属、可选空间门控、原图分数融合和框级评估；除用户已选切块参数外，其余建议待设计确认，不是已完成算法。

**验证命令与实际输出**：用 Python pathlib 读取文档，断言包含 224×224、112、YOLO、image_id、NotImplementedError、待用户确认，输出 `design document verified: 125 lines; required contracts present`，退出码 0。这仅验证文档保存及关键契约存在，不验证算法可行性或性能。`git -C ... status --short` 报 `git: command not found`，未能检查 Git 状态或提交。

**限制与下一步**：待设计审阅及用户确认后制定实施计划，再逐步编写框架。数据清单、标注完整性、类别语义和对齐坐标系需在正式实验前确认。

## 2026-10-23 · pic1 数据复制与重叠切块代码框架

**目标**：按用户批准的设计交付框架及可运行输入；本轮不实现完整检测算法。

**文件**：新增 `config.py`、`contracts.py`、`dataset.py`、`tiling.py`、`pipeline.py`、`descriptors.py`、`association.py`、`memory_bank.py`、`prediction.py`、`stitching.py`、`evaluation.py`、`tests/test_tiled_input.py`、`.gitignore`、实施计划与 `docs/pic1_input_check.json`；更新 README、ROADMAP、AGENTS、设计文档及本日志。

**数据复制**：用户指定从 `/home/g2080-2/Project/data/all` 复制 pic1 到新版 `data/pic1/images` 与 `labels`。复制158张图片、158个txt，共285872254字节，316文件逐项SHA-256验证；清单 `data/pic1/copy_manifest.json` 保存来源、相对路径、大小及摘要。复制前检查已有同名目标，不覆盖内容不同的文件。源数据未修改。

**实际数据**：158张均为847×711，均有框，共300框，类别0/1/2分别163/69/68。用户确认全部框表示缺陷，因此图片级污染率100%，取代40%的估计；不据此断言局部特征污染率，也不能计算正常/异常两类图像 AUROC。正式数据划分尚未指定。

**与基线区别与参数**：新增224×224原尺度切块、步长112、默认batch_size=4；每张42块，原图边缘完整覆盖。RGB归一化复用现有函数，不调用整图缩放裁剪；骨干未修改。本轮未加载骨干做切块特征推理。算法接口均显式 NotImplementedError，不能生成伪分数。

**决策记录**：
- 用户指定新版目录，原位实施；git不可用，未创建worktree或提交，计划与本日志承担进度记录。
- 未指定比例及同工件分组，不自动生成正式参考/验证/测试划分，避免暗中引入泄漏或评估假设。
- 描述子提取保留全部有效观测，新增 `select_reference_descriptors` 仅供建库做位置归属，保证预测仍能重叠融合。
- 全量检查最初在pic1_00152第1行失败；排查发现8个框边界为1.0000005，属于六位小数舍入。加入回归测试先观察失败，再实现1e-6归一化容差，告警裁回边界；更大越界仍报错，原txt未改。

**验证命令与实际结果**：
- 复制后原有 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v`：4项通过。
- 新测试首次运行因 `config` 模块未实现失败；输入实现后7项通过。舍入回归测试先因边界非法报错；修正后全套上述命令12项通过，退出码0。
- `PYTHONDONTWRITEBYTECODE=1 /home/g2080-2/miniforge3/envs/inreach/bin/python pipeline.py --inspect --limit 1`：退出码0，pic1_00006原图847×711、3框、42块、首批(4,3,224,224)。
- 全量遍历图片、解析标签与切块，输出158张、300框、每张42块、8个舍入告警；再次校验316文件SHA-256一致，退出码0，报告保存在 `docs/pic1_input_check.json`。
- 新模块 AST 语法解析通过；检测算法占位入口单独检查明确抛出 NotImplementedError。

**已知限制与下一步**：未实现完整检测，不声称改善小缺陷准确率。下一步逐项实现单图多层描述子、网格到原图坐标和仅建库的位置归属；正式指标前确认原图划分及标注完整性。

**最终审阅**：独立审阅未发现 Critical/Important 问题，复跑8项输入测试通过。两项文字问题（ROADMAP历史状态、描述子错误消息中的职责）已更正，不改变行为。审阅者未独立重跑复制历史、全量哈希及旧测试，相关证据由本轮执行记录提供；完整算法及效果不在框架审阅范围。CLI帮助退出码0，7个算法入口均明确抛出 NotImplementedError。

## 2026-10-23 · 清理可重新生成的文件

按用户要求删除两处 Python 缓存共4个 pyc，以及 result 中4张旧预处理预览图；空目录一并移除。保留源码、测试、数据、编辑器配置及设计记录。历史日志中的预览文件路径仅代表当时产物，现已清理。

验证：逐项检查8个目标文件均已删除；data/pic1 清单中的316个文件重新 SHA-256 校验全部一致。未改变代码行为，未重复运行模型测试。

## 2026-10-23 · 多层描述子及原图坐标映射

**目标**：将已实现的单块13层骨干特征聚合到共同网格，并输出每个描述子的原图位置和补边有效性。

**文件**：实现 `descriptors.py` 的 `build_descriptors`；新增 `tests/test_descriptors.py`；更新 README 和本日志。`select_reference_descriptors` 仍待实现，通道/参考库/预测未修改。

**输入与参数**：输入为 `extract_features` 返回的多层 `(B,C,Hf,Wf)` 和对应 `TileMeta`。使用每层3×3邻域、零补边、双线性插值 `align_corners=False`、每层1024维一维自适应池化、最终1024维一维自适应池化；共同网格采用首层真实 `(Hf,Wf)`，不增加空间降采样。坐标为 `(origin_x+(col+0.5)*tile_size/Wf, origin_y+(row+0.5)*tile_size/Hf)`，表示网格单元中心，不是精确卷积感受野中心。

**与原版区别**：保持原版局部邻域与两次池化思路；聚合在CPU逐块处理以控制GPU峰值。原版可在描述子末尾附加两维块内位置编码，本步未附加，因为切块匹配需使用独立保存的原图坐标。原版 BGR 预处理与当前 RGB 的既有差异仍存在。上述改变可能影响最终距离和检测指标，尚未评估。

**验证**：新测试先因 `NotImplementedError` 失败4项；实现后 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -p test_descriptors.py -v` 4项通过。测试覆盖多层形状、非规则贴边起点、补边有效性、多图片隔离、批次错误和邻域变化。真实单图骨干及完整测试结果见本节后续验证补充。

**限制与下一步**：每批单独输出描述子；跨批按原图汇总及仅建库的位置归属尚未实现。此步只有特征构建，没有异常分数、检测指标或小缺陷改善证据。

**实际验证补充**：真实 `pic1_00006` 首块输入现有预训练骨干，共13层；`build_descriptors` 输出 `grid_hw={0:(56,56)}`、`features=(3136,1024)`、首末原图坐标分别为 `(2,2)` 和 `(222,222)`，3136项均有效且数值有限，退出码0。完整命令 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v`：16项通过，退出码0。真实单块CPU聚合耗时明显，尚未测完整42块吞吐与峰值内存，后续批量实验前应测量。

## 2026-10-23 · AI实施交接

新增 `docs/AI_HANDOFF.md`，基于实际源码中的函数和占位入口核对当前状态，整理已确定框架、后续实现顺序、数据契约、验收及未冻结细节。仅新增文档，未改算法，未重复运行测试。已有16项通过是前一步的实际记录，不作为本轮新运行结果。

## 2026-10-23 · 跨批描述子汇总与建库位置归属

**目标**：将各批同一原图的描述子合为一组，并在建库前处理重叠图块的重复空间证据。

**文件**：`descriptors.py` 新增 `merge_descriptor_sets`、实现 `select_reference_descriptors`；新增 `tests/test_descriptor_assembly.py`；更新 README、AGENTS、ROADMAP、AI_HANDOFF 与本日志。

**输入输出**：输入若干 `DescriptorSet` 批次片段及该原图完整 `TileMeta`。汇总输出保持 features、xy、tile_ids、valid 行一一对应，拒绝同图重复tile_id和无效形状。建库筛选仅保留 valid 且属于当前块的行；对每个原图坐标，在所有覆盖块中比较到各有效块边缘的最小距离，较大者获胜，平局取较小tile_id。`grid_hw`保留原块尺寸，但筛选后为稀疏行集，不能直接reshape成整块热图。

**与原版区别**：原版只按整张图的单一特征网格组织描述子；这里为重叠切块新增原图身份、真实坐标和只在建库时执行的归属筛选。预测阶段使用原始完整观测。

**验证**：首次新测试因 `merge_descriptor_sets` 不存在而报 ImportError；实现后，一个贴边测试的预期错误，误以为空间顺序输出，实际接口按块与行优先顺序输出；修正测试预期后 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -p test_descriptor_assembly.py -v`：5项通过，退出码0。覆盖拆批一致、重复tile_id、错位字段、非规则贴边、平局、补边及元数据缺失。

**限制与下一步**：尚未实现跨原图互近邻关联、通道、正常库或检测结果。下一步核对旧版 `gen_assoc`/`gen_channels` 并以不同image_id建立候选关联。

**完整回归验证补充**：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v`：21项通过，退出码0。本步未运行全部42块的真实骨干提取；先前的真实单块验证不代表本次整图运行结果。

## 2026-10-23 · 临时PPT汇报材料

新增 `docs/TEMP_PPT_新思路汇报.md`，按9页组织研究问题、原尺度重叠切块、原图坐标、建库重复贡献处理、空间邻域关联、完整流程、对照实验及阶段进展。每页包含可用页面文字、讲解补充和配图建议。依据当前代码/交接/日志区分已实现、待实现和待验证；未新增算法或实验结果。

## 2026-10-23 · 指定图片缩放

**目标与文件**：按用户要求，将 `data/pic1/images/pic1_00036.jpg` 缩放至244×244，保存为 `result/pic1_00036.jpg`；仅新增结果图和本日志，未修改输入模块或算法。

**输入与参数**：源文件实际为711×847的BMP数据（虽使用 `.jpg` 扩展名），三处同名源文件SHA-256一致。使用Pillow `Image.Resampling.LANCZOS` 直接缩放到 `(宽,高)=(244,244)`，转换RGB，以JPEG质量95、色度子采样0输出；按指定长宽缩放，未保持原长宽比、未裁剪或补边，也未变换标注。该图只作为单独结果文件，未纳入复现实验预处理或评价。

**验证**：`python3 -c` 打开输出图并完整解码，断言 `format == "JPEG"` 且 `size == (244,244)`，退出码0；实际输出RGB、65929字节。源文件SHA-256仍为 `716928b038fd17b6894cfc2f33e72a2354cbadd66a06d0310bfa5eced6dc1962`。未运行模型测试，因为本次未修改模型代码。

**限制与下一步**：本图的几何变换与当前256×256再224×224中心裁剪流程不同；如用于检测或可视化标注，需要同步处理框坐标。

## 2026-10-23 · 第二张指定图片缩放

**目标与文件**：按用户要求，以与 `pic1_00036.jpg` 相同的方式，将 `data/pic1/images/pic1_00595.jpg` 缩放至244×244，保存为 `result/pic1_00595.jpg`；仅新增结果图和本日志，未修改算法或源图。

**输入与参数**：源文件为711×847的BMP数据（扩展名为 `.jpg`），三处同名源文件SHA-256一致。Pillow `Image.Resampling.LANCZOS` 直接缩放到 `(宽,高)=(244,244)`，转换RGB，以JPEG质量95、色度子采样0输出；不保持长宽比，不裁剪或补边，也未变换标注。

**验证**：`python3 -c` 完整解码输出并断言JPEG格式和244×244尺寸，退出码0；实际输出RGB、66167字节。源文件SHA-256仍为 `5758686cd4a455ca8562d18b06b73ad738da6b62a796d0714af8b4c5437c4d28`。未运行模型测试，本次未修改模型代码。输出图如用于检测或标注展示，仍需同步变换框坐标。
