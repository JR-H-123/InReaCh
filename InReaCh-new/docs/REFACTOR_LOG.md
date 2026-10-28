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

## 2026-10-26 · 跨图关联

**输入输出和目的**：仅建库归属后的各原图描述子 -> 互近邻候选通道；支持按原图计数，补边无效行不参与。原有跨批汇总与归属5项复核通过，不重复实现。

**文件**：config.py、contracts.py、descriptors.py、matching.py、association.py、tests/test_association.py；续作计划为 docs/superpowers/plans/2026-10-26-algorithm-completion.md。

**对照旧版与选择**：读取 InReaCh-main/InReaCh.py 的 gen_assoc/gen_channels 和 utils.measure_distances，确认关联使用欧氏距离。种子使用传入参考清单前assoc_depth张（默认10，实际不超过N-1），只向后续图像关联；同距离按种子顺序和行号确定。跨种子只比较已成立的互近邻，修正旧版非互近邻记录仍以有限距离参与最小值选择的行为。空间门控为两轴差均<=radius_px，空间分桶仅减少候选，最终仍精确门控。距离矩阵分块上限match_chunk_size²，默认256；保留全部描述子维度。DescriptorSet和ChannelMember新增original_hw，空间门控拒绝未知/冲突尺寸。

**实测**：中断后检查发现上次新增计划与测试不在工作区，重新落盘后测试首次因缺original_hw契约失败5项；实现后命令 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v`：26项通过，8.886秒，退出0。覆盖互近邻、同图排除、目标跨种子选择、无候选、无效行、跨块平局和坐标系尺寸冲突。先前21项基线通过发生于中断前。

**限制**：尚无真实小样本完整检测或精度指标。当前待实现正常库、预测、融合与评估。本轮在用户指定目录原位修改，不自动提交；保留原有PPT删除状态。上级AGENTS.md本机不存在，以本目录规则和已批准设计为准。

## 2026-10-26 · 通道筛选、正常库与缓存

**输入输出**：跨原图候选通道 -> 合格通道成员组成的正常库，逐行保存image_id、原图内描述子索引、xy和original_hw。相同来源跨通道去重，来源数据冲突报错。

**文件与参数**：memory_bank.py、contracts.py、config.py、tests/test_memory_bank.py。min_channel_length默认3，筛前和筛后均要求成员数严格大于阈值；max_channel_std默认5。离散度为各成员到特征均值的欧氏距离的总体标准差(ddof=0)，保留距离严格小于倍数阈值者。

**对照旧版与选择**：保留旧版球面距离标准差及严格边界；唯一零离散度例外是全部成员与均值完全相同，此时保留全部。等距但不相同的成员仍按旧规则被排除。这一选择避免完全一致通道被0<0删空，但最终检测影响尚未评估。库与预测统一标注euclidean_l2，旧版FAISS预测返回平方L2的差异将在预测步骤消除。

**缓存**：NPZ纯数组+JSON头，不加载pickle；包含特征、坐标、逐行来源、算法签名和调用者提供的参考图哈希/模型签名。算法签名含切块、描述子、坐标、归属、门控和筛选配置；batch_size和match_chunk_size不改变数学结果，不纳入兼容性限制。后续流程负责生成真实数据/模型哈希。

**验证**：4项新测试先因NotImplementedError失败；实现后完整unittest命令（同上一节，OMP/MKL线程2）30项通过，9.608秒，退出0。手算[0,0,0,10]去除10、严格长度边界、完全相同/等距不同通道、重复图支持、空库拒绝和缓存配置/权重失配均通过。

**限制**：尚未预测或产生真实检测指标；缓存本步使用人工来源签名测试，真实来源由后续入口生成。

## 2026-10-26 · 描述子预测

**输入输出**：未经过建库归属筛选的完整测试描述子、正常库、完整块元数据 -> 每块真实(Hf,Wf)分数及valid。文件：prediction.py、tests/test_prediction.py。

**选择与旧版差异**：预测复用关联的精确欧氏最近邻及相同radius_px门控；不直接使用旧版FAISS平方L2输出，故分数单位改变，旧阈值不可复用。无候选输出NaN/False；不查询无效补边行。检索保持所有有效重叠观测。恢复网格前核对原图身份、尺寸、真实起点、逐行坐标与网格中心、完整行数，拒绝把稀疏建库描述子reshape成热图。

**验证**：4项新测试先因NotImplementedError失败；实现后完整unittest（同前OMP/MKL=2）34项通过，9.245秒，退出0。手算(3,4)到零向量为5、无候选NaN/False、无效输入、稀疏行、错位坐标、原图/配置冲突测试通过。尚未实现融合或运行真实完整推理，不能据此声称检测效果。

## 2026-10-26 · 原图分数融合

**输入输出**：各块真实分数网格、valid和TileMeta -> 原图大小Prediction，保留image_id、valid_map、原始最大image_score和coverage。文件：stitching.py、contracts.py、tests/test_stitching.py。

**明确选择**：分数有效分子和有效支持分别用bilinear/align_corners=False插值到块尺寸；无效NaN先用0替换，避免NaN乘0污染。融合分子为中心窗×插值分子，分母为中心窗×插值支持，支持为0的原图像素输出NaN/False。窗为在像素中心采样的二维三角窗，边缘严格为正。按真实origin回放并裁去右/下补边，不做逐块min-max或额外高斯平滑。此插值/融合与旧版重复像素加高斯平滑不同，不能复用旧阈值。

**验证**：4项新测试先因占位NotImplementedError失败；实现后完整unittest（同前OMP/MKL=2）38项通过，8.973秒，退出0。覆盖常量重叠、末块偏移、长方形原图、网格中心插值的手算数值、无效NaN、补边裁剪、未覆盖像素、混原图和重复图块拒绝。尚无真实准确率结果。

## 2026-10-26 · 框级评估与验证阈值

**输入输出**：原图Prediction、YOLO浮点框、冻结阈值 -> 框召回、未匹配区域/已知标注图、异常面积/全部原图面积、有效覆盖及有条件图像AUROC。文件：evaluation.py、tests/test_evaluation.py。

**明确选择**：分数>=阈值为异常，8连通区域不做最小面积删除，每区域取最高分像素，平局行优先；定位点为像素中心(x+.5,y+.5)，采用半开框边界[x1,x2)×[y1,y2)。用增广路径求最大基数一对一匹配，不能用简单贪心替代。默认拒绝不完全有效覆盖；显式require_full_coverage=False只作探索性统计。缺失标注和未确认完整时的空标注为unknown；只有标注完整且正常/异常两类均存在才算图像AUROC，从不算矩形伪像素AUROC。

**阈值选择**：调用者必须给出候选值和允许的未匹配区域/图预算；在完整验证标注、全覆盖且至少一个缺陷框的条件下最大化框召回，平局选异常面积更小、再选更高阈值。没有满足预算的候选时报错。该目标是本轮实现选择，不声称是用户指定的最终业务目标。

**验证**：5项测试先因NotImplementedError/缺select_threshold失败；实现后完整unittest（同前OMP/MKL=2）43项通过，9.069秒，退出0。覆盖重叠框增广匹配、对角8连通、行优先平局、缺标注与缺覆盖、AUROC前提、显式预算下选择阈值5。

**数据限制**：用户本轮回复正式参考/验证/测试清单、同工件分组与标注完整性约定“无”。后续真实数据只作流程小样本检查；正式准确率对照不执行，不凭现有pic1全异常子集报告图像AUROC。

## 2026-10-26 · 完整流程入口

**输入输出**：显式参考/验证/测试ImageRecord清单、eval骨干与ExperimentConfig -> 正常库、原图NPZ、冻结验证阈值和JSON报告。新增extract_record_descriptors/fit_reference_bank/predict_records/run_experiment、JSON清单解析与--run。建库不读取标签；逐图提取后只保留归属行，预测保留完整重叠观测。

**验证**：5项新增集成测试先因缺接口失败；实现后完整48项unittest通过，9.313秒，退出0。真实读写合成图+小型确定性骨干串联全部数学模块，确认非法参考标签完全不被读取、全原图输出、阈值在验证集冻结、NPZ身份、清单相对路径、内容/工件跨集合泄漏拒绝、特征预算超限拒绝。这是软件集成验证，不能代替真实预训练骨干的检测验证。

**缓存及资源选择**：数据SHA-256、模型前向结构与state_dict共同签名；报告保留验证/测试图像和标签哈希、配置、耗时、进程峰值RSS与CUDA峰值分配。默认累计参考特征预算2048MiB，超限拒绝而不下采样；该值不限制通道/库等额外内存，不是进程硬限。全量158图流式性能仍待完成。既有--inspect继续可用。输出目录必须新建，避免覆盖旧结果。

**硬件实测**：Python3.12.15、torch2.6.0+cu118、torchvision0.21.0+cu118、NumPy1.26.4、OpenCV4.11.0、FAISS1.8.0；GPU NVIDIA GeForce RTX2080，总显存8164474880字节，检查时可用7948140544字节；主机约64GB内存。接下来执行完整真实原图流程检查，结果另记。

## 2026-10-26 · 用户要求暂停：当前验收状态与下一步

**暂停指令**：用户要求以后“完成一部分功能之后停下更新说明文件，再询问是否进行下一步并说明下一步做什么”，随后明确要求现在先停下。本轮停止实现，只记录状态。后续每完成一项可检查功能先更新说明并等待用户确认，不连续推进所有剩余步骤。

**目前进度**：已实现到显式清单的完整流程入口；最近完整测试48项通过（9.313秒）。这包含人工数学样例、小骨干合成多图端到端，以及真实预训练骨干单块形状验证，不代表真实完整检测验收。尚无正式准确率或小缺陷改善结论。

**独立只读审阅**：审阅者读取代码并做小CPU复现，未发现Critical，发现以下3项Important，均尚未修复：
1. fit_reference_bank只拒绝重复ID/路径，未拒绝参考集内部同内容不同ID的图片副本，会把重复证据计为独立原图支持。现有合成集成测试的3张参考图恰好同内容，需要改为不同内容并增加重复内容拒绝测试。
2. run_experiment在严格全覆盖评估/阈值选择后才保存NPZ；合法无候选孔洞会导致抛错，只留正常库，未保存已计算的score_map、valid_map及覆盖报告。应先保存预测/覆盖产物，再拒绝不成立的指标；验证阶段失败也应保留已完成结果。
3. _model_digest的源码/repr/state_dict可能遗漏自定义nn.Module影响前向的普通实例属性（例如非Parameter的scale）。应要求可核验的完整前向配置签名，或明确限制/拒绝不支持的模型缓存复用。
审阅没有重跑全套测试/GPU，没有判断缺失正式划分下的准确率，不将明示的158图流式限制及原有PPT删除当成本轮缺陷。上述正式实验/大规模边界继续保留。

**真实完整原图验证已停止**：启动命令为 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/g2080-2/miniforge3/envs/inreach/bin/python -u scripts/smoke_tiled_pipeline.py --reference pic1_00006 pic1_00010 pic1_00030 pic1_00036 --query pic1_00053 --output result/smoke-20261026 --device cuda`。参考4张、查询1张均拟使用完整原图，每张42块，仅用于流程检查。用户暂停时已运行约11分钟，尚未打印首张完整参考图完成信息，也未形成查询分数或summary。按用户指令向本轮PID21860发送SIGTERM，进程退出143；不是算法测试通过，也不是算法异常退出。新增结果目录可能为空，不覆盖它重跑。

**性能调查（未修复）**：调用现有_pooled_level，随机单层输入、目标56×56、OMP/MKL=2，分别实测(256,56,56)为1.1223秒、(512,28,28)为4.7933秒、(1024,14,14)为15.2935秒、(2048,7,7)为37.0953秒，诊断脚本退出0。计时与真实smoke并行，不能视作独占资源性能基准；已显示深层聚合耗时明显，但尚未完成更细粒度根因确认或数值等价优化。

**建议下一步（等待用户确认）**：先只修复上述3项审阅问题，逐项失败测试→修复→回归，更新说明后停止汇报。性能优化与完整真实小样本重跑作为再下一步单独确认；正式数据对照继续等待清单、工件分组和标注完整性。

## 2026-10-28 · 修复最终审阅的3项问题（本阶段结束）

**授权范围与节奏**：用户在上轮“先修复3项问题，补测试后停止”的提议后回复“继续”。本轮仅执行这3项修复与验证，完成后更新说明并停下，下一阶段等待用户确认。未重跑完整真实原图流程或正式准确率实验。

**问题1：重复参考文件冒充独立支持**
- 输入仍为无标签原图清单，输出为经过文件内容唯一性检查的建库来源。fit_reference_bank在模型签名、特征提取及缓存读取前拒绝相同SHA-256的不同ID/路径文件；validate_splits同样拒绝参考集内部重复内容。
- 合成参考图由相同灰度128改成128/129/130，确保集成测试不依赖重复证据。新增测试涵盖直接建库、已有缓存复用和完整清单检查的拒绝行为。
- 首次回归测试失败：预期ValueError未抛出；实现后6项流程测试通过，0.155秒，退出0。
- 限制：内容检查是文件字节SHA-256，重新编码或其他派生图仍需要真实group_id，不能宣称已自动识别全部近重复图。

**问题2：缺覆盖或后续读图失败时丢失已完成预测**
- predict_records增加可选on_prediction回调；run_experiment在每张原图预测完成时立即保存NPZ与覆盖信息，再进行标注读取、阈值选择和严格评估。
- 报告用临时JSON替换更新，记录running或incomplete状态、failed_stage、错误类型/消息、已完成产物及冻结阈值。评估仍会拒绝缺覆盖并抛出原异常；test_metrics保持null，不将部分覆盖冒充完整指标。
- 3项测试使用真实小模型与算法：验证图网格相位改变造成无候选、测试图无候选、后续损坏图像。首次因报告/NPZ缺失失败；损坏图片的已有异常类型实际为FileNotFoundError，先修正测试后确认真正失败是前一张NPZ缺失。
- 实现后发现测试原先把部分覆盖误判为全无覆盖；其他重叠块仍有有效观测，修正为检查覆盖率在0与1之间、报告与保存有效图一致、有效分数有限且无效分数为NaN。随后9项流程测试通过，0.290秒，退出0。

**问题3：自定义模型非state_dict配置未进入签名**
- 增加显式协议inreach_cache_config()，返回带正整数version的有限JSON对象。自定义模型必须完整声明所有影响forward的非state_dict配置，包括普通属性、子模块/外部设置；未声明、非JSON或含NaN/Inf的配置直接拒绝建库/复用。
- 模型摘要升级为signature_version=2，散列声明配置、模型类型、结构源码/FX code、repr及真实state_dict；正常库provenance同时保存model_config。默认Wide ResNet-50-2 V1工厂提供架构、权重名与有序13节点配置。
- 新测试首次失败3项：普通scale从1变2仍读取缓存、未声明模型获准、含NaN配置获准。修复后测试确认前向确实从1变2，并同时拒绝缓存复用和使用旧库预测。默认真实骨干测试另验证真实权重变更也会改变签名。
- 兼容性：旧模型摘要缓存需重建，不静默回退。完整声明是自定义模型提供者的接口责任；框架验证JSON/版本并散列声明，不自动证明任意Python模型的声明已经穷尽全部前向依赖。

**本轮改动文件**：pipeline.py、feature_extraction.py、tests/test_pipeline_algorithms.py、tests/test_feature_extraction.py；同步README、AGENTS、AI_HANDOFF、ROADMAP、续作计划与本日志。未修改旧版、源数据或旧结果，保留工作区既有PPT删除。

**完整验证**：
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v`
实际55项通过，12.495秒，退出0；完整输出位于/tmp/inreach-20261028-final-suite.log。git diff --check退出0。此次未再次派生审阅者；依据上轮独立审阅逐项修复并以新增失败测试→通过及全套回归验证。

**当前状态与下一步**：3项审阅问题的本轮修复完成。描述子聚合性能仍慢，真实完整原图验证未完成，正式清单/工件分组/标注完整性仍缺失。建议下一阶段先只定位深层聚合耗时、验证等价优化的可行性并记录数值误差/资源，再停下汇报；真实全图重跑在之后单独确认。本轮到此停止，等待用户决定是否进行性能调查。

## 2026-10-28 · 热力图、原图叠加与定位结果导出

**本阶段授权**：用户确认“先补齐热力图、原图叠加图和定位点导出，完成测试后停下汇报；真实完整原图验证另行确认”。本轮只完成结果导出功能，不进行真实全图模型或正式准确率实验。

**输入输出与目的**：输入原图RGB(H,W,3)、Prediction原图分数/valid/身份、阈值和固定显示范围；输出同原图尺寸的heatmap.png、overlay.png、阈值已确定时的localization.png，以及detections.json。原始NPZ继续保存，不原地修改输入或分数。

**文件与规则**：
- 新增localization.py，extract_regions统一8连通、score>=threshold、峰值平局行优先、像素中心(x+.5,y+.5)、半开xyxy框和区域面积；evaluation.py改为复用它，定位口径不另起一套。
- 新增visualization.py。TURBO显示颜色，alpha默认0.5；无效区域固定RGB(128,128,128)，不当正常低分显示。红框表示阈值区域外接框，黄十字表示峰值；它们不是回归的真实缺陷框，原始距离分数不是概率。
- JSON保存原图身份、尺寸、覆盖率、原始峰值分数、阈值/比较规则、区域面积/外接框/峰值、显示范围和文件路径。未选阈值时regions=null、localization_status=threshold_not_selected，不伪装成未检测到缺陷。全无效或部分覆盖明确记录状态。
- pipeline.py完整运行接入导出。显示范围可用清单visualization.score_range指定；默认取[0,验证集有效分数最大值]，没有正有效最大值则取[0,1]；验证/测试共用，测试分数不参与确定色标。颜色裁剪只影响PNG，不影响NPZ、定位或指标。
- 验证预测全部完成后，先导出显示图，再读标签/选择阈值；阈值成功后补定位。测试图逐图保存全部导出；缺覆盖时继续保留PNG/JSON/NPZ并拒绝完整指标。验证阶段较早的后续图读入失败时，已完成NPZ仍保存，因共享色标尚未确定可不生成PNG。

**失败→通过证据**：
1. 定位测试首次ModuleNotFoundError；实现后3项新测试+原5项评估测试全部通过，0.023秒。覆盖对角8连通、平局、半开框/面积、阈值边界和无效高分排除。
2. 显示测试首次缺visualization模块失败；实现后显示6项+定位3项+评估5项，共14项通过，0.152秒。覆盖RGB写读、固定色标、常量/全无效、原始值不变、显示上限10但峰值100仍按原值定位、阈值缺失状态和坏输入。
3. 流程导出回归首次5项报KeyError/缺visualization_range参数；接入后完整套件66项通过，12.199秒，退出0。覆盖同一次运行的共用验证色标、显式范围/alpha、定位文件与冻结阈值一致及验证/测试缺覆盖产物保留。

**完整命令**：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v`，输出/tmp/inreach-result-export-suite.log。

**离线CLI与人工查看**：用明确写有SYNTHETIC DEMO的160×240合成原图/分数测试，输入保存于result/export-input-20261028。执行：
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/g2080-2/miniforge3/envs/inreach/bin/python visualization.py --image result/export-input-20261028/synthetic_demo.png --prediction result/export-input-20261028/prediction.npz --output result/export-preview-20261028 --threshold 4 --score-max 10 --alpha 0.5`
退出0，4个输出文件，覆盖率0.9166666666666666，两个峰值分别(60.5,50.5)/(170.5,110.5)，原始分数9/7，均为285像素区域。查看overlay和localization PNG，确认尺寸、RGB颜色、灰色无效边和原图坐标标注。此图完全是导出测试，不是真实模型检测或准确率结果。

**限制与后续**：真实完整原图模型运行、性能定位/等价优化和正式准确率对照均未完成。计划见docs/superpowers/plans/2026-10-28-result-export.md。独立只读审阅结果后续补记；本阶段完成后按约定停止，等待用户确认下一阶段。

**最终独立只读审阅补充**：按requesting-code-review/executing-plans流程派生一次独立审阅，审阅者未修改文件、提交或运行真实模型；独立运行定位、显示、评估及流程相关28项测试，全部通过0.721秒，并复现测试后续图像失败时前一图NPZ/三张PNG/JSON保留。未发现Critical/Important，标准pipeline/CLI可接受。

**Minor（延期）**：直接调用export_prediction，对同目录同stem先设threshold、再刷新为None时，JSON正确标记未就绪，但旧_localization.png仍留在目录。当前pipeline只从无阈值补到有阈值，CLI强制新目录，不触发此状态回退。本轮记录并在README提示直接API调用者换目录/前缀；未增加删除文件行为，延期处理该边界。

**审阅边界与结论限制**：未独立判断真实模型准确率、正式划分、真实图耗时或GPU/全量性能，这些继续保持未验证；旧版、此前算法和用户已有PPT删除不属于本阶段改动；未模拟磁盘损坏/进程强杀时多文件写入一致性，因此仅确认正常异常路径的逐图保留与报告更新，不承诺PNG/JSON/NPZ整组事务性写入。

**阶段完成并停下**：热力图/叠加图/定位PNG与JSON已实现并接入，66项完整测试通过，离线合成导出和独立审阅通过。下一步可在用户确认后，用少量明确指定的真实完整原图做探索性试跑，记录真实耗时、覆盖率与导出；若聚合仍过慢则再单独确认性能定位/等价优化。正式准确率实验继续等待可靠划分与标注约定。本轮没有启动这些后续任务。

## 2026-10-28 · 少量真实完整原图运行（后按用户选择暂停）

用户授权检查真实完整原图的实际耗时、有效覆盖率与导出。本轮复用既有算法，不修改切块、特征维度或聚合数学操作。运行脚本/日志/报告在result/real-run-20261028；运行前该目录新建，不覆盖旧结果。每张原图独立提取并跨批合并，不把重叠块当独立图像。

**资源与前测**：本机实际可见8张RTX2080，约8GB/卡，48逻辑CPU、约53GB主机可用内存；临时盘约794GB可用。单个真实pic1_00006首块，CPU线程8，骨干0.879秒、现有描述子聚合68.70921193299秒，输出3136×1024且有限。另用随机(2048,7,7)深层特征、CPU线程1测得23.017866416019388秒；该项仅比较运行线程设置，不改变算法，也不是全图耗时。

**明确清单与配置**：4张参考pic1_00006/00010/00030/00036；1张独立展示校准pic1_00053；1张查询pic1_00066。它们是本轮按排序指定的探索性角色，不是已确认正式划分，真实工件分组/标注完整性仍未知。尺寸保持原图，每张预计42块；tile224、stride112、batch2、13层、1024维、assoc_depth1、min_channel_length3（严格>3）、max_channel_std5、radius8，seed0。

**运行方式**：6个独立原图工作进程各用一张GPU、CPU线程8，并行执行同一prepare_batches/extract_features/build_descriptors/merge_descriptor_sets。中间完整描述子保存到/tmp/inreach-real-run-20261028，附源图/模型/算法签名，主进程核对后调用现有归属、关联、筛选、predict_tiles、stitch_scores和export_prediction。仅为运行调度和计时，没有替换聚合算法。每批日志保留块ID和骨干/聚合进度。完整命令：
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/g2080-2/miniforge3/envs/inreach/bin/python -u result/real-run-20261028/runner.py > result/real-run-20261028/run.log 2>&1`

**展示规则预先固定**：只用独立校准图有效分数的99.5%分位作探索性定位阈值，显示范围为该校准图[0,有效max]并供查询沿用；标签不参与建库或此阈值。此阈值不是已验证缺陷判界，不计算或声称正式准确率。若有无效像素，继续导出并如实报告覆盖，不把它记正常。

当前运行尚未完成，实际完整原图耗时、覆盖及导出检查结果将在本节后追加；不能用上述单块计时当最终结果。

**真实长跑暂停补充**：用户选择“暂停当前长跑，先验证数值等价的聚合优化”。向本轮父进程34471发送SIGTERM，由父进程停止6个工作进程；父进程退出143，确认7个PID均已结束。summary.json记录interrupted、signal15、elapsed_wall=1014.5916058080038秒，predictions为空；尚未完成整图、建库、覆盖率或真实导出。逐批日志均停在首批2块聚合；不把本次启动当作完整通过。

**新的授权范围**：当前只做聚合性能定位与数值等价验证，完成测试/说明后停下，不自动重启真实整图长跑。原实现已快照到result/aggregation-profile-20261028/baseline_descriptors.py。

**受控分段测量**：停止其他本轮任务后，随机(2048,7,7)输入、目标56×56、CPU线程2。unfold=0.027254491秒，interpolate=16.609771525秒，reshape=0.002816345秒；旧非连续池化=4.186329690秒，连续复制=3.084824629秒、连续池化=1.148956764秒。连续副本输出逐位相同，最大误差0，但复制+池化几乎抵消收益，因此不采用仅.contiguous的改法。主要问题是把18432个邻域通道先插值到56×56的临时张量。测量保存layout_profile.json。

**待验证方案**：3×3邻域不变，将每位置的线性自适应通道均值池化提前到原层网格，再对1024维通道做原来的bilinear/align_corners=False插值到原公共网格。线性算子在实数运算下可交换；浮点运算不承诺逐位相同，提前固定比较容差atol=1e-6、rtol=1e-5。输出网格/维度/原图坐标/valid必须保持一致。方案尚待真实特征与回归验证。

## 2026-10-28 · 聚合等价优化验证完成（阶段停止）

**本阶段输入/输出与目标**：在用户要求暂停真实长跑后，对同一局部特征输入比较原始源码快照与优化聚合，输出数值误差、原图元数据一致性及受控耗时。未恢复完整原图运行，未生成真实预测覆盖率或检测导出。

**实现**：descriptors.py只改_pooled_level的线性操作顺序：原层unfold3×3 -> 原层各位置通道adaptive_avg_pool1d到1024 -> 原bilinear/align_corners=False到同一公共网格。最终层间1024池化、坐标公式、行序、grid_hw、tile_id与valid不变。设X为位置×邻域分量，S为空间插值，P为通道均值池化，原式(SX)P与新式S(XP)由矩阵乘法结合律在实数下相等；这一理由依赖二者作用于不同轴，不是任意线性步骤都能重排。浮点算术不承诺逐位相同。

**缓存**：config.algorithm_signature的descriptor升级为patch3-pool1024-before-resize-pool1024-no-position-v2；正常库和描述子缓存的完整签名比较会拒绝旧版。后续重跑使用新目录/新缓存，不混入本次已中断的旧实现产物。

**测试与性能检查**：
- 新增tests/test_descriptor_equivalence.py，独立保留原公式作参考，覆盖多层不同网格、非连续输入、贴边起点、补边valid。修改前参考测试通过，确认原公式正确；性能脚本先失败才改生产实现，避免给通用unittest加入机器耗时门槛。
- 新增scripts/check_aggregation_equivalence.py，读取指定原源码快照，固定atol=1e-6、rtol=1e-5，要求相对快照至少2倍加速；把原/新源码SHA-256、线程、输入及结果存JSON。
- 修改前深层基准（CPU2线程）原快照20.247029961秒、当时当前实现16.712977673秒，同源码的单次顺序/环境波动造成1.211倍，未达到2倍，退出1。输出before.json，数值逐位相同。
- 修改后同类深层输入(2048,7,7)、目标56×56、CPU2线程：原20.058759480秒、新0.668209761秒，本次单次测量30.0187倍；最大绝对误差2.384185791e-7，平均1.433875862e-8，容差通过、非逐位相同，退出0，after.json。
- 真实pic1_00006首块、原预训练骨干13层、CPU8线程：对同一次骨干输出分别运行原/新聚合，原79.401880687秒、新10.815612801秒，本次单块7.3414倍；(3136,1024)输出，最大绝对误差8.940696716e-8、平均1.723650112e-9，容差通过，非逐位相同。image_id、original_hw、xy、tile_ids、grid_hw、valid逐项完全一致，退出0，real_tile.json。不是完整原图检测通过。
- 同一次真实单块脚本的随机深层CPU8线程复测17.184250200秒 -> 0.508861600秒、33.7700倍；与CPU2的结果分开记录，不混成总体倍率。
- 完整回归命令 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/g2080-2/miniforge3/envs/inreach/bin/python -m unittest discover -s tests -v`：67项通过，12.756秒，退出0，输出/tmp/inreach-aggregation-equivalence-suite.log。

**性能检查命令**：
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/g2080-2/miniforge3/envs/inreach/bin/python -u scripts/check_aggregation_equivalence.py --baseline result/aggregation-profile-20261028/baseline_descriptors.py --output result/aggregation-profile-20261028/after.json --threads 2`
真实单块使用同脚本追加--real-tile、--threads 8并将输出设为real_tile.json。原快照、分段测量与before/after/real_tile的JSON及日志均在result/aggregation-profile-20261028。

**内存解释**：以深层float32插值结果单个张量计算，原18432×56×56为220.5MiB，新1024×56×56为12.25MiB；这是该临时张量的尺寸，不是进程总RSS实测下降18倍。最终56×56空间网格与1024维均未减少。

**独立只读审阅**：审阅确认不同轴线性算子重排、unfold分量/行序、padding、矩形及上下采样推导正确，签名拒绝路径真实存在；未发现Critical/Important。审阅时为避免干扰计时未自行跑测试，真实13层最终通过由本轮执行结果提供。审阅未判断整图性能、覆盖率、准确率或近平局匹配一致性。

**Minor延期（3项）**：普通小回归尚缺通道压缩不整除及空间下采样专门用例（真实骨干补充了压缩验证）；基准未预热/重复/交替顺序，以上只称本次单次测量，不能当稳定或整图倍率；现有缓存失配测试未专门替换旧descriptor字段，旧聚合签名拒绝由静态完整字典比较确认，不称专项测试通过。

**结论边界与下一步**：在已测随机深层与真实单块上达到固定容差并明显加速，可保留这次实现优化；不承诺所有最近邻/通道筛选的近平局离散决策逐项相同，也尚未验证完整原图性能和覆盖率。当前阶段结束，等待用户确认用新实现、新目录重新运行少量真实完整原图。正式划分、工件组与标注完整性仍未确认，不报告正式准确率。
