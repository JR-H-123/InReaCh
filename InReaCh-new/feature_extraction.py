"""提取 InReaCh 原版选定节点的图像特征."""

import torch
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor


# 节点名称与原版 InReaCh.py 的实际运行配置一致.
RETURN_NODES = {
    
    "layer1.0.relu_2": "Level_1",
    "layer1.1.relu_2": "Level_2",
    "layer1.2.relu_2": "Level_3",
    "layer2.0.relu_2": "Level_4",
    "layer2.1.relu_2": "Level_5",
    "layer2.2.relu_2": "Level_6",
    "layer2.3.relu_2": "Level_7",
    "layer3.1.relu_2": "Level_8",
    "layer3.2.relu_2": "Level_9",
    "layer3.3.relu_2": "Level_10",
    "layer3.4.relu_2": "Level_11",
    "layer3.5.relu_2": "Level_12",
    "layer4.0.relu_2": "Level_13",
}


def load_feature_model(device: torch.device) -> torch.nn.Module:
    """用途: 加载原版预训练骨干, 并保留 13 个指定特征节点.

    参数: device (运行设备, torch.device).
    返回: 处于推理模式的特征模型 (torch.nn.Module).
    """
    backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    model = create_feature_extractor(backbone, return_nodes=RETURN_NODES)
    return model.to(device).eval()


def extract_features(
    model: torch.nn.Module, model_input: torch.Tensor
) -> dict[str, torch.Tensor]:
    """用途: 对模型输入执行前向推理, 返回原版节点的特征图.

    参数: model (特征模型, torch.nn.Module).
    参数: model_input (输入图像张量, torch.Tensor), 形状为 (N, 3, H, W).
    返回: 节点名称到特征图的映射 (dict[str, torch.Tensor]), 每项为 (N, C, Hf, Wf).
    """
    # 输入与模型放在同一设备, 推理时不创建梯度图.
    with torch.inference_mode():
        return model(model_input.to(next(model.parameters()).device))
