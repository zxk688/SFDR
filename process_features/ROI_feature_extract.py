import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F

# 加载预训练模型
model = models.resnet50(pretrained=True)
model.eval()

# 自定义特征提取器类
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = torch.nn.Sequential(*list(model.children())[:-2])
        self.fc = torch.nn.Linear(2048, 1024)  # 修改通道数

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # 池化操作
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)  # 归一化到 [0, 1] 范围
        return x

# 初始化特征提取器
feature_extractor = FeatureExtractor(model)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((500, 500)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 滑动窗口参数
window_size = 56  # 窗口大小为56x56
step_size = 28  # 每次滑动28个像素

# 提取滑动窗口中的ROI特征
def extract_roi_features(image_tensor, window_size, step_size):
    _, _, H, W = image_tensor.size()
    roi_features = []

    for y in range(0, H - window_size + 1, step_size):
        for x in range(0, W - window_size + 1, step_size):
            window = image_tensor[:, :, y:y+window_size, x:x+window_size]
            with torch.no_grad():
                features = feature_extractor(window)
            roi_features.append(features.view(-1))  # 展平特征

            if len(roi_features) >= 50:
                return torch.stack(roi_features[:50])  # 确保只取50个特征

    # 如果提取的特征不足50个，进行填充
    while len(roi_features) < 50:
        roi_features.append(torch.zeros_like(roi_features[0]))

    return torch.stack(roi_features[:50])

# 处理文件夹中的所有图像
input_dir = 'your_path'  # 替换为你的输入图像文件夹路径
output_dir = 'your_path'  # 替换为你的输出文件夹路径
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(input_dir):
    if img_name.endswith('.tif'):
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # 增加批次维度

        # 提取并保存特征
        roi_features = extract_roi_features(img_tensor, window_size, step_size)

        # 将特征转换为NumPy数组
        roi_features_np = roi_features.cpu().detach().numpy()

        # 指定保存路径
        file_path_pooled = os.path.join(output_dir, img_name + '.npy')

        # 保存池化后的ROI特征到.npy文件
        np.save(file_path_pooled, roi_features_np)

