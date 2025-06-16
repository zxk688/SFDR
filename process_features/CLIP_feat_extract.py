import torch
import clip
from PIL import Image
import numpy as np
import os
import re

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 图像和文本文件路径
image_folder = "your_path"  # 图像文件夹路径
text_file = "your_path"  # 文本描述的txt文件路径
output_feature_folder = "your_path"  # 新特征文件的输出路径

# 提取 CLIP 图像特征
def extract_clip_image_feature(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.cpu().numpy()

# 提取 CLIP 文本特征
def extract_clip_text_features(text_list):
    text_tokens = clip.tokenize(text_list).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    return text_features.cpu().numpy()

# 读取文本文件并按每5行文本分组
def load_text_descriptions(text_file):
    with open(text_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    # 每5行作为一组文本描述
    return [lines[i:i+5] for i in range(0, len(lines), 5)]

def embed_features(clip_image_feature, clip_text_features, embed_method="concat"):
    # 计算文本特征的均值并扩展维度，使其与图像特征匹配
    text_features_mean = clip_text_features.mean(axis=0, keepdims=True)  # 扩展到 2D

    if embed_method == "concat":
        # 确保图像特征和文本特征在同一维度上进行拼接
        embedded_features = np.concatenate((clip_image_feature, text_features_mean), axis=1)
    elif embed_method == "sum":
        # 如果是加和操作，需要保持两个特征的形状一致
        embedded_features = clip_image_feature + text_features_mean
    else:
        raise ValueError("Unknown method")

    return embedded_features

# 保存新特征文件
def save_fused_features(fused_features, output_path):
    np.save(output_path, fused_features)

# 读取文本描述
text_descriptions = load_text_descriptions(text_file)

# 提取文件名前的字母部分和数字部分
def extract_key(filename):
    match = re.match(r"([a-zA-Z_]+)(\d+)", filename)
    if match:
        return match.group(1), int(match.group(2))
    return filename, 0  # 如果无法匹配，返回整个文件名作为关键字

# 获取图像文件列表，并按文件名前的字母和数字部分排序
image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.tif'))],
                         key=lambda x: extract_key(os.path.splitext(x)[0]))

# 确保图像和文本的数量匹配
assert len(image_filenames) == len(text_descriptions), "图像文件数与文本描述数不匹配！"

# 遍历图像文件夹，提取图像和文本特征，进行融合并保存
for idx, image_file in enumerate(image_filenames):
    image_path = os.path.join(image_folder, image_file)
    output_path = os.path.join(output_feature_folder, image_file.replace('.tif', '_fused.npy').replace('.jpg', '_fused.npy'))

    # 提取 CLIP 图像特征
    clip_image_feature = extract_clip_image_feature(image_path)

    # 提取与该图像对应的5条文本描述的特征
    clip_text_features = extract_clip_text_features(text_descriptions[idx])

    # 特征嵌入
    embedding_features = embed_features(clip_image_feature, clip_text_features, embed_method="concat")

    # 保存融合后的特征
    save_fused_features(embedding_features, output_path)

print("CLIP特征完成")