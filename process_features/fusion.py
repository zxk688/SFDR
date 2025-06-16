import numpy as np
import os

# 文件夹路径
clip_feature_folder = "your_path"  # CLIP特征的文件夹
grid_feature_folder = "your_path"  # 网格特征（.npz文件）的文件夹
output_feature_folder = "your_path"  # 输出融合特征的文件夹


# 初始化线性降维和升维矩阵
def initialize_linear_matrices(input_dim, output_dim):
    # 使用较小的随机数初始化
    linear_down = np.random.randn(input_dim, output_dim) * 0.01  # 降维矩阵
    linear_up = np.random.randn(output_dim, input_dim) * 0.01    # 升维矩阵
    return linear_down, linear_up

# 特征融合操作（加权相加）
def fuse_features(grid_feature, clip_feature, linear_down, linear_up, fusion_method="weighted_sum"):
    # 线性降维
    grid_feature_reduced = np.dot(grid_feature, linear_down)  # 将 245x2048 的网格特征降维到 245x1024

    # 重复 CLIP 特征，使其尺寸匹配
    clip_feature_repeated = np.repeat(clip_feature, grid_feature_reduced.shape[0], axis=0)  # 重复 clip_feature 245 次

    # 加权相加
    if fusion_method == "weighted_sum":
        alpha = 0.5  # 加权系数
        fused_features = alpha * grid_feature_reduced + (1 - alpha) * clip_feature_repeated  # 加权相加

    # 线性升维
    fused_features_upscaled = np.dot(fused_features, linear_up)  # 将 245x1024 的特征升维到 245x2048
    return fused_features_upscaled

# 初始化降维和升维矩阵
linear_down, linear_up = initialize_linear_matrices(2048, 1024)

# 遍历网格特征文件夹，并根据对应的 CLIP 特征进行融合
for grid_file in os.listdir(grid_feature_folder):
    if grid_file.endswith('.tif.npz'):
        # 获取对应的文件名，不带扩展名
        file_name = os.path.splitext(grid_file)[0].replace('.tif', '')

        # 读取 CLIP 对齐特征
        clip_feature_path = os.path.join(clip_feature_folder, file_name + '_fused.npy')
        if not os.path.exists(clip_feature_path):
            print(f"CLIP feature file {file_name}_fused.npy not found. Skipping.")
            continue
        clip_feature = np.load(clip_feature_path)

        # 读取网格特征
        grid_feature_path = os.path.join(grid_feature_folder, grid_file)
        with np.load(grid_feature_path) as data:
            grid_feature = data['feat']  # 从 'feat.npy' 加载网格特征

        # 确保特征维度匹配（grid_feature 为 245x2048，clip_feature 为 1x1024）
        assert grid_feature.shape == (245, 2048), f"Unexpected grid feature shape: {grid_feature.shape}"
        assert clip_feature.shape == (1, 1024), f"Unexpected CLIP feature shape: {clip_feature.shape}"

        # 融合特征
        fused_features = fuse_features(grid_feature, clip_feature, linear_down, linear_up, fusion_method="weighted_sum")

        # 保存融合后的特征到与原来格式一致的 .npz 文件中的 feat.npy
        output_path = os.path.join(output_feature_folder, grid_file)  # 使用原始 .tif.npz 文件名
        np.savez_compressed(output_path, feat=fused_features)

print("特征融合已完成并保存！")