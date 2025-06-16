import numpy as np
import os

# 文件夹路径
clip_feature_folder = "your_path"  # CLIP特征的文件夹
output_feature_folder = "your_path"  # 输出调整尺寸后的 CLIP 特征的文件夹

# 初始化降维和升维矩阵
def initialize_linear_matrices(input_dim, hidden_dim, output_dim):
    linear_down = np.random.randn(input_dim, hidden_dim) * 0.01  # 1024 → 512
    linear_up = np.random.randn(hidden_dim, output_dim) * 0.01  # 512 → 2048
    return linear_down, linear_up

# 处理 CLIP 特征
def transform_clip_feature(clip_feature, linear_down, linear_up):
    # 1. 线性降维 (1, 1024) → (1, 512)
    clip_feature_reduced = np.dot(clip_feature, linear_down)

    # 2. 复制 245 次，使其变为 (245, 512)
    clip_feature_repeated = np.repeat(clip_feature_reduced, 245, axis=0)

    # 3. 线性升维 (245, 512) → (245, 2048)
    clip_feature_expanded = np.dot(clip_feature_repeated, linear_up)

    return clip_feature_expanded

# 初始化降维和升维矩阵 (1024 → 512 → 2048)
linear_down, linear_up = initialize_linear_matrices(1024, 512, 2048)

# 遍历 CLIP 特征文件夹，并调整尺寸
for clip_file in os.listdir(clip_feature_folder):
    if clip_file.endswith('_fused.npy'):
        file_name = clip_file.replace('_fused.npy', '')

        # 读取 CLIP 特征
        clip_feature_path = os.path.join(clip_feature_folder, clip_file)
        clip_feature = np.load(clip_feature_path)  # (1, 1024)

        assert clip_feature.shape == (1, 1024), f"Unexpected CLIP feature shape: {clip_feature.shape}"

        # 调整 CLIP 特征尺寸
        transformed_feature = transform_clip_feature(clip_feature, linear_down, linear_up)  # (245, 2048)

        # 保存为 .tif.npz 格式
        output_path = os.path.join(output_feature_folder, file_name + '.tif.npz')
        np.savez_compressed(output_path, feat=transformed_feature)

print("CLIP 特征尺寸调整完成")
