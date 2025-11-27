# encoding: utf-8

import numpy as np

npz_2d_path = r"./demo/output/kunkun/input_2D/keypoints.npz"

npz_3d_path = r'demo/output/C-TJ1-try/pose_3d.npz'

# data = np.load(npz_2d_path)
data = np.load(npz_3d_path)
print(f"列出所有数组名：{data.files}\n")

# keypoints = data['reconstruction']
keypoints = data['poses_3d']

print(f"特征点形状：{keypoints.shape}\n")

# print(f"数据样例：{keypoints[0, 0, :]}\n")
print(f"数据样例：{keypoints}\n")



