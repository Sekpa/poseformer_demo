# This code is used to convert the npz data to c3d format.

'''
C3D（Coordinate 3D）是一种紧凑的二进制文件格式，专为运动捕捉、生物力学和动画领域设计，用来把一次试验的所有三维坐标数据、力台信号、采样参数、标记名称等全部打包进一个文件，避免数据与描述信息分离 
'''

# --------------------- #

print("\n# ------ Converting npz to c3d format...  ------ #")

import numpy as np
import ezc3d

# TODO: data preparation

print("\n#1|Loading npz data...",end='|')

video_name = 'C-TJ1-try'

npz_path = 'demo/output/' + video_name + '/pose_3d.npz'

points = np.load(npz_path)['poses_3d'] # (frames=, marker=17, Coordinates = 3)

## 格式？

args = {
    'frames' : points.shape[0],
    'marker' : points.shape[1],
    'Coordinates' : points.shape[2]
}

print("Averaging...",end='|')

pass

print("Done!")

# TODO: save to c3d format

print("\n#2|Converting to c3d format...",end='|')

c3d = ezc3d.c3d()

print("Setting parameters...",end='|')
# 点采样率，标记点设定名称，数据单位，数据格式（坐标，标记点，帧数）
c3d['parameters']['POINT']['RATE']['value'] = [args['frames']] 
c3d['parameters']['POINT']['LABELS']['value'] = [f'M{i:02d}' for i in range(args['marker'])] 
c3d['parameters']['POINT']['UNITS']['value'] = ['m']  
c3d['data']['points'] = points.transpose(2, 1, 0)  

output_path = 'demo/output/' + video_name + '/pose_3d.c3d'
c3d.write(output_path)

print("Done!\n")
