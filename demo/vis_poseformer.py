import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os 
import numpy as np
import torch
import glob
from tqdm import tqdm
import copy
from IPython import embed

sys.path.append(os.getcwd())
from model.poseformer import Model_poseformer
from common.camera import *

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# TODO: 绘制 2D 图像
def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]] # 关节链接

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3
    # img[:, :, :] = 255

    # 建立连线，并绘制 img
    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)

        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img

# TODO: 在 gs 画布上绘制 3D 图像
def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('equal') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


# TODO: 
def get_pose2D(cap, output_dir):
    # 1|视频长宽参数
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 2|TODO: 得到特征点；
    print('\n#get_pose2D|1-Generating',end="")
    keypoints, scores = hrnet_pose(cap, det_dim=416, num_peroson=1, gen_output=True)
        # @ 核心模型 -> tuple[NDArray, NDArray]
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
        # 结合人体模型 -> tuple[NDArray[floating[_32Bit]], NDArray[floating[_32Bit]], list]
    # re_kpts = revise_kpts(keypoints, scores, valid_frames) 
        # 不明意义
    print('|2-Done!',end="\n")
    
    # 3|生成 npz 文件 并返回 Array 数据；
    output_npz = output_dir + 'pose_2d.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)
    
    return keypoints

# TODO: 
def img2video(cap, output_dir):
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size) 

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()

# TODO: 
def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)

# TODO: 2D -> 3D lifting
def get_pose3D(cap, keypoints, output_dir):
    # 1|模型载入
    print('\n#get_pose3D|1-Loading model',end="")
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 3, 512, 1024, 9
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/poseformer_9'
    args.n_joints, args.out_joints = 17, 17

    ## Reload 载入模型
    model = Model_poseformer(args).cuda()

    model_dict = model.state_dict() # 加载模型参数
    # Put the pretrained model of MHFormer in 'checkpoint/pretrained/351'
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))[0]

    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model.load_state_dict(model_dict)

    model.eval() # 评估模式

    # 2|2D数据读入
    print('|2-Loading data',end="")
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 帧数
    np_3d_pose = np.zeros((video_length, 17, 3)) # 3D 坐标储存位置

    # 3|每一帧进行处理
    print('|3-Generating',end="\n")

    for i in tqdm(range(video_length)): # tqdm 可视化进度条
        ret, img = cap.read() # 逐帧抽取
        img_size = img.shape # 图片尺寸

        ## input frames 设定处理的开始和结束帧数
        start = max(0, i - args.pad)
        end =  min(i + args.pad, len(keypoints[0])-1)
        input_2D_no = keypoints[0][start:end+1]
        
        ## padding 
        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != args.frames:
            if i < args.pad:
                left_pad = args.pad - i
            if i > len(keypoints[0]) - args.pad - 1:
                right_pad = i + args.pad - (len(keypoints[0]) - 1)

            input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')
        
        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[ :, :, 0] *= -1
        input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        
        input_2D = input_2D[np.newaxis, :, :, :, :]

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

        # N = input_2D.size(0)

        ## estimation 评估？
        output_3D_non_flip = model(input_2D[:, 0])
        output_3D_flip     = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        output_3D = output_3D[0:, args.pad].unsqueeze(1) 
        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()
        
        # ---------------
        np_3d_pose[i] = post_out
        # ---------------

        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0) # 
        post_out[:, 2] -= np.min(post_out[:, 2])

        input_2D_no = input_2D_no[args.pad]

        ## | 2D 图片输出
        image = show2Dpose(input_2D_no, copy.deepcopy(img))

        output_dir_2D = output_dir +'pose2D/'
        os.makedirs(output_dir_2D, exist_ok=True)
        cv2.imwrite(output_dir_2D + str(('%04d'% i)) + '_2D.png', image)

        ## | 3D 图片输出 gs
        fig = plt.figure( figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05) 
        ax = plt.subplot(gs[0], projection='3d')
        show3Dpose( post_out, ax)

        ## | 输出
        output_dir_3D = output_dir +'pose3D/'
        os.makedirs(output_dir_3D, exist_ok=True)
        plt.savefig(output_dir_3D + str(('%04d'% i)) + '_3D.png', dpi=200, format='png', bbox_inches = 'tight')
        plt.close()
        
    np.savez(f'{output_dir}pose_3d.npz', poses_3d=np_3d_pose)
    print('|4-Done!')

    #  all
    image_dir = 'results/' 
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(9.6, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = font_size)

        ## save
        output_dir_pose = output_dir +'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')


# TODO: MAIN fun
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser() # 
#     parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video') # 视频路径
#     parser.add_argument('--gpu', type=str, default='0', help='input video') # @
#     args = parser.parse_args() # 处理传入参数

#     # gpu 处理：
#     try: 
#         os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#     except Exception as e:
#         print(f"Worry as {e}")

#     # 路径处理
#     video_path = './demo/video/' + args.video
#     video_name = video_path.split('/')[-1].split('.')[0]
#     output_dir = './demo/output/' + video_name + '/'

#     cap = cv2.VideoCapture(video_path) # 视频读取 ：： 打开视频文件 → 连接解码器 → 创建资源句柄

#     # 生成结果
#     keypoints = get_pose2D(cap, output_dir) # 生成二维数据
#     get_pose3D(cap, keypoints, output_dir) # 生成三维数据
#     img2video(cap, output_dir) # 将图片合成视频
#     print('Generating demo successful!')

# py3.12 demo/vis_poseformer.py --video C-TJ1-try.mp4

# TODO:

try: 
    os.environ["CUDA_VISIBLE_DEVICES"] = 0
except Exception as e:
    print(f"Worry as {e}")

video_name = 'C-TJ1-try'
video_path = './demo/video/' + video_name + '.mp4' 
output_dir = './demo/output/' + video_name + '/'

cap = cv2.VideoCapture(video_path)

# keypoints = get_pose2D(cap, output_dir) # 生成二维数据

keypoints = np.load('./demo/output/C-TJ1-try/pose_2d.npz')['reconstruction']

get_pose3D(cap, keypoints, output_dir) # 生成三维数据
img2video(cap, output_dir) # 将图片合成视频

print('Generating demo successful!')

