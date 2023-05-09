#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   examples.py
@Time    :   2023/05/09 20:38:39
@Author  :   yxhuang 
@Desc    :   样例demo
'''


from platforms.radars import Radar
from platforms.targets import Target, gen_batch_targets
from rfs.distributions import Poisson
from rfs.filters import LGMPHD_3D
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def example1():
    # 雷达定义
    n = 1  # 雷达数量
    radars = []
    radarsPos = np.array([[0, 0, 0]])  # 雷达位置
    radarsTs = [1]  # 采样频率

    cfg_path = "/media/hyx/工作盘/codes/rfs-track-fusion/3D/configs/standard.yaml"   # 配置文件路径
    for i in range(n):
        radar = Radar(radarsPos[i, :], i, radarsTs[i])
        radar.config(cfg_path)
        radars.append(radar)
    K = 100  # 观测时长

    # 目标生成
    m = 15   # 目标数量
    [trajs, targets] = gen_batch_targets(m, K)

    # 生成量测+滤波
    for i in range(n):
        lc = 10 # 杂波服从的泊松分布参数
        kz = Poisson(lc)    # 杂波泊松分布
        meas_cart = []  # 直角坐标系下量测
        [meas_cart.append([]) for _ in range(K)]
        meas_pol = radars[i].gen_all_trajs_meas(trajs, kz)  # 极坐标下量测
        [[meas_cart[j].append(radars[i].pol2cart(z)) for z in meas_pol[j]] for j in range(len(meas_pol))]

        # 定义滤波器
        lgmphd = LGMPHD_3D()
        lgmphd.config(cfg_path)
        cdf = lambda z: kz.cdf(z, np.array([[-radars[i].r_max, radars[i].r_max], 
                                            [-radars[i].r_max, radars[i].r_max],
                                            [-radars[i].r_max, radars[i].r_max]]))

        # 运行滤波器
        X = lgmphd.run(meas_cart, cdf)


    # 解决中文乱码
    plt.rcParams["font.sans-serif"]=["SimHei"]
    plt.rcParams["font.family"]="sans-serif"
    # 解决负号无法显示的问题
    plt.rcParams['axes.unicode_minus'] =False
    plt.figure(figsize=(8, 8))    # 动图
    ax = plt.axes(projection='3d')
    #plt.ion()
    image_list = []
    for k in range(K):
        traj_x, traj_y, traj_z = [], [], []
        for t in trajs[k]:
            traj_x.append(t[0])
            traj_y.append(t[3])
            traj_z.append(t[6])
        # type1 = plt.scatter(traj_x, traj_y, c='blue', s=5, marker='d', alpha=0.5, label='目标真值')
        type1 = ax.scatter3D(traj_x, traj_y, traj_z, c='blue', s=5, marker='d', alpha=0.5, label='目标真值')

        meas_x, meas_y, meas_z = [], [], []
        if len(meas_cart[k]) > 0:
            for meas_ in meas_cart[k]:
                meas_x.append(meas_[0])
                meas_y.append(meas_[1])
                meas_z.append(meas_[2])
        # type2 = plt.scatter(meas_x, meas_y, marker='x', c='gray', s = 5, alpha=0.5, label='杂波量测')
        type2 = ax.scatter3D(meas_x, meas_y, meas_z, marker='x', c='gray', s = 5, alpha=0.5, label='杂波量测')

        result_x, result_y, result_z = [], [], []
        X_confirm = X[k][0]
        if len(X_confirm) > 0:
            for x in X_confirm:
                result_x.append(x[0])
                result_y.append(x[2])
                result_z.append(x[4])
        # type3 = plt.scatter(result_x, result_y, c='red', s=12, label='跟踪结果')
        type3 = ax.scatter3D(result_x, result_y, result_z, c='red', s=12, label='跟踪结果')

        plt.legend((type1, type2, type3), (u'目标真值', u'杂波量测', '跟踪结果'),loc="upper right", fontsize=16)
        ax.set_xlabel("X-位置/(m)", fontsize=16)
        ax.set_ylabel("Y-位置/(m)", fontsize=16)
        ax.set_zlabel("Z-位置/(m)", fontsize=16)
        plt.xlim((-4000, 4000))
        plt.ylim((-4000, 4000))
        ax.set_zlim((-4000, 4000))
        #plt.pause(0.1)

    #plt.ioff()
        plt.savefig("test-2.png")
        image_list.append(imageio.imread('test-2.png'))
    imageio.mimsave('test-2.gif', image_list, duration=1)
