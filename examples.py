import sys
import numpy as np
import matplotlib.pyplot as plt
from filters.gmphd import *
from filters.common import clutter_intensity
from platforms.radars import Radar
from platforms.targets import Target
from platforms.gen_tools import gen_batch_targets
import random


def example1():
    # --- 雷达定义 --- #
    n = 3  # 雷达数量
    radars = []
    radarsPos = np.array([[0, 0], [5000, 0], [0, 5000]])  # 雷达位置
    radarsTs = [1, 1, 1]  # 采样频率
    for i in range(n):
        radar = Radar(radarsPos[i, :], i, radarsTs[i])
        radars.append(radar)
    K = 100  # 观测时长

    # --- 目标生成 --- #
    m = 5  # 目标数量
    [trajs, targets] = gen_batch_targets(m, K)
    lc = 20  # 杂波参数

    # --- 生成量测并运行滤波器 --- #
    clutt_int_fun = \
        lambda z: clutter_intensity(z, lc, np.array([targets[0].posRange, targets[0].posRange]))
    X = []  # 所有雷达滤波后的数据
    for i in range(n):
        meas = []
        [meas.append([]) for _ in range(K)]
        Z = radars[i].gen_all_meas(trajs, lc)
        [[meas[j].append(radars[i].pol2cart(z)) for z in Z[j]] for j in range(len(Z))]
        gmphd = GM_PHD(radars[i], clutt_int_fun)
        X_collection = gmphd.run(meas)
        X.append(X_collection)

    plt.figure()
    color = ['blue', 'green', 'red']
    i = 0
    for single_X in X:
        for X_k in single_X:
            for x in X_k:
                plt.scatter(x[0] + radars[i].pos[0], x[1] + radars[i].pos[1], c=color[i], s=5)
        i = i + 1
    plt.show()
