#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   targets.py
@Time    :   2023/05/08 11:38:04
@Author  :   yxhuang 
@Desc    :   目标类
'''

import numpy as np
from typing import List
import random
import matplotlib.pyplot as plt

# 运动模型库

import numpy as np
import math
import random


class SSM:
    def __init__(self, dim, T):
        self.dim = dim  # 状态的维数 [x,vx,ax,y,vy,ay]
        self.T = T  # 时间间隔
        self.numsF = 3  # 运动状态个数

    def CV(self):
        if self.dim == 6:  # 2D
            F = np.array([[1, self.T, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, self.T, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0]])
            return F

    def CA(self):
        if self.dim == 6:  # 2D
            F = np.array([[1, self.T, math.pow(self.T, 2) / 2, 0, 0, 0],
                          [0, 1, self.T, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, self.T, math.pow(self.T, 2) / 2],
                          [0, 0, 0, 0, 1, self.T],
                          [0, 0, 0, 0, 0, 1]])
            return F

    def CT(self, w):
        if self.dim == 6:  # 2D
            F = np.array([[1, math.sin(w * self.T) / w, 0, 0, (math.cos(w * self.T) - 1) / w, 0],
                          [0, math.cos(w * self.T), 0, 0, -math.sin(w * self.T), 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, (1 - math.cos(w * self.T)) / w, 0, 1, math.sin(w * self.T) / w, 0],
                          [0, math.sin(w * self.T), 0, 0, math.cos(w * self.T), 0],
                          [0, 0, 0, 0, 0, 1]])
            return F

    def selectOneF(self, w):
        flag = random.randint(1, self.numsF)
        if flag == 1:
            return self.CV()
        elif flag == 2:
            return self.CA()
        else:
            return self.CT(w)

class Target:
    def __init__(self, birth_time: int, death_time: int, dt: float, label: int):
        self.birth_time = birth_time    # 出生时间
        self.death_time = death_time
        self.dt = dt
        self.label = label

        self.seq_len = int((death_time - birth_time) / dt)  # 目标存在时间

        self.dim_x = 6
        self.state = np.zeros([self.seq_len,
                               self.dim_x])  # 目标状态
        self.accRange = [-0.03, 0.03]  # 加速度范围
        self.posRange = [-2000, 2000]  # 位置范围（针对的是目标的出生位置）
        self.vRange = [-15, 15]  # 速度范围
        self.wRange = [-10, 10]  # 转向率范围

    def genState(self, pos=None, v=None, a=None):
        """
        生成初始状态
        :param a: 初始加速度
        :param v: 初始速度
        :param pos: 给定初始位置
        :return: 初始状态
        """
        if pos is None:
            x = random.randint(self.posRange[0], self.posRange[1])
            y = random.randint(self.posRange[0], self.posRange[1])
            pos = [x, y]
        if v is None:
            vx = random.uniform(self.vRange[0], self.vRange[1])
            vy = random.uniform(self.vRange[0], self.vRange[1])
            v = [vx, vy]
        if a is None:
            ax = random.uniform(self.accRange[0], self.accRange[1])
            ay = random.uniform(self.accRange[0], self.accRange[1])
            a = [ax, ay]

        # state = np.array([pos[0], pos[1], v[0], v[1], a[0], a[1]]).T    # 初始状态
        state = np.array([pos[0], v[0], a[0], pos[1], v[1], a[1]]).T
        return state

    def genTraj(self):
        """
        生成轨迹
        :return:无
        """
        ssm = SSM(self.dim_x, self.dt)

        self.state[0, :] = self.genState()

        w = (random.randint(self.wRange[0], self.wRange[1]) + 0.1) * np.pi / 180
        F = ssm.selectOneF(w)
        maneuvering_t = random.randint(1, self.seq_len)     # 机动时刻
        for k in range(1, self.seq_len):
            # TODO 探索一下转移噪声
            if k == maneuvering_t:
                F = ssm.selectOneF(w)
            self.state[k, :] = F @ self.state[k - 1, :]

        Traj = self.state
        return Traj

    def plotTraj(self):
        """
        绘制轨迹
        :return:
        """
        plt.figure()
        plt.plot(self.state[:, 0], self.state[:, 3], color='blue', marker='o', linestyle='dashed')
        plt.savefig("test.png")

    def getOnePoints(self, k):
        """
        输入时刻k，获取该时刻状态信息
        :param k:
        :return:
        """
        return self.state[k, :]

    def getOnePointsPos(self, k):
        """
        获取k时刻位置
        :param k:
        :return:
        """
        return [self.state[k, 0], self.state[k, 3]]


def gen_batch_targets(batch: int, K=100, label_start=1, T_s=1) -> List:
    """批量生成目标

    Args:
        batch (int): 批量大小
        K (int, optional): 观测时长. Defaults to 100.
        label_start (int, optional): 起始标签. Defaults to 1.
        T_s (int, optional): 观测频率. Defaults to 1.

    Returns:
        List: 轨迹list和目标对象list
    """
    trajs = []
    [trajs.append([]) for _ in range(K)]
    targets = []
    for _ in range(batch):
        birth_time = random.randint(0, int(0.2 * K))
        death_time = random.randint(int(0.8 * K), K)
        target = Target(birth_time,
                        death_time,
                        T_s,
                        label_start)
        traj = target.genTraj()
        targets.append(target)
        # target.plotTraj()
        [trajs[j].append(traj[j]) for j in range(traj.shape[0])]

    return [trajs, targets]

if __name__ == "__main__":
    target = Target(0, 100, 1, 1)
    target.genTraj()
    target.plotTraj()
