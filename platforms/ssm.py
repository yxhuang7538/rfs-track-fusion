# 运动模型库

import numpy as np
import math
import random


class SSM:
    """
    运动模型类
    """

    def __init__(self, dim, T, sigmaV=5):
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
            # 改变顺序
            F = F[:, [0, 3, 1, 4, 2, 5]]
            F = F[[0, 3, 1, 4, 2, 5], :]
            return F

    def CA(self):
        if self.dim == 6:  # 2D
            F = np.array([[1, self.T, math.pow(self.T, 2) / 2, 0, 0, 0],
                          [0, 1, self.T, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, self.T, math.pow(self.T, 2) / 2],
                          [0, 0, 0, 0, 1, self.T],
                          [0, 0, 0, 0, 0, 1]])
            F = F[:, [0, 3, 1, 4, 2, 5]]
            F = F[[0, 3, 1, 4, 2, 5], :]
            return F

    def CT(self, w):
        if self.dim == 6:  # 2D
            F = np.array([[1, math.sin(w * self.T) / w, 0, 0, (math.cos(w * self.T) - 1) / w, 0],
                          [0, math.cos(w * self.T), 0, 0, -math.sin(w * self.T), 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, (1 - math.cos(w * self.T)) / w, 0, 1, math.sin(w * self.T) / w, 0],
                          [0, math.sin(w * self.T), 0, 0, math.cos(w * self.T), 0],
                          [0, 0, 0, 0, 0, 1]])
            F = F[:, [0, 3, 1, 4, 2, 5]]
            F = F[[0, 3, 1, 4, 2, 5], :]
            return F

    def selectOneF(self, w):
        flag = random.randint(1, self.numsF)
        if flag == 1:
            return self.CV()
        elif flag == 2:
            return self.CA()
        else:
            return self.CT(w)
