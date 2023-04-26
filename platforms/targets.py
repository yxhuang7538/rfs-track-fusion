import numpy as np
from typing import List
import random
from platforms.ssm import SSM
import matplotlib.pyplot as plt


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
        self.accRange = [-3, 3]  # 加速度范围
        self.posRange = [-50000, 50000]  # 位置范围（针对的是目标的出生位置）
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

        state = np.array([pos[0], pos[1], v[0], v[1], a[0], a[1]]).T    # 初始状态
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
            self.state[k, :] = np.dot(F, self.state[k - 1, :])

        Traj = self.state
        return Traj

    def plotTraj(self):
        """
        绘制轨迹
        :return:
        """
        plt.figure()
        plt.plot(self.state[:, 0], self.state[:, 1], color='blue', marker='o', linestyle='dashed')
        plt.show()

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
        return [self.state[k, 0], self.state[k, 1]]


if __name__ == "__main__":
    target = Target(0, 100, 1, 1)
    target.genTraj()
    target.plotTraj()
