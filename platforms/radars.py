# 生成不同的场景例子
from typing import Dict, List
import numpy as np
import random
import math
from platforms.targets import Target
import matplotlib.pyplot as plt


class Radar:
    def __init__(self, pos: np.ndarray, id: int, T_s: float):
        self.pos = pos  # 雷达位置
        self.id = id  # 雷达的编号
        self.T_s = T_s  # 雷达的采样周期

        # --- 量测范围参数 ---#
        self.R_min = 0
        self.R_max = 60000
        self.Azi_min = 0 * np.pi / 180
        self.Azi_max = 360 * np.pi / 180

        self.p_s = 0.99  # 目标存活概率
        self.p_d = 0.98  # 检测概率

        self.dim_z = 2  # 量测维度
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])  # 观测矩阵
        self.sigma_R = random.randint(5, 8)  # 距离标准差
        self.sigma_Azi = random.uniform(0.1, 0.3) * np.pi / 180  # 方位角标准差
        self.R = np.array([[self.sigma_R ** 2, 0], [0, self.sigma_Azi ** 2]])  # 量测噪声协方差矩阵

        Q = np.zeros((4, 4))
        I_2 = np.eye(2)
        Q[0:2, 0:2] = (self.T_s ** 4) / 4 * I_2
        Q[0:2, 2:] = (self.T_s ** 3) / 2 * I_2
        Q[2:, 0:2] = (self.T_s ** 3) / 2 * I_2
        Q[2:, 2:] = (self.T_s ** 2) * I_2
        # 过程噪声
        sigma_w = 5.
        self.Q = Q * (sigma_w ** 2)

    def gen_all_meas(self, targets: List[List[np.ndarray]], lc: int) -> List[List[np.ndarray]]:
        """
        产生所有时刻的量测值
        :param lc: 杂波
        :param targets: 每个时刻的目标真值
        :return: 量测集合
        """
        meas = []
        for X in targets:
            m = []
            for state in X:
                if np.random.rand() <= self.p_d:  # 目标存活 被检测到
                    z = self.cart2pol(state) + np.random.multivariate_normal(np.zeros(self.H.shape[0]),
                                                                             self.R)
                    m.append(z)
            # 加入杂波
            for i in range(np.random.poisson(lc)):
                R = random.uniform(self.R_min, self.R_max)
                Azi = random.uniform(self.Azi_min, self.Azi_max)
                m.append(np.array([R, Azi]))
            meas.append(m)

        return meas

    def cart2pol(self, state: np.ndarray) -> np.ndarray:
        R = np.linalg.norm(self.pos - state[0:2])
        # Azi = (90 - math.degrees(math.atan2(state[1] - self.pos[1],
        #                                     state[0] - self.pos[0])) + 360) % 360
        # Azi = Azi * np.pi / 360
        Azi = np.arctan2(state[1] - self.pos[1], state[0] - self.pos[0])

        return np.array([R, Azi])

    def pol2cart(self, z: np.ndarray) -> np.ndarray:
        tmp = [z[0] * np.cos(z[1]), z[0] * np.sin(z[1])]
        tmp = np.array(tmp)
        return tmp

    def plot_all_meas(self, meas):
        plt.figure()
        for zk in meas:
            for z in zk:
                z_cart = self.pol2cart(z)
                plt.scatter(z_cart[0], z_cart[1], c='blue')
        plt.show()


if __name__ == "__main__":
    radar = Radar(np.array([1, 2]), 1, 1)
    target = Target(0, 100, 1, 1)
    traj = target.genTraj()
    target.plotTraj()
    targets = []
    [targets.append([traj[i, :]]) for i in range(traj.shape[0])]

    meas = radar.gen_all_meas(targets, 10)

    radar.plot_all_meas(meas)
