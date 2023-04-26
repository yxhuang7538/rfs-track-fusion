# ------------------------------ #
# 滤波器的基本组件，基本类
# 一些公用的函数
# ------------------------------ #

import numpy as np
from scipy.stats import multivariate_normal
from typing import List, Dict, Any
from copy import deepcopy as cdp


class GaussianMixtureModel:
    def __init__(self, w: List[float],
                 m: List[np.ndarray],
                 P: List[np.ndarray]):
        """
        高斯混合模型类，每个GMM将描述一个目标
        :param w: 每个高斯模型的权重List
        :param m: 每个高斯模型的均值List
        :param P: 每个高斯模型的协方差List
        """
        self.w = w
        self.m = m
        self.P = P

    def mixture_value(self, x: np.ndarray):
        """
        给定状态x时，计算该点强度（即所有高斯分布的加权和）
        :param x: 给定的状态x
        :return: GMM值
        """
        sum_value = 0
        for i in range(len(self.w)):
            sum_value += self.w[i] * multivariate_normal.pdf(x, self.m[i], self.P[i])
        return sum_value

    def mixture_single_component_value(self, x: np.ndarray, i: int) -> float:
        """
        给定状态向量，求第i个高斯分量的值
        :param x: 给定的状态向量
        :param i: 高斯分量的index
        :return: SMM值
        """
        value = self.w[i] * multivariate_normal.pdf(x, self.m[i], self.P[i])
        return value

    def mixture_component_value_list(self, x: np.ndarray) -> List[float]:
        """
        给定状态向量，求所有高斯分量的值，但保存为List
        :param x: 给定状态向量
        :return: value List
        """
        val = []
        for i in range(len(self.w)):
            val.append(self.mixture_single_component_value(x, i))
        return val

    def copy(self):
        """
        复制一个GMM类
        :return:
        """
        new_w = cdp(self.w)
        new_m = cdp(self.m)
        new_P = cdp(self.P)
        new_gmm = GaussianMixtureModel(new_w, new_m, new_P)

        return new_gmm

    def merge(self, v):
        """
        合并两个gmm
        :param v: 合并后的gmm
        :return:
        """
        [self.w.append(w_) for w_ in v.w]
        [self.m.append(m_) for m_ in v.m]
        [self.P.append(P_) for P_ in v.P]
        new_gmm = GaussianMixtureModel(cdp(self.w), cdp(self.m), cdp(self.P))
        return new_gmm


def get_inv_P(P_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    计算协方差列表的每个协方差的逆
    :param P_list: 协方差列表
    :return: 协方差逆列表
    """
    invP = []
    [invP.append(np.linalg.inv(P)) for P in P_list]
    return invP


class Filter:
    def __init__(self, radar, clutt_int_fun):
        """
        filter基类，定义基本的滤波方法
        x[k] = Fx[k-1] + w[k-1]
        z[k] = Hx[k] + v[k]
        :param model:
            keys: value
            F: 状态转移矩阵
            H: 观测矩阵
            Q: 过程噪声协方差矩阵 w[k-1]的
            R: 量测噪声协方差矩阵 v[k]的
            p_d: 检测率
            p_s: 存活率
            TODO 衍生暂不考虑，未来考虑加入
            clutt_int_fun: 杂波强度
            T, U, Jmax: 模型剪枝参数
            birth_GM: 出生目标的高斯混合模型
        """
        self.p_s = radar.p_s
        self.flag = "cv"
        self.T_s = radar.T_s
        self.F = self.ssm(self.flag)
        self.p_d = radar.p_d
        self.H = radar.H
        self.R = radar.R
        self.Q = radar.Q
        self.clutter_density_func = clutt_int_fun

        self.T = 1e-5
        self.U = 4.
        self.Jmax = 100

    def kalman_predict(self, v: GaussianMixtureModel) -> GaussianMixtureModel:
        """
        基本的用kalman滤波预测某个GMM强度
        :param v: 要预测的GMM强度
        :return: 预测后的GMM
        """
        w = []
        m = []
        P = []
        [w.append(weight * self.p_s) for weight in v.w]
        [m.append(self.F @ mean) for mean in v.m]
        [P.append(self.Q + self.F @ cov @ self.F.T) for cov in v.P]
        v_predict = GaussianMixtureModel(w, m, P)

        return v_predict

    def kalman_update(self, v: GaussianMixtureModel, Z: List[np.ndarray]) -> GaussianMixtureModel:
        """
        基本的kalman更新步骤，仅会对m和p进行更新
        :param v: 预测强度
        :param Z: 当前步的量测
        :return: 后验强度
        """
        w = []
        m = []
        P = []
        [w.append(weight * self.p_d) for weight in v.w]
        [m.append(self.H @ mean) for mean in v.m]
        [P.append(self.R + self.H @ cov @ self.H.T) for cov in v.P]
        v_residual = GaussianMixtureModel(w.copy(), m.copy(), P.copy())
        invP = get_inv_P(v_residual.P)

        K = []  # 卡尔曼增益
        P_kk = []
        for i in range(len(v_residual.w)):
            k = v.P[i] @ self.H.T @ invP[i]
            K.append(k)
            P_kk.append(v.P[i] - k @ self.H @ v.P[i])

        v_copy = cdp(v)
        w = (np.array(v_copy.w) * (1 - self.p_d)).tolist()
        m = v_copy.m
        P = v_copy.P

        for z in Z:
            values = v_residual.mixture_component_value_list(z)
            normalization_factor = np.sum(values) + self.clutter_density_func(z)
            for i in range(len(v_residual.w)):
                w.append(values[i] / normalization_factor)
                m.append(v.m[i] + K[i] @ (z - v_residual.m[i]))
                P.append(cdp(P_kk[i]))

        return GaussianMixtureModel(w, m, P)

    def pruning(self, v: GaussianMixtureModel) -> GaussianMixtureModel:
        """
        剪枝某个v，防止产生太多的分量
        :param v: 后验强度
        :return: 剪枝后的后验强度
        """
        # 按阈值进行剪枝
        I = (np.array(v.w) > self.T).nonzero()[0]
        w = [v.w[i] for i in I]
        m = [v.m[i] for i in I]
        P = [v.P[i] for i in I]
        v = GaussianMixtureModel(w, m, P)
        I = (np.array(v.w) > self.T).nonzero()[0].tolist()

        # 合并那些接近的高斯分量
        invP = get_inv_P(v.P)
        vw = np.array(v.w)
        vm = np.array(v.m)
        w = []
        m = []
        P = []
        while len(I) > 0:
            j = I[0]
            for i in I:
                if vw[i] > vw[j]:
                    j = i
            L = []
            for i in I:
                if (vm[i] - vm[j]) @ invP[i] @ (vm[i] - vm[j]) <= self.U:
                    L.append(i)
            w_new = np.sum(vw[L])
            m_new = np.sum((vw[L] * vm[L].T).T, axis=0) / w_new
            P_new = np.zeros((m_new.shape[0], m_new.shape[0]))
            for i in L:
                P_new += vw[i] * (v.P[i] + np.outer(m_new - vm[i], m_new - vm[i]))
            P_new /= w_new
            w.append(w_new)
            m.append(m_new)
            P.append(P_new)
            I = [i for i in I if i not in L]

        # 按照数量进行截断

        if len(w) > self.Jmax:
            L = np.array(w).argsort()[-self.Jmax:]
            w = [w[i] for i in L]
            m = [m[i] for i in L]
            P = [P[i] for i in L]

        return GaussianMixtureModel(w, m, P)

    def state_estimation(self, v: GaussianMixtureModel, w_t: float) -> List[np.ndarray]:
        """
        状态提取
        :param w_t: 提取阈值
        :param v: 要提取的gmm强度
        :return: 提取后的状态
        """
        X = []
        for i in range(len(v.w)):
            if v.w[i] >= w_t:
                for j in range(int(np.round(v.w[i]))):
                    X.append(v.m[i])
        return X

    def ssm(self, flag: str) -> np.ndarray:
        """
        运动模型库
        :param flag:要选取的运动模型 cv ct
        :return:运动模型
        """
        if flag == "cv":
            F = np.array([[1, 0, self.T_s, 0], [0, 1, 0, self.T_s],
                          [0, 0, 1, 0], [0, 0, 0, 1]])
        else:
            # ct
            F = np.array([[1, 0, self.T_s, 0], [0, 1, 0, self.T_s],
                          [0, 0, 1, 0], [0, 0, 0, 1]])

        return F

    def run(self, Z: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """
        给定量测集合，运行滤波器
        :param Z: 给定的量测集合
        :return: 滤波状态集合
        """
        X = []
        print("请先自定义run方法")
        return X


def clutter_intensity(z: np.ndarray, lc: int, scenes_range: np.ndarray) -> float:
    """
    杂波强度函数，杂波将在预设场景范围内均匀分布
    :param z: k时刻量测
    :param lc: 每个时间步长的平均误检数量
    :param scenes_range: 场景范围，shape为 (量测状态维度， 2)
    :return: 杂波强度
    """
    part1 = 1.0
    for i in range(z.shape[0]):
        part1 *= (scenes_range[i][1] - scenes_range[i][0])
    return lc / part1
