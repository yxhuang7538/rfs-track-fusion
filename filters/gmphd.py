import numpy as np
from scipy.stats import multivariate_normal
from typing import List, Dict, Any
import copy


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
        if scenes_range[i][0] <= z[i] <= scenes_range[i][1]:
            part1 *= (scenes_range[i][1] - scenes_range[i][0])
        else:
            part1 *= 0.0
    return lc / part1


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
        给定状态x时，计算GMM值
        :param x: 给定的状态x
        :return: GMM值
        """
        sum_value = 0
        for i in range(len(self.w)):
            sum_value += self.w[i] * multivariate_normal.pdf(x, self.m[i], self.P[i])
        return sum_value

    def mixture_single_component_value(self, x: np.ndarray, i: int) -> float:
        """
        给定状态向量，求第i各高斯分量的值
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
        new_w = copy.deepcopy(self.w)
        new_m = copy.deepcopy(self.m)
        new_P = copy.deepcopy(self.P)
        new_gmm = GaussianMixtureModel(new_w, new_m, new_P)

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


def get_det_P(P_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    计算协方差列表的每个协方差的行列式
    :param P_list: 协方差列表
    :return: 协方差行列式列表
    """
    detP = []
    [detP.append(np.linalg.det(P)) for P in P_list]
    return detP


class GmphdFilter:
    def __init__(self, model: Dict[str, Any]):
        """
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
        self.p_s = model['p_s']
        self.F = model['F']
        self.Q = model['Q']
        self.birth_GM = model['birth_GM']
        self.p_d = model['p_d']
        self.H = model['H']
        self.R = model['R']
        self.clutter_density_func = model['clutt_int_fun']
        self.T = model['T']
        self.U = model['U']
        self.Jmax = model['Jmax']

    def prediction(self, v: GaussianMixtureModel) -> GaussianMixtureModel:
        """
        GM-PHD每步的预测过程
        :param v: 上一步的GMM强度v
        :return: 预测的强度v
        """
        # 预测v = 新生v + 存活v + 衍生v（后续再加）

        # step1 预测新生
        v_birth = copy.deepcopy(self.birth_GM)  # TODO 自适应新生目标

        # step2 预测存活
        w = []
        m = []
        P = []
        [w.append(weight * self.p_s) for weight in v.w]
        [m.append(self.F @ mean) for mean in v.m]
        [P.append(self.Q + self.F @ cov @ self.F.T) for cov in v.P]
        v_s = GaussianMixtureModel(w, m, P)

        # step3 预测衍生 TODO

        # step4 合并
        gmm_predict = GaussianMixtureModel(v_birth.w + v_s.w,
                                           v_birth.m + v_s.m,
                                           v_birth.P + v_s.P)

        return gmm_predict

    def correction(self, v: GaussianMixtureModel, Z: List[np.ndarray]) -> GaussianMixtureModel:
        """
        更新步骤
        :param Z: 当前步的量测信息
        :param v: 预测强度
        :return: 后验强度
        """
        # step1 漏检目标
        w = []
        m = []
        P = []
        [w.append(weight * self.p_d) for weight in v.w]
        [m.append(self.H @ mean) for mean in v.m]
        [P.append(self.R + self.H @ cov @ self.H.T) for cov in v.P]
        v_residual = GaussianMixtureModel(w.copy(), m.copy(), P.copy())
        invP = get_inv_P(v_residual.P)

        # step2 构建PHD更新分量
        K = []
        P_kk = []
        for i in range(len(v_residual.w)):
            k = v.P[i] @ self.H.T @ invP[i]
            K.append(k)
            P_kk.append(v.P[i] - k @ self.H @ v.P[i])

        v_copy = copy.deepcopy(v)
        w = (np.array(v_copy.w) * (1 - self.p_d)).tolist()
        m = v_copy.m
        P = v_copy.P

        # step3 卡尔曼更新
        for z in Z:
            values = v_residual.mixture_component_value_list(z)
            normalization_factor = np.sum(values) + self.clutter_density_func(z)
            for i in range(len(v_residual.w)):
                w.append(values[i] / normalization_factor)
                m.append(v.m[i] + K[i] @ (z - v_residual.m[i]))
                P.append(copy.deepcopy(P_kk[i]))

        return GaussianMixtureModel(w, m, P)

    def pruning(self, v: GaussianMixtureModel) -> GaussianMixtureModel:
        """
        剪枝，防止产生太多的分量
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

    def state_estimation(self, v: GaussianMixtureModel) -> List[np.ndarray]:
        """
        状态提取
        :param v:
        :return:
        """
        X = []
        for i in range(len(v.w)):
            if v.w[i] >= 0.5:
                for j in range(int(np.round(v.w[i]))):
                    X.append(v.m[i])
        return X

    def run(self, Z: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """
        给定量测集合，运行GM-PHD
        :param Z: 给定的量测集合
        :return: 滤波状态集合
        """
        X = []
        v = GaussianMixtureModel([], [], [])
        for z in Z:
            v = self.prediction(v)
            v = self.correction(v, z)
            v = self.pruning(v)
            x = self.state_estimation(v)
            X.append(x)
        return X
