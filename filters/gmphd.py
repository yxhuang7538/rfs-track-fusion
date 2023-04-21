from common import Filter
from common import GaussianMixtureModel as GMM

import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm


class GM_PHD(Filter):
    def __init__(self, model):
        super().__init__(model)

    def meas_birth_v(self, Z: List[np.ndarray]) -> GMM:
        """
        量测驱动的新生模型
        :param Z: 量测值
        :return: 新生gmm
        """
        w = []
        m = []
        P = []
        [w.append(0.03) for _ in range(len(Z))]
        [m.append(self.H.T @ z) for z in Z]
        [P.append(np.diag([100, 100, 100, 100])) for _ in Z]

        return GMM(w, m, P)

    def meas_update_birth_v(self, Z: List[np.ndarray], X) -> GMM:
        """
        量测更新新生目标
        :param X: 当前状态
        :param Z: 量测值
        :return: 更新后的新生目标gmm
        """
        w = []
        m = []
        P = []
        [w.append(0.03) for z in Z]
        [m.append(self.H.T @ z) for z in Z]
        [P.append(np.diag([100, 100, 100, 100])) for z in Z]

        return GMM(w, m, P)

    def prediction(self, v: GMM) -> GMM:
        """
        GM-PHD滤波器的预测过程
        :param v: 上一步的强度
        :return: 预测强度
        """
        # 预测v = 存活v + 衍生v（后续再加）
        v_s = self.kalman_predict(v)
        return v_s

    def update(self, v: GMM, Z: List[np.ndarray]) -> GMM:
        """
        更新预测gmm
        :param Z: 当前步的量测
        :param v: 要更新的gmm强度
        :return: 更新后的gmm强度
        """
        v = self.kalman_update(v, Z)
        return v

    def run(self, Z: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """
        给定量测集合，运行GM-PHD
        :param Z: 给定的量测集合
        :return: 滤波状态集合
        """
        X = []
        v = GMM([], [], [])
        # step1 量测驱动新生 初始化
        v_b = self.meas_birth_v(Z[0])
        for z in tqdm(Z):
            # step2 预测存活目标
            v_s = self.prediction(v)
            # step3 更新存活目标
            v_s = self.update(v_s, z)
            # step4 状态估计
            x_s = self.state_estimation(v_s, 0.3)
            # step5 裁剪高斯分量
            v_s = self.pruning(v_s)
            # step6 更新新生目标
            v_b = self.meas_update_birth_v(z, x_s)
            # step7 合并新生和存活
            v = v_s.merge(v_b)

            X.append(x_s)
        return X
