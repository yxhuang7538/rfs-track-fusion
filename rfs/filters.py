#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   filters.py
@Time    :   2023/05/07 20:32:18
@Author  :   yxhuang 
@Desc    :   定义各种基于贝叶斯的滤波器
'''

from .rfs_types import BayesianFilter
from .distributions import LabelGaussianMixtureModel as LGMM
import numpy as np
from typing import List
import yaml
from copy import deepcopy as cdp
from tqdm import tqdm


class LGMPHD_2D(BayesianFilter):
    def __init__(self) -> None:
        """标签GM-PHD滤波器，目标状态为2D点云形式
            X=[x, vx, y, vy].T
            Z=[r, phi].T
        """
        super().__init__()
        self.p_s = 0.99    # 存活概率
        self.p_d = 0.98    # 检测概率
        self.T_s = 1       # 采样频率

        self.F = self._CV()                 # 状态转移矩阵
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])   # 观测矩阵（量测为直角坐标情况）
        
        self.R = np.zeros([2, 2])           # 量测噪声协方差矩阵（量测为极坐标）
        self.Q = np.zeros([4, 4])           # 过程噪声协方差矩阵

        # 剪枝-合并-截断 参数
        self.threshold_pruning = 1e-5       # 剪枝阈值
        self.threshold_merge = 4.           # 合并阈值
        self.threshold_cutoff = 100         # 截断阈值

        # 场景范围
        self.r_min = 0                                # 距离范围
        self.r_max = 2000
        self.phi_min = -180 * np.pi / 180           # 方位角范围
        self.phi_max = 180 * np.pi / 180

        # 目标
        self.id = 1     # 标签里的最后一位唯一标志位

    def _CV(self):
        """CV运动转移模型 2D
        """
        T = self.T_s
        f = np.array([[1, T], [0, 1]])
        z = np.zeros([2, 2])
        F = np.block([[f, z], [z, f]])
        return F
    
    def config(self, cfg_path: str) -> None:
        """加载配置文件，更改默认参数

        Args:
            cfg_path (str): 配置文件路径
        """
        with open(cfg_path, 'r') as stream:
            cfg = yaml.safe_load(stream)

        # 滤波器基础参数
        self.p_s, self.p_d, self.T_s = cfg['p_s'], cfg['p_d'], cfg['T_s']

        # 设置观测噪声协方差矩阵
        sigma_r, sigma_phi = cfg['sigma_r'], cfg['sigma_phi']
        self.set_R(sigma_r, sigma_phi)

        # 设置过程噪声协方差矩阵
        sigma_v = cfg['sigma_a']
        self.set_Q(sigma_v)

        # 剪枝-合并-截断 参数
        self.threshold_pruning = cfg['threshold_pruning']   # 剪枝阈值
        self.threshold_merge = cfg['threshold_merge']       # 合并阈值
        self.threshold_cutoff = cfg['threshold_cutoff']     # 截断阈值

        # 场景范围
        self.r_min, self.r_max = cfg['r_min'], cfg['r_max'] # 距离范围
        self.phi_min = cfg['phi_min'] * np.pi / 180     # 方位角范围 弧度
        self.phi_max = cfg['phi_max'] * np.pi / 180

    def set_R(self, sigma_r: float, sigma_phi: float) -> None:
        """生成观测噪声协方差矩阵

        Args:
            sigma_r (float): 距离标准差 （米）
            sigma_phi (float): 方位标准差 （角度）
        """
        sigma_phi *= (np.pi / 180)    # 转弧度
        self.R = np.eye(2) * np.array([sigma_r ** 2, sigma_phi ** 2])

    def set_Q(self, sigma_v: float) -> None:
        """设置过程噪声协方差矩阵

        Args:
            sigma_v (float): 速度标准差
        """
        q = np.array([[self.T_s ** 4 / 4, self.T_s ** 3 / 2],
                       [self.T_s ** 3 / 2, self.T_s ** 2]])
        zeros = np.zeros([2, 2])
        self.Q = np.block([[q, zeros], [zeros, q]]) * (sigma_v ** 2)

    def set_F(self, F: np.ndarray) -> None:
        """设置状态转移矩阵

        Args:
            F (np.ndarray): 状态转移矩阵
        """
        self.F = F
    
    def initialization(self, N: int) -> LGMM:
        """初始化，用N个高斯分量覆盖场景范围

        Args:
            N (int): 高斯分量的个数参数

        Returns:
            LGMM: 初始化后的LGMM
        """
        r = np.linspace(self.r_min, self.r_max, N)  # 距离分量
        dr = (self.r_max - self.r_min) / N          # 距离间隔
        phi = np.linspace(self.phi_min, self.phi_max, N) # 方位角分量
        w, m, P, L = [], [], [], []
        for r_ in r:
            for phi_ in phi:
                x, y = r_ * np.cos(phi_), r_ * np.sin(phi_) # 直角坐标系下位置
                w.append(0.03)
                m.append(np.array([x, 0, y, 0]).T)
                P.append(np.diag([dr, 100, dr, 100]))
                L.append(np.array([0, 0, self.id]))
                self.id += 1
        return LGMM(w, m, P, L)

    def predict(self, v: LGMM) -> LGMM:
        """预测某个LGMM中的分量，基础方法为KF

        Args:
            v (LGMM): 要预测的LGMM

        Returns:
            LGMM: 预测后的LGMM
        """
        w, m, P, L = [], [], [], []

        # 存活概率更新权值
        [w.append(weight * self.p_s) for weight in v.w]

        # 每个高斯项执行KF（基本预测方法）得到新的m和P
        [m.append(self.F @ mean) for mean in v.m]
        [P.append(self.Q + self.F @ cov @ self.F.T) for cov in v.P]

        # 标签和预测前的保持一致
        [L.append(label) for label in v.L]

        # 存活目标的LGMM
        v_predict = LGMM(w, m, P, L)

        return v_predict

    def predict_born(self, Z: List[np.ndarray]) -> LGMM:
        """量测驱动的新生目标预测

        Args:
            Z (List[np.ndarray]): k时刻量测

        Returns:
            LGMM: 新生目标LGMM
        """
        w, m, P, L = [], [], [], []
        for z in Z:
            w.append(1 / len(Z))
            m.append(np.array([z[0], 0, z[1], 0]).T)
            P.append(np.diag([100, 100, 100, 100]))
            L.append(np.array([0, 0, self.id]))
            self.id += 1
        return LGMM(w, m, P, L)

    def update(self, v: LGMM, Z: List[np.ndarray], cdf) -> LGMM:
        """更新算法，默认为基本的KF更新过程

        Args:
            v (LGMM): 预测后的LGMM
            Z (List[np.ndarray]): k时刻量测信息
            v_b (LGMM): 预测后的新生LGMM
            cdf (function): 杂波函数

        Returns:
            LGMM: 更新后的LGMM
        """
        # 更新后的LGMM包含漏检和检测部分
        # 漏检
        v_m = cdp(v)
        # 利用漏检概率更新权值
        w_m = []
        [w_m.append(weight * (1 - self.p_d)) for weight in v_m.w]
        
        # 均值、协方差、标签均沿用
        v_m.w = w_m

        # 检测
        v_d = cdp(v)
        w_d, m_d, P_d, L_d = [], [], [], []
        # 利用检测概率更新权值
        [w_d.append(weight * self.p_d) for weight in v_d.w]

        # 将m和P转换成量测的维度形式
        [m_d.append(self.H @ mean) for mean in v_d.m]
        R = np.eye(2) * self.R[0, 0]
        [P_d.append(R + self.H @ cov @ self.H.T) for cov in v_d.P]

        # 标签沿用
        [L_d.append(label) for label in v_d.L]

        # 构造未经过量测值z更新前的LGMM
        v_temp = LGMM(w_d, m_d, P_d, L_d)
        w_, m_, P_, L_ = [], [], [], []
        # 逐量测更新
        for z in Z:
            # 利用高斯分布似然函数qk(z)、杂波函数值更新权值
            qkz = v_temp.pdf_i_list(z)
            norm = np.sum(qkz) + cdf(z)
            # 利用KF的更新步骤更新m和P，L将沿用
            for i in range(v_temp.J):
                w_.append(qkz[i] / norm)
                K = v.P[i] @ self.H.T @ np.linalg.pinv(v_temp.P[i]) # kalman增益
                m_.append(v.m[i] + K @ (z - v_temp.m[i]))
                P_.append(v.P[i] - K @ self.H @ v.P[i])
                L_.append(v_temp.L[i])
        v_new = LGMM(w_, m_, P_, L_)
        v_d = cdp(v_new)

        # 合并漏检和检测部分
        v_update = v_m.merge(v_d)

        return v_update

    def update_born(self, v_b: LGMM, Z: List[np.ndarray], cdf) -> LGMM:
        """更新新生目标

        Args:
            v_b (LGMM): 更新前的新生目标LGMM
            v_s (LGMM): 更新后的存活目标LGMM
            Z (List[np.ndarray]): k时刻量测信息
            cdf (function): 杂波函数

        Returns:
            LGMM: 更新后的新生目标LGMM
        """
        # 检测
        v_d = cdp(v_b)
        w_d, m_d, P_d, L_d = [], [], [], []
        # 新生权值
        [w_d.append(weight) for weight in v_d.w]

        # 将m和P转换成量测的维度形式
        [m_d.append(self.H @ mean) for mean in v_d.m]
        [P_d.append(self.R + self.H @ cov @ self.H.T) for cov in v_d.P]

        # 标签沿用
        [L_d.append(label) for label in v_d.L]

        # 构造未经过量测值z更新前的LGMM
        v_temp = LGMM(w_d, m_d, P_d, L_d)
        w_, m_, P_, L_ = [], [], [], []
        # 逐量测更新
        for z in Z:
            # 利用高斯分布似然函数qk(z)、杂波函数值更新权值
            qkz = v_temp.pdf_i_list(z)
            norm = np.sum(qkz) + cdf(z)
            # 利用KF的更新步骤更新m和P，L将沿用
            for i in range(v_temp.J):
                w_.append(qkz[i] / norm)
                K = v_b.P[i] @ self.H.T @ np.linalg.pinv(v_temp.P[i]) # kalman增益
                m_.append(v_b.m[i] + K @ (z - v_temp.m[i]))
                P_.append(v_b.P[i] - K @ self.H @ v_b.P[i])
                L_.append(v_temp.L[i])
        v_new = LGMM(w_, m_, P_, L_)
        return v_new

    def pruning(self, v: LGMM) -> LGMM:
        """对高斯分量进行剪枝

        Args:
            v (LGMM): 剪枝前的高斯分量

        Returns:
            LGMM: 剪枝后的高斯分量
        """
        # 按给定阈值进行剪枝
        I = (np.array(v.w) > self.threshold_pruning).nonzero()[0]   # 大于阈值的高斯分量索引
        w, m, P, L = [v.w[i] for i in I], [v.m[i] for i in I], [v.P[i] for i in I], [v.L[i] for i in I]
        v_ = LGMM(w, m, P, L)
        return v_
    
    def merge(self, v: LGMM, threshold_realloc=0.1) -> LGMM:
        """合并高斯分量

        Args:
            v (LGMM): 合并前的LGMM
            threshold_realloc (float): 重分配门限 默认为0.1
        Returns:
            LGMM: 合并后的LGMM
        """
        I = list(range(v.J))    # 要合并的索引集合
        w_m, m_m, P_m, L_m = [], [], [], [] # 合并后的w m p L
        index_m = -1    # 合并后的高斯分量计数索引
        while len(I) > 0:
            # 寻找权值最大的分量索引
            j = v.w.index(max(v.w))

            # 求取符合合并阈值的索引
            I_merge = []
            w_list = []
            for i in I:
                if (v.m[i] - v.m[j]) @ np.linalg.pinv(v.P[j]) @ (v.m[i] - v.m[j]) <= self.threshold_merge:
                    I_merge.append(i)
                    w_list.append(v.w[i])
            
            # 建立前N个最大权值对应的序号集合W
            N = np.round(sum(w_list))
            # 求前N个最大权值
            tmp = cdp(v.w)
            W = []
            W_list = []
            for _ in range(int(N)):
                number = max(tmp)
                index = tmp.index(number)
                tmp[index] = 0
                W.append(index)
                W_list.append(v.w[index])
            
            # 开始合并
            if N >= 2 and np.array(w_list).all() > threshold_realloc:
                [w_m.append(v.w[m] * sum(w_list) / sum(W_list)) for m in W]
                [m_m.append(v.m[m]) for m in W]
                [P_m.append(v.P[m]) for m in W]
                [L_m.append(v.L[m]) for m in W]
                index_m += N
            else:
                w_m.append(sum(w_list))
                m_m.append(np.dot(np.array(v.w)[I_merge].T, np.array(v.m)[I_merge, :]) / sum(w_list))
                P_tmp = np.zeros([4, 4])
                for i in I_merge:
                    P_tmp += v.w[i] * v.P[i]
                P_m.append(P_tmp / sum(w_list))
                # 取权值最大的作为id
                wm = np.array(v.w)[I_merge].tolist()
                max_w_index = I_merge[wm.index(max(wm))]
                L_m.append(v.L[max_w_index])
                index_m += 1
            
            # 去除掉已合并的高斯分量
            w_, m_, P_, L_ = [], [], [], []
            [w_.append(v.w[i]) for i in range(v.J) if i not in I_merge]
            [m_.append(v.m[i]) for i in range(v.J) if i not in I_merge]
            [P_.append(v.P[i]) for i in range(v.J) if i not in I_merge]
            [L_.append(v.L[i]) for i in range(v.J) if i not in I_merge]
            v = LGMM(w_, m_, P_, L_)
            I = list(range(v.J))    # 要合并的索引集合
        
        return LGMM(w_m, m_m, P_m, L_m)

    def cutoff(self, v: LGMM) -> LGMM:
        """截断高斯分量

        Args:
            v (LGMM): 截断前的LGMM

        Returns:
            LGMM: 截断后的LGMM
        """
        if v.J > self.threshold_cutoff:
            tmp = cdp(v.w)
            I = []
            for _ in range(int(self.threshold_cutoff)):
                number = max(tmp)
                index = tmp.index(number)
                tmp[index] = 0
                I.append(index)
            w, m, P, L = [], [], [], []
            [w.append(v.w[i]) for i in I]
            [m.append(v.m[i]) for i in I]
            [P.append(v.P[i]) for i in I]
            [L.append(v.L[i]) for i in I]
            return LGMM(w, m, P, L)
        else:
            return v
    
    def label_manage(self, v: LGMM, threshold_death=3) -> LGMM:
        """管理标签
        首先，将权值小于 0.5 的增广高斯项中的漏检次数计数变量加 1，若漏检次数达
        到终止门限 则将该增广高斯项删掉。然后，将权值大于等于 0.5 的增广高斯项中
        的确认次数计数变量加 1 以及漏检次数计数变量置 0。
        Args:
            v (LGMM): 要管理标签的LGMM
            threshold_death (int, optional): 终止门限. Defaults to 3.
        Returns:
            LGMM: 管理标签后的LGMM
        """
        I = []  # 要移除的高斯分量索引
        for j in range(v.J):
            if v.w[j] < 0.5:
                v.L[j][1] += 1
                if v.L[j][1] >= threshold_death:
                    I.append(j)
            else:
                v.L[j][0] += 1
                v.L[j][1] = 0
        # 去除掉要移除的高斯分量
        w_, m_, P_, L_ = [], [], [], []
        [w_.append(v.w[i]) for i in range(v.J) if i not in I]
        [m_.append(v.m[i]) for i in range(v.J) if i not in I]
        [P_.append(v.P[i]) for i in range(v.J) if i not in I]
        [L_.append(v.L[i]) for i in range(v.J) if i not in I]
        v = LGMM(w_, m_, P_, L_)
        return v

    def state_extract(self, v: LGMM, threshold_confirm=2) -> List[List[np.ndarray]]:
        """状态提取
        将权值大于等于 0.5 的增广高斯项子集作为目标点迹集合输出到航迹管理模块。
        根据确认次数计算变量值和确认门限 将目标点迹集合中的点迹分为临时目标点
        迹集合和确认目标点迹集合。其中，确认目标点迹集合用于航迹维持和航迹终止，而
        临时目标点迹集合用于航迹起始。
        Args:
            v (LGMM): 要进行状态提取的LGMM
            threshold_confirm (int, optional): 确认门限. Defaults to 2.

        Returns:
            List[List[np.ndarray]]: [[确认目标点迹], [临时目标点迹]]
        """
        X_confirm = []          # 确认目标点迹
        L_confirm = []          # 确认目标的标签
        X_tmp = []              # 临时目标点迹
        L_tmp = []              # 临时目标的标签

        for i in range(v.J):
            if v.w[i] >= 0.5:
                if v.L[i][0] < threshold_confirm:   # 临时目标点迹
                    X_tmp.append(v.m[i])
                    L_tmp.append(v.L[i])
                else:                               # 确认目标点迹
                    X_confirm.append(v.m[i])
                    L_confirm.append(v.L[i])
        return [X_confirm, L_confirm, X_tmp, L_tmp]

    def run(self, Z: List[List[np.ndarray]], cdf) -> List[List[np.ndarray]]:
        """运行滤波器

        Args:
            Z (List[List[np.ndarray]]): 量测集合
            cdf 杂波函数

        Returns:
            List[List[np.ndarray]]: 滤波结果
        """
        X = []  # 滤波结果
        # step1 初始化
        v = self.initialization(4)
        for z in tqdm(Z):
            # step2 量测驱动新生目标
            v_b = self.predict_born(z)
            # step3 预测存活目标
            v_s = self.predict(v)
            # step4 更新存活目标
            v_s = self.update(v_s, z, cdf)
            # step5 更新新生目标
            v_b = self.update_born(v_b, z, cdf)
            # step6 对存活目标进行剪枝-合并-截断
            v_s = self.pruning(v_s)
            v_s = self.merge(v_s)
            v_s = self.cutoff(v_s)
            # step7 对存活目标进行标签管理
            v_s = self.label_manage(v_s)
            # step8 对存活目标进行状态提取
            [X_confirm, L_confirm, X_tmp, L_tmp] = self.state_extract(v_s)
            # step9 合并存活和新生
            v = v_s.merge(v_b)
            
            X.append([X_confirm, L_confirm, X_tmp, L_tmp])

        return X
    
    