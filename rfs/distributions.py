#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   distributions.py
@Time    :   2023/05/07 20:41:47
@Author  :   yxhuang 
@Desc    :   实现一些RFS中会用到的分布类
'''

from .rfs_types import Distribution
from scipy.stats import multivariate_normal
import numpy as np
from typing import List
from scipy import stats
from copy import deepcopy as cdp


class LabelGaussianMixtureModel(Distribution):
    def __init__(self,
                 w: List[float],
                 m: List[np.ndarray],
                 P: List[np.ndarray],
                 L: List[np.ndarray]) -> None:
        super().__init__()
        """LGMM的基本类

        Args:
            w (List[float]): 初始权重
            m (List[np.ndarray]): 初始均值
            P (List[np.ndarray]): 初始协方差
            L (List[np.ndarray]): 初始标签 
        
        标签形式 (nc, nm, t) -> k时刻(高斯项确认次数，高斯项漏检次数，唯一数字ID)
        """
        self.w, self.m, self.P, self.L = w, m, P, L

        self.J = len(w) # 高斯项的个数

    def pdf(self, x: np.ndarray) -> float:
        """给定点x，计算其LGMM下的强度

        Args:
            x (np.ndarray): 给定的点

        Returns:
            float: 强度值
        """
        Dx = 0  # x点的强度值
        for i in range(len(self.J)):
            Dx += self.w[i] * multivariate_normal.pdf(x, self.m[i], self.P[i])
        return Dx

    def pdf_i(self, x: np.ndarray, i: int) -> float:
        """计算第i个高斯分量单独的强度

        Args:
            x (np.ndarray): 给定的状态
            i (int): 高斯分量的索引

        Returns:
            float: 第i个高斯分量的强度
        """
        return self.w[i] * multivariate_normal.pdf(x, self.m[i], self.P[i])
    
    def pdf_i_list(self, x: np.ndarray) -> List[float]:
        """将每个高斯分量的强度作为列表返回

        Args:
            x (np.ndarray): 给定状态

        Returns:
            List[float]: 高斯分量强度列表
        """
        Dx = []
        for i in range(len(range(self.J))):
            Dx.append(self.pdf_i(x, i))
        return Dx
    
    def merge(self, LGMM):
        """和给定的LGMM进行合并

        Args:
            LGMM (LabelGaussianMixtureModel): 要合并的LGMM
        """
        [self.w.append(w_) for w_ in LGMM.w]
        [self.m.append(m_) for m_ in LGMM.m]
        [self.P.append(P_) for P_ in LGMM.P]
        [self.L.append(L_) for L_ in LGMM.L]
        return cdp(LabelGaussianMixtureModel(self.w, self.m, self.P, self.L))

class Poisson(Distribution):
    def __init__(self,
                 lc: int) -> None:
        """泊松分布

        Args:
            lc (int): 泊松分布的参数λ
        """
        super().__init__()
        self.lc = lc

    def rvs(self) -> int:
        """生成服从该泊松分布的随机数

        Returns:
            int: 随机数
        """
        return stats.poisson.rvs(mu=self.lc, size=1)
    
    def cdf(self, z: np.ndarray, scenes_range: List[np.ndarray]) -> float:
        """杂波强度函数

        Args:
            z (np.ndarray): k时刻量测
            scenes_range (np.ndarray): 场景范围

        Returns:
            float: 杂波强度
        """
        part1 = 1.0
        for i in range(z.shape[0]):
            part1 *= (scenes_range[i][1] - scenes_range[i][0])
        return (self.rvs() / part1)[0]


