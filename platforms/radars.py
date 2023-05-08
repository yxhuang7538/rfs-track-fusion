#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   radars.py
@Time    :   2023/05/08 10:08:48
@Author  :   yxhuang 
@Desc    :   雷达类
'''

import numpy as np
import yaml
from rfs.distributions import Poisson
from typing import List
import random


class Radar:
    def __init__(self, pos: np.ndarray, id: int, T_s: float) -> None:
        """雷达类

        Args:
            pos (np.ndarray): 雷达位置
            id (int): 雷达编号
            T_s (float): 雷达采样频率
        """
        self.pos = pos  # 雷达位置
        self.id = id    # 雷达的编号
        self.T_s = T_s  # 雷达的采样周期

        self.p_d = 0.99 # 检测概率

        self.R = np.zeros([2, 2])           # 量测噪声协方差矩阵（量测为极坐标）
        self.Q = np.zeros([4, 4])           # 过程噪声协方差矩阵

        # 场景范围
        self.r_min = 0                                # 距离范围
        self.r_max = 2500
        self.theta_min = -180 * np.pi / 180              # 方位角范围
        self.theta_max = 180 * np.pi / 180

        self.all_meas = []                            # 所有量测
    
    def config(self, cfg_path: str) -> None:
        """加载配置文件，更改默认参数

        Args:
            cfg_path (str): 配置文件路径
        """
        with open(cfg_path, 'r') as stream:
            cfg = yaml.safe_load(stream)

        self.T_s, self.p_d = cfg['T_s'], cfg['p_d']

        # 设置观测噪声协方差矩阵
        sigma_r, sigma_theta = cfg['sigma_r'], cfg['sigma_theta']
        self.set_R(sigma_r, sigma_theta)

        # 设置过程噪声协方差矩阵
        sigma_a = cfg['sigma_a']
        self.set_Q(sigma_a)

        # 场景范围
        self.r_min, self.r_max = cfg['radar_r_min'], cfg['radar_r_max'] # 距离范围
        self.theta_min = cfg['radar_theta_min'] * np.pi / 180     # 方位角范围 弧度
        self.theta_max = cfg['radar_theta_max'] * np.pi / 180

    def set_R(self, sigma_r: float, sigma_theta: float) -> None:
        """生成观测噪声协方差矩阵

        Args:
            sigma_r (float): 距离标准差 （米）
            sigma_theta (float): 方位标准差 （角度）
        """
        sigma_theta *= (np.pi / 180)    # 转弧度
        self.R = np.eye(2) * np.array([sigma_r ** 2, sigma_theta ** 2])

    def set_Q(self, sigma_a: float) -> None:
        """设置过程噪声协方差矩阵

        Args:
            sigma_a (float): 加速度标准差
        """
        q = np.array([[self.T_s ** 4 / 4, self.T_s ** 3 / 2],
                       [self.T_s ** 3 / 2, self.T_s ** 2]])
        zeros = np.zeros([2, 2])
        self.Q = np.block([[q, zeros], [zeros, q]]) * (sigma_a ** 2)

    def gen_all_trajs_meas(self, trajs: List[List[np.ndarray]], kz: Poisson) -> List[List[np.ndarray]]:
        """为所有轨迹生成量测

        Args:
            trajs (List[List[np.ndarray]]): 所有目标
            kz (Poisson): 杂波函数

        Returns:
            List[List[np.ndarray]]: 量测集合
        """
        all_meas = []
        for traj in trajs:  # 取k时刻所有目标轨迹点
            meas = []
            for state_k in traj:
                if np.random.rand() <= self.p_d:    # 目标存活，被检测到
                    z = self.cart2pol(state_k) + np.random.multivariate_normal(np.zeros(2), self.R)
                    meas.append(z)
            # 加入杂波
            for _ in range(int(kz.rvs())):
                r = random.uniform(self.r_min, self.r_max)
                theta = random.uniform(self.theta_min, self.theta_max)
                meas.append(np.array([r, theta]))
            all_meas.append(meas)   # k时刻所有量测
        self.all_meas = all_meas
        return all_meas
    
    def cart2pol(self, state: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(self.pos - state[[0, 3]])
        theta = np.arctan2(state[3] - self.pos[1], state[0] - self.pos[0])

        return np.array([r, theta])

    def pol2cart(self, z: np.ndarray) -> np.ndarray:
        tmp = [z[0] * np.cos(z[1]), z[0] * np.sin(z[1])]
        tmp = np.array(tmp)
        return tmp

    
    
    