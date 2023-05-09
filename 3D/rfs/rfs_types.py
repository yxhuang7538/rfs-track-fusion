#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   types.py
@Time    :   2023/05/07 20:16:57
@Author  :   yxhuang 
@Desc    :   基本类的定义
'''

import numpy as np
from typing import List


class Distribution:
    """分布的基本类
    """
    def __init__(self) -> None:
        pass
    
    def pdf(self, x: np.ndarray):
        """计算x点处的概率密度值

        Args:
            x (np.ndarray): 输入参数
        """
        pass

    
class BayesianFilter:
    """贝叶斯类滤波器
    """
    def __init__(self) -> None:
        pass
    
    def initialization(self):
        """初始化函数
        """
        pass

    def predict(self):
        """预测函数
        """
        pass

    def update(self, z: List[np.ndarray]):
        """更新函数

        Args:
            z (List[np.ndarray]): k时刻的输入量测值
        """
        pass

    def state_extract(self):
        """状态提取
        """
        pass

    def run(self):
        """运行滤波器
        """
        pass
