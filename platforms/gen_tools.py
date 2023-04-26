# 生成工具箱

import sys
import numpy as np
import matplotlib.pyplot as plt
from filters.gmphd import *
from filters.common import clutter_intensity
from platforms.radars import Radar
from platforms.targets import Target
import random


def gen_batch_targets(batch: int, K=100, label_start=1, T_s=1) -> List:
    """
    batch: 要生成的目标数量
    K:     时长
    label_start: 起始的label
    T_s: 目标的采样频率
    """
    trajs = []
    [trajs.append([]) for _ in range(K)]
    targets = []
    for i in range(batch):
        birth_time = random.randint(0, 20)
        death_time = random.randint(80, K)
        target = Target(birth_time,
                        death_time,
                        T_s,
                        label_start)
        traj = target.genTraj()
        targets.append(target)
        # target.plotTraj()
        [trajs[j].append(traj[j]) for j in range(traj.shape[0])]

    return [trajs, targets]


