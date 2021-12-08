# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 


from numba import njit
from numba.typed import List as _list
import random
import math
from typing import List, Tuple
import numpy as np


@njit
def uniform_list(X: List[int]) -> List[int]:
    n = len(X)
    i = random.randint(0, n - 1)  # [0...n-1]
    return X[i]


@njit
def SetCount(X: List[int]) -> int:
    k = 0
    S = set()
    x = uniform_list(X)
    while x not in S:
        k += 1
        S.add(x)
        x = uniform_list(X)
    return int(round(2 * k * k / math.pi))


@njit
def SetCount_K(X: List[int], k: int) -> Tuple[int, int]:
    """运行k次取平均. 返回平均误差和每次误差的平均"""
    sum_ = []
    for i in range(k):
        sum_.append(SetCount(X))
    sum_ = np.array(sum_)
    return np.mean(sum_), np.std(sum_)


print("每个n跑100次")
print("n mean std")
for i in range(1, 8):
    n = 10 ** i
    mean, std = SetCount_K(_list(range(n)), 100)
    print("%d %d %d" % (n, mean, std))
