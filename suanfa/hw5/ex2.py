# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

from numba import njit
import random
import math
from typing import Callable


@njit
def func(x: float) -> float:
    return math.sqrt(1 - x ** 2)


@njit
def HitorMiss(f: Callable[[float], float], n: int) -> float:
    """f: [0, 1]->[0, 1]"""
    k = 0  # 命中
    for _ in range(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if y <= f(x):  # (x, y)在阴影处
            k += 1
    return k / n


pi = math.pi
print("精确解: %f" % pi)
#
print()
print("n 计算的pi值 误差")
for i in range(9):
    n = 10 ** i
    res = 4 * HitorMiss(func, n)
    print(n, res, abs(pi - res))

"""Out
精确解: 3.141593

n 计算的pi值 误差
1 4.0 0.8584073464102069
10 3.2 0.05840734641020706
100 3.16 0.018407346410207026
1000 3.188 0.04640734641020705
10000 3.1424 0.000807346410206744
100000 3.14836 0.006767346410206709
1000000 3.14018 0.0014126535897931447
10000000 3.1423656 0.0007729464102070871
100000000 3.14156932 2.3333589793228526e-05
"""
