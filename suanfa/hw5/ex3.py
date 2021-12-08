# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

from numba import njit
import random
from typing import Callable


@njit
def func(x: float) -> float:
    return x ** 2 - 1


def func_integral(x: float) -> float:
    return x ** 3 / 3 - x


@njit
def HitorMiss2(f: Callable[[float], float],
               a: float, b: float, c: float, d: float,
               n: int) -> float:
    """f: [a, b]->[c, d]"""
    k = 0  # 命中
    c, d = min(c, 0), max(d, 0)
    S = (d - c) * (b - a)
    #
    for _ in range(n):
        x = random.uniform(a, b)
        y = random.uniform(c, d)
        fx = f(x)
        #
        if 0 < y <= fx:  # (x, y)在阴影处
            k += 1
        if fx <= y < 0:
            k -= 1
    return k / n * S


a, b, c, d = -1, 3, func(0), func(3)
print("准确解: %f" % (func_integral(b) - func_integral(a)))
#
n = int(1e8)
res = HitorMiss2(func, a, b, c, d, n)
print("n: %d, 计算的积分值: %f" % (n, res))

"""Out
准确解: 5.333333
n: 100000000, 计算的积分值: 5.337512
"""
