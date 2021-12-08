# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:


from numba import njit
import random
import math


@njit
def Darts(n: int) -> float:
    k = 0  # 命中
    for _ in range(n):
        x = random.uniform(0, 1)
        y = x
        if (x ** 2 + y ** 2) < 1:
            k += 1
    return 4.0 * k / n


n = int(1e9)
print(Darts(n) - 2 * math.sqrt(2))  # -4.350874619030165e-05
