# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

from typing import List
import random
import math
from numba import njit


@njit
def fast_pow(x: int, y: int, mod: int = None) -> int:
    """Ot(Log(Y))"""
    res = 1
    while y > 0:
        if y % 2 == 1:
            res *= x
        x *= x
        y //= 2
        if mod:
            res %= mod
            x %= mod
    return res


@njit
def test_prime(a: int, n: int) -> bool:
    """测试a是否属于B(n)

    :param a: 属于[2, n-2]
    :param n: 为奇数
    :return: True: 属于B(n)-(n为素数/强伪素数. 或称为强拟素数)
    """
    # 满足 `n-1=2^st`. 使得t为奇数
    s = 0
    t = n - 1
    while t % 2 == 0:
        s += 1
        t //= 2
    #
    x = fast_pow(a, t, n)
    if x == 1 or x == n - 1:
        return True
    #
    for i in range(1, s):
        x = x * x % n
        if x == n - 1:
            return True
    return False


@njit
def MillRab(n: int) -> bool:
    """n > 4奇"""
    a = random.randint(2, n - 2)  # [2...n-2]. 含
    return test_prime(a, n)


@njit
def RepeatMillRab(n, k):
    for _ in range(k):
        if MillRab(n) is False:
            return False
    return True


@njit
def get_prime_nums(n: int) -> List[int]:
    """确定性算法. 求[2...n]内的质数"""
    res = []
    n += 1
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    #
    for i in range(2, n):
        if not is_prime[i]:
            continue
        #
        res.append(i)
        for j in range(i * i, n, i):
            is_prime[j] = False
    return res


@njit
def log2(x: float) -> float:
    return math.log(x) / math.log(2)


@njit
def get_prime_nums_2(n: int) -> List[int]:
    """概率算法. 求[2...n]内的质数"""
    res = [2, 3]
    for i in range(5, n + 1, 2):
        if RepeatMillRab(i, int(math.floor(log2(i)))):
            res.append(i)
    return res


s_100 = set(get_prime_nums(100))
s_10000 = set(get_prime_nums(10000))
s_100_10000 = s_10000 - s_100
n = 10000

# 实验了10000次
error = 0
k = 10000
for i in range(k):
    s2 = set(get_prime_nums_2(n)) - s_100
    error += len(s2 ^ s_100_10000) / len(s_100_10000)
print(error / k)
