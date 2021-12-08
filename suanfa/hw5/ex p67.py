# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import random
from typing import List, Callable, Tuple, Any
import math
import numpy as np
import time

# 创建数据
n = 1000000
x = n // 2


def generate_date(n, seed=None, random=True) -> Tuple[List, List, int]:
    """

    :param n:
    :param seed:
    :param random:
    :return: val, ptr, head
    """
    np.random.seed(seed)
    if random:
        val = np.random.permutation(n)
    else:
        val = np.arange(n)
    ptr = np.argsort(val)
    head = ptr[0]
    ptr[ptr.copy()] = np.r_[ptr[1:], -1]
    return list(val), list(ptr), head


val, ptr, head = generate_date(n, 30, False)


def test_time(f: Callable, *args, **kwargs) -> Tuple[float, Any]:
    t = time.time()
    res = f(*args, **kwargs)
    return time.time() - t, res


def Search(x: int, i: int) -> int:
    """假设x在表中. 从i位置在有序表中开始查找x."""
    while x > val[i]:  # == 则跳出循环
        i = ptr[i]
    return i


def A(x: int, head: int) -> int:
    return Search(x, head)


def D(x: int, head: int) -> int:
    i = random.randint(0, n - 1)
    y = val[i]
    if x < y:
        return Search(x, head)
    elif x > y:
        return Search(x, ptr[i])
    else:
        return i


def search_sqrt_n(x: int, i_list: List[int]) -> int:
    """假设x在表中. 从i_list中找不大于x的最大整数y相应的下标i, 从i处继续顺序查找."""
    i = 0
    max_ = int(-1e9)
    for j in i_list:
        y = val[j]
        if max_ < y <= x:  # 最接近x, 并<=x
            i = j
            max_ = y
    return Search(x, i)


def B(x: int) -> int:
    """取前sqrt(n)个元素作为i_list"""
    sqrt_n = int(math.sqrt(n))  # 下底
    i_list = list(range(sqrt_n))
    return search_sqrt_n(x, i_list)


def choice(x: List[int], k: int) -> List[int]:
    """在x中不放回(replace=False)的随机取k个"""
    n = len(x)
    for i in range(k):
        idx = random.randint(i, n - 1)  # [0...n-1]. 含
        x[i], x[idx] = x[idx], x[i]
    return x[:k]


def C(x: int) -> int:
    """随机取sqrt(n)个元素作为i_list"""
    sqrt_n = int(math.sqrt(n))  # 下底
    i_list = choice(list(range(n)), sqrt_n)
    return search_sqrt_n(x, i_list)


ta, res_a = test_time(A, x, head)
td, res_d = test_time(D, x, head)
tb, res_b = test_time(B, x)
tc, res_c = test_time(C, x)

print("查找x: %d, n: %d. (顺序)" % (x, n))
print("A |Time: %.6f |Result: %d" % (ta, val[res_a]))
print("D |Time: %.6f |Result: %d" % (td, val[res_d]))
print("B |Time: %.6f |Result: %d" % (tb, val[res_b]))
print("C |Time: %.6f |Result: %d" % (tc, val[res_c]))
print()
#
val, ptr, head = generate_date(n, 42, True)
ta, res_a = test_time(A, x, head)
td, res_d = test_time(D, x, head)
tb, res_b = test_time(B, x)
tc, res_c = test_time(C, x)

print("查找x: %d, n: %d. (随机)" % (x, n))
print("A |Time: %.6f |Result: %d" % (ta, val[res_a]))
print("D |Time: %.6f |Result: %d" % (td, val[res_d]))
print("B |Time: %.6f |Result: %d" % (tb, val[res_b]))
print("C |Time: %.6f |Result: %d" % (tc, val[res_c]))

"""Out
查找x: 500000, n: 1000000. (顺序)
A |Time: 0.095720 |Result: 500000
D |Time: 0.054885 |Result: 500000
B |Time: 0.103692 |Result: 500000
C |Time: 0.022937 |Result: 500000

查找x: 500000, n: 1000000. (随机)
A |Time: 0.247916 |Result: 500000
D |Time: 0.250041 |Result: 500000
B |Time: 0.000998 |Result: 500000
C |Time: 0.025151 |Result: 500000
"""
