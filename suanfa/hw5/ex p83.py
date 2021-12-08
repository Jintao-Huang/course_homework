# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from typing import Optional, List, Set, Tuple
import random
import time
from numba import njit
import numpy as np


@njit
def QueensLV(n: int, k: int) -> Optional[Tuple[List[int], Set[int], Set[int], Set[int]]]:
    """若成功(未失败)返回解. 失败返回None.
    n: n皇后, k: 随机取k次.
    返回: col, diag45, diag135, ans"""
    # diag135: i+j=i2+j2: 冲突
    col, diag45, diag135 = set(), set(), set()
    ans = []
    for i in range(k):
        st = []  # s.t. 满足的
        for j in range(n):  # 列
            # 符合条件
            if j not in col and (i - j) not in diag45 and (i + j) not in diag135:
                st.append(j)
        nb = len(st)
        #
        if nb > 0:
            j = st[random.randint(0, nb - 1)]  # [0...nb-1]含. 随机选一个
            ans.append(j)
            col.add(j)
            diag45.add(i - j)
            diag135.add(i + j)
        else:
            return None

    return ans, col, diag45, diag135


@njit
def Backtrack(n: int, track: List[int],
              col: Set[int], diag45: Set[int], diag135: Set[int]) -> bool:
    if len(track) == n:
        return True
    #
    i = len(track)  # in [0, n)
    #
    for j in range(n):  #
        # 剪枝
        if j in col or i - j in diag45 or i + j in diag135:
            continue
        #
        col.add(j)
        diag45.add(i - j)
        diag135.add(i + j)
        track.append(j)
        #
        if Backtrack(n, track, col, diag45, diag135):
            return True
        #
        col.remove(j)
        diag45.remove(i - j)
        diag135.remove(i + j)
        track.pop()
    return False


@njit
def QueensLVBacktrace(n: int, k: int) -> Optional[List[int]]:
    res = QueensLV(n, k)
    if res is None:
        return None
    track, col, diag45, diag135 = res
    success = Backtrack(n, track, col, diag45, diag135)
    if not success:
        return None
    return track


@njit
def QueensSuccessT(n: int, k: int, st: int) -> None:
    """n: n皇后. k: 随机取k次. st. success times.
    返回: 耗时"""
    success = 0
    while True:
        if QueensLVBacktrace(n, k) is not None:
            success += 1
            if success == st:
                break
    return


# numba编译
n = 4
QueensSuccessT(n, n // 2, 1)
#
for n in range(12, 21):
    ts = []
    for i in range(n):
        t = time.time()
        QueensSuccessT(n, i, 1000)
        ts.append(time.time() - t)
    print(n, np.argmin(ts))
