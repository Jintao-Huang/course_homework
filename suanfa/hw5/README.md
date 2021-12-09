# hw5

学号: 

姓名: 

代码地址: https://github.com/Jintao-Huang/course_homework/tree/main/suanfa/hw5



## 目录

[toc]



## Ex1 p20

题: 若将 y ← uniform(0, 1) 改为 y ← x, 则上述的算法估计的值是什么?

答: 

y ← uniform(0, 1) 改为 y ← x, 所以 $\frac{k}{n}=\frac{1}{\sqrt{2}}$​, 
所以算法估计的值 $\frac{4k}{n}=2\sqrt{2}$​​​.

代码: 

```python
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
```



## Ex2 p23

题: 在机器上用 $4\int_0^1{\sqrt{1-x^2}dx}$​ 估计π值, 给出不同的n值及精度.

答: 

由运行结果可得, n越大, 精度越高, 越接近$\pi$​值. 不同n值对应的精度(误差)见运行结果.

运行结果: 

```
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
```

代码:

```python
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
```



## EX3 p23

题: 设a, b, c和d是实数, 且a ≤ b, c ≤ d, f:[a, b] → [c, d] 是一个连续函数, 写一概率算法计算积分: 
$$
\int_a^b{f(x)dx}
$$
注意: 函数的参数是a, b, c, d, n和f, 其中f用函数指针实现, 请选一连续函数做实验, 并给出实验结果.

答:

我实验的是求解$\int_{-1}^3{x^2-1dx}$的结果. 其准确结果为$5.333333$, 我们运行了 $n=100000000$ 次, 得到了近似结果$5.337512$.

运行结果:

```
准确解: 5.333333
n: 100000000, 计算的积分值: 5.337512
```

代码:

```python
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
```



## Ex4 p24

不用做.



## Ex p36

题: 用上述算法, 估计整数子集1~n的大小, 并分析n对估计值的影响.

答: 

我们将每个n跑了100次, 求100次结果的均值和标准差. 由运行结果可知, n对估计值的影响不大.

运行结果: 

```
每个n跑100次
n mean std
10 10 8
100 98 91
1000 1259 1262
10000 13747 12670
100000 108523 123713
1000000 1331652 1540778
10000000 10899882 11692731
```

代码: 

```python
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

```



## Ex p54

题: 分析dlogRH的工作原理, 指出该算法相应的u和v.

答: 

1. 首先得到随机变量r.
   r ← uniform(0..p-2);

2. 下面两行为该算法相应的u, 将被解的实例a根据随机变量r变换到随机实例c. 

   b ← ModularExponent(g, r, p);
   c ← ba mod p;

3. 用确定算法解随机实例c, 得到解y.
   y ← $log_{g,p}c$;
4. 下面一行为该算法相应的v, 将此解y变换为原实例的解并返回.
   return (y-r) mod (p-1); 



## Ex p67

题: 写一Sherwood算法C, 与算法A, B, D比较, 给出实验结果.

答: 

A: 确定算法, 平均和最坏时间复杂度都是O(n).
D: 概率算法, 平均和最坏期望时间复杂度都是O(n).
B: 确定算法, 平均时间复杂度$O(\sqrt{n})$, 最坏时间复杂度O(n).
C: 概率算法, 平均期望时间复杂度$O(\sqrt{n})$​​, 最坏期望时间复杂度O(n).

算法C根据算法B进行改进, 将取前$\sqrt{n}$个数中找不大于x的最大整数y的下标改为随机取$\sqrt{n}$​个数中找不大于x的最大整数y的下标.

以下运行结果可知: 
在顺序情况下, 算法B退化为O(n)复杂度算法, 运行时间不佳,  而概率算法C(由B改进)的复杂度为$O(\sqrt{n})$​​​, 运行时间明显优于A, D, B算法.
在随机情况下, B的平均运行时间会略少于C, 因为少了C中为了均匀性而付出的开销. 
以上4种算法, 都能保证得到正确解.

运行结果: 

```
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
```

代码:

```python
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
```



## Ex p77

题: 证明: 当放置 (k+1)th 皇后时, 若有多个位置是开放的, 则算法QueensLV选中其中任一位置的概率相等.

答: 

以下用数学归纳法证明若有n个位置是开放的, 算法QueensLV选中其中任一位置的概率相等.

当n=1时, 即nb=1时, 结论显然成立.
假设n=a成立时, 即$p_1=p_2=...=p_a=\frac{1}{a}$. 第a+1次循环时, $j\leftarrow i$的概率是$\frac{1}{a+1}$, $j$取值不变的概率时$\frac{a}{a+1}$. 
所以$p_1=p_2=...=p_a=\frac{1}{a}\times \frac{a}{a+1} = \frac{1}{a+1}$.
所以$p_1=p_2=...=p_{a+1}=\frac{1}{a+1}$. 所以结论成立.

综上结论成立.



## Ex p83

题: 写一算法, 求n=12~20时最优的StepVegas值.

答: 

我分别测试了n=12-20, StepVegas=0-n时, 测试成功1000次所需要的时间, 取$最优的StepVegas=\arg\min_{StepVegas\in[0...n]}t(n, StepVegas, 1000)$, 得到以下表格.

```
n 		最优的StepVegas
12		5
13		6
14		7
15		7
16		8
17		9
18		10
19		10
20		11
```

运行结果:

```
12 5
13 6
14 7
15 7
16 8
17 9
18 10
19 10
20 11
```

代码:

```python
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

```



## 习题 p147

问: PrintPrimes{  //打印1万以内的素数
    print 2, 3;
    n ← 5;
    repeat
        if RepeatMillRab(n, $\lfloor{lgn}\rfloor$) then
        print n;
        n ←n+2;
    until  n=10000;
}

与确定性算法相比较, 并给出100~10000以内错误的比例.

答: 

我们实验了10000次, 计算了概率算法平均100~10000以内错误比率. 大约为$8.3*10^{-8}$左右.

运行结果:

```
8.305647840531561e-08
```

代码:

```python
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

```

