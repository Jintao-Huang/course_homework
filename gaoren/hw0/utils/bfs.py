# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from typing import Tuple
import numpy as np
from . import search_alg
from functools import partial


def f(cur_pos: Tuple, end_pos: Tuple, pre_cost: float) -> float:
    return pre_cost


bfs = partial(search_alg, f=f)
