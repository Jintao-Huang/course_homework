# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from typing import Tuple
import numpy as np
from . import search_alg
from functools import partial


def f(cur_pos: Tuple, end_pos: Tuple, pre_cost: float) -> float:
    cur_pos = np.array(cur_pos)
    end_pos = np.array(end_pos)
    return np.linalg.norm(cur_pos - end_pos)


greed = partial(search_alg, f=f)
