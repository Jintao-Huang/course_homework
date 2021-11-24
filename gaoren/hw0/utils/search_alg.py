# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from typing import Callable, List, Tuple, Optional, Set
from .environment import PointType
from .priority_queue import PriorityQueue
import numpy as np
from numpy import ndarray


def search_alg(env_matrix: ndarray, f: Callable[[Tuple, Tuple, float], float]) -> Optional[Tuple]:
    """使用欧几里得距离

    :param env_matrix: shape[H, W]
    :param f: 启发式函数. [cur_pos, end_pos, cost] -> f_cost
        cost: cost from start_pos to cur_pos
        f_cost: 启发式cost
    :return: Tuple[path, cost] or None. 如果没找到 返回None.
        path: Tuple[left, top]
    """
    #
    path = None  # type: Optional[List[Tuple[int, int]]]
    height, width = env_matrix.shape
    # 直线与对角线
    direction = np.array([[0, 1], [1, 0], [0, -1], [-1, 0],
                          [1, 1], [-1, 1], [-1, -1], [1, -1]])
    # Top Left!
    start_pos = cur_pos = tuple(np.argwhere(env_matrix == PointType.START)[0].tolist())  # type: Tuple[int, int]
    end_pos = tuple(np.argwhere(env_matrix == PointType.END)[0].tolist())  # type: Tuple[int, int]
    wall_pos = np.argwhere(env_matrix == PointType.WALL).tolist()
    wall_pos = set(tuple(item) for item in wall_pos)  # type: Set[Tuple[int, int]]
    #
    cost_matrix = np.full_like(env_matrix, np.inf, dtype=np.float64)
    parent_matrix = np.full_like(env_matrix, -1, dtype=np.int32)  # parent: i * w + j
    #
    cost = cost_matrix[cur_pos] = 0
    f_cost = f(cur_pos, end_pos, cost)
    queue = PriorityQueue([(f_cost, cost, cur_pos)])  # Tuple[f_cost: float, cost: float, pos: Tuple]
    while len(queue) > 0:
        _, pre_cost, cur_pos = queue.pop()
        if cur_pos not in (start_pos, end_pos):
            env_matrix[cur_pos] = PointType.VISITED
            #
        if cur_pos == end_pos:
            path = find_path(end_pos, parent_matrix)
            path = [(item[1], item[0]) for item in path]  # top, left -> left, top
            cost = pre_cost
            break
        # add
        for i in range(direction.shape[0]):
            d = direction[i]
            pos = cur_pos[0] + d[0], cur_pos[1] + d[1]
            #
            if not 0 <= pos[0] < height or not 0 <= pos[1] < width:
                continue
            if pos in wall_pos:
                continue
            #
            cost = pre_cost + np.linalg.norm(d)  # 加上d的距离. e.g. 1 or sqrt(2)
            if cost_matrix[pos] <= cost:
                continue
            #
            cost_matrix[pos] = cost
            parent_matrix[pos] = cur_pos[0] * width + cur_pos[1]
            f_cost = f(pos, end_pos, cost)
            queue.add((f_cost, cost, pos))

    return (path, cost) if path else None


def find_path(end_pos: Tuple[int, int], parent_matrix: ndarray) -> List[Tuple[int, int]]:
    """

    :param end_pos:
    :param parent_matrix: shape[H, W]. no parent: = -1; else: = i * width + j
    :return:
    """
    width = parent_matrix.shape[1]
    cur_pos = end_pos
    pos_path = [cur_pos]
    #
    while parent_matrix[cur_pos] != -1:
        cur_pos = parent_matrix[cur_pos]
        cur_pos = cur_pos // width, cur_pos % width
        pos_path.append(cur_pos)
    #
    pos_path.reverse()  # from start_pos -> end_pos
    return pos_path
