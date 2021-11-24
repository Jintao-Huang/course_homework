# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import heapq


class PriorityQueue:
    def __init__(self, initial_list=None):
        queue = initial_list if initial_list else []
        heapq.heapify(queue)
        #
        self._queue = queue

    def add(self, x):
        queue = self._queue
        heapq.heappush(queue, x)

    def pop(self):
        queue = self._queue
        return heapq.heappop(queue)

    def __len__(self):
        return len(self._queue)
