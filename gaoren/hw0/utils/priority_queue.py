# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import heapq


class PriorityQueue:
    def __init__(self, initial_list=None, key=None):
        if key and initial_list:
            initial_list = [(key(item), item) for item in initial_list]
        #
        queue = initial_list if initial_list else []
        heapq.heapify(queue)
        #
        self._queue = queue
        self.key = key

    def add(self, x):
        queue = self._queue
        key = self.key
        #
        if key:
            x = key(x), x
        heapq.heappush(queue, x)

    def pop(self):
        queue = self._queue
        key = self.key
        #
        res = heapq.heappop(queue)
        return res[1] if key else res

    def __len__(self):
        return len(self._queue)
