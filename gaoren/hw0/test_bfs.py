# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from utils import Environment, a_star, bfs, greed
from utils.env_example import env1

env = Environment()
env.init_env(env_matrix=env1)
path, cost = bfs(env.env_matrix)
#
print("cost: %.6f" % cost)
print("path: ")
for pos in path:
    print(pos)
#
env.path = path
while True:
    env.step()
