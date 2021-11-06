# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from utils import Environment
from utils.env_example import env1

env = Environment()
env.init_env(env_matrix=env1)
#
while True:
    env.step()
