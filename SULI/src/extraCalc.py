
import numpy as np
from gym import spaces
from random import random

from pandas.compat import numpy
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from heapq import nlargest

rewards = np.array([88.72, 87.89, 89.04, 90.64, 89.73, 87.71, 88.55, 87.38, 82.53, 87.61])
rewards = np.array([204.15,
177.82,
186.86,
168.82,
201.77,
190.66,
186.13,
179.63,
183.05,
195.58])
print(np.average(rewards))


distro = np.array([[9700, 0, 0],
[9700, 0, 0],
[9700, 0, 0],
[9700, 0, 0],
[9700, 0, 0],
[9661, 39, 0],
[9700, 0, 0],
[9700, 0, 0],
[8897, 803, 0],
[9700, 0, 0]])


distro = np.array([[9277, 235, 118],
[9288, 224, 115],
[9417, 165, 79],
[9388, 184, 96],
[9581, 88, 30],
[9368, 190, 95],
[9581, 88, 28],
[9322, 205, 106],
[8992, 449, 164],
[9378, 180, 95]])


distro = np.array(
[[9460, 156, 59],
[9325, 201, 108],
[9411, 185, 85],
[9303, 223, 112],
[8986, 373, 194],
[9227, 261, 133],
[8805, 567, 199],
[9299, 222, 112],
[9264, 238, 122],
[9398, 187, 84]]
)

distro = np.array([
[9700, 0, 0],
[9700, 0, 0],
[9700, 0, 0],
[9195, 503, 2],
[9700, 0, 0],
[9700, 0, 0],
[9700, 0, 0],
[9700, 0, 0],
[9700, 0, 0],
[9639, 57, 4]
])




distroStds = np.array([])
for mini in distro:
    distroStds = np.append(distroStds, np.std(mini))

print(distroStds)
print(np.average(distroStds))