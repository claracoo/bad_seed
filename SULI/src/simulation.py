import numpy as np
import gym
from gym import spaces

import tensorflow as tf
from random import random, randint
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from heapq import nlargest
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from packaging import version
from tensorflow import keras
from tensorflow.python.keras.utils import losses_utils


def stdDeviaiton(array):
    cleanedUp = np.array([])
    for elem in array:
        if elem != 0:
            cleanedUp = np.append(cleanedUp, elem)
    return np.std(cleanedUp)


def findLastNonLen(action_arr, sampleCount):
    for i in range(len(action_arr)):
        if action_arr[i] == sampleCount and i > 2:
            return i - 1
    return len(action_arr) - 1


# fakeArr = np.array([1, 1, 1, 1, 1, 1, 1, 2, 1, 5, 5, 5])
# print(np.array([fakeArr[findLastNonLen(fakeArr, 5)], fakeArr[findLastNonLen(fakeArr, 5) - 1]]))
# print(np.array([2, 1]).shape)


class CustomEnvironment(Environment):
    # LEFT = 0
    # RIGHT = 1
    sum = 0
    extraCounter = 3
    firstCount = 0
    secondCount = 0
    thirdCount = 0
    highest_arr = np.array([])
    sndHighest_arr = np.array([])
    trdHighest_arr = np.array([])
    # allActions = np.array([])
    allEpisodes = np.array([])
    expected = np.array([])
    actual = np.array([])
    losses = np.array([])

    for i in range(10000):
        allEpisodes = np.append(allEpisodes, i)


    # def __init__(self):
    def __init__(self):
            super().__init__()

            self.startingPoint = 3
            CustomEnvironment.extraCounter = self.startingPoint
            # Initialize the agent at the right of the grid
            self.agent_pos = self.startingPoint
            self._max_episode_timesteps = 500
            self.TRIALS = 100
            self.SAMPLES = 10
            self.GRID = []
            gridCopy = []
            self.minSampling = {}
            self.stdDev = {}
            # self.stdDevSim = {}
            self.sum = 0
            self.reward = 0
            self.highest = 0
            self.sndHighest = 0
            self.trdHighest = 0
            # self.simulation = []
            self.worstSeed = randint(0, self.SAMPLES - 1)


            for i in range(self.SAMPLES):
                col = []
                for j in range(self.TRIALS):
                    if j < self.startingPoint:
                        col.append(1)
                    else:
                        col.append(0)
                self.GRID.append(col)
                gridCopy.append(col)

            # second length will be the actions, but actions is not in scope in init
            # the width should always be the number of trials
            self.shapeHeight = len(self.GRID) + len([self.agent_pos]) + len([0]) + len([0])

            for i in range(self.SAMPLES):
                self.minSampling[i] = 0

            for i in range(self.SAMPLES):
                self.stdDev[i] = self.startingPoint
            # for i in range(self.SAMPLES):
            #     self.stdDevSim[i] = 0
            # for i in range(self.SAMPLES):
                # print(i)
                # print(self.simulation[i])
                # print(stdDeviaiton(array=[0, 0, 0, 0, 0, 0, 1, 0, 8, 0]))
                # self.stdDevSim[i] = stdDeviaiton(array=self.simulation[i])
                # print(self.stdDevSim)
            # print(nlargest(3, self.stdDevSim, key=self.stdDevSim.get))


            # Define action and observation space
            # They must be gym.spaces objects
            # Example when using discrete actions, we have two: left and right
            n_actions = self.SAMPLES
            self.action_space = spaces.Discrete(n_actions)
            # The observation will be the coordinate of the agent
            # this can be described both by Discrete and Box space
            # self.observation_space = spaces.Box(low=0, high=self.SAMPLES,
            #                                     shape=(self.SAMPLES, self.TRIALS), dtype=np.float32)

            self.shape = gridCopy
            # to keep track of the agent position, until it is filled in the agent position will be something it would never be, self.TRIALS
            #same with the actions, never will be samples, so good null value
            agent_pos_arr = []
            actions_arr = []
            rewards_arr = []
            for i in range(self.TRIALS):
                agent_pos_arr.append(self.TRIALS)
                actions_arr.append(self.SAMPLES)
                rewards_arr.append((0))

            self.shape.insert(0, rewards_arr)
            self.shape.insert(0, actions_arr)
            self.shape.insert(0, agent_pos_arr)


    def states(self):
        return dict(type='float', shape=(len([0]),))

    def actions(self):
        return dict(type='int', num_values=self.SAMPLES)

    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()


    def reset(self):
        # self.extraCounter = self.startingPoint
        CustomEnvironment.extraCounter = self.startingPoint
        self.reward = 0
        self.highest = 0
        self.sndHighest = 0
        self.trdHighest = 0
        self.agent_pos = self.startingPoint
        self.worstSeed = randint(0, self.SAMPLES - 1)
        CustomEnvironment.actual = np.array([])
        CustomEnvironment.expected = np.array([])
        for i in range(self.SAMPLES):
            for j in range(self.TRIALS):
                if j < self.startingPoint:
                    self.GRID[i][j] = 1
                else:
                    self.GRID[i][j] = 0

        for i in range(self.TRIALS):
            self.shape[0][i] = self.TRIALS
            self.shape[1][i] = self.SAMPLES
            self.shape[2][i] = 0
            for j in range(self.SAMPLES):
                self.shape[j + 3] = self.GRID[j]

        for i in range(self.SAMPLES):
            self.minSampling[i] = 0
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([0]).astype(np.float32)

    def execute(self, actions):
        # self.extraCounter += 1
        CustomEnvironment.extraCounter += 1
        maxStdDev = []
        reward = 0
        if (actions >= 0 and actions < self.SAMPLES):

            # CustomEnvironment.allActions = np.append(CustomEnvironment.allActions, actions)
            for i in range(self.SAMPLES):
                self.stdDev[i] = stdDeviaiton(array=self.GRID[i])
            maxStdDev = nlargest(3, self.stdDev, key=self.stdDev.get)
            print(actions, maxStdDev)
            print(actions, "vs.", self.shape[1][self.agent_pos - 1])
            # if actions == self.shape[1][self.agent_pos - 1]:
            #     self.reward -= 1
            #     self.repeatCounter += 1
            # elif actions == self.shape[1][self.agent_pos - 2]:
            #     self.reward -= 1
            #     self.skippedRepeat += 1
            CustomEnvironment.expected = np.append(CustomEnvironment.expected, maxStdDev[0])
            CustomEnvironment.actual = np.append(CustomEnvironment.actual, actions)
            if actions == maxStdDev[0]:
                self.reward += 1
                self.highest += 1
            # elif actions == maxStdDev[1]:
            #     self.reward += 2
            #     self.sndHighest += 1
            # elif actions == maxStdDev[2]:
            #     self.reward += 1
            #     self.trdHighest += 1
            # else:
            #     self.reward += 0
                # print(maxStdDev, actions)
            # if self.agent_pos <= self.TRIALS:
            self.shape[0][self.agent_pos] = self.agent_pos
            self.shape[1][self.agent_pos] = actions
            self.shape[2][self.agent_pos] = self.reward

            if actions == self.worstSeed:
                self.GRID[actions][self.agent_pos] = randint(0, 1000000)
            else:
                self.GRID[actions][self.agent_pos] = randint(0, 10000)
            self.minSampling[actions] += 1
            self.agent_pos += 1
            print(list(self.stdDev.values()))
            maxStdDev = nlargest(3, self.stdDev, key=self.stdDev.get)
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(actions))
            # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.TRIALS)

        # if self.sum > 200:
        #     if actions == maxStdDev[0]:
        #         CustomEnvironment.firstCount += 1
        #     if actions == maxStdDev[1]:
        #         CustomEnvironment.secondCount += 1
        #     if actions == maxStdDev[2]:
        #         CustomEnvironment.thirdCount += 1
        # Are we at the right of the grid?
        done = bool(self.agent_pos == self.TRIALS)

        reward = self.reward
        print(reward)
        if done:
            CustomEnvironment.highest_arr = np.append(CustomEnvironment.highest_arr, self.highest)
            CustomEnvironment.sndHighest_arr = np.append(CustomEnvironment.sndHighest_arr, self.sndHighest)
            CustomEnvironment.trdHighest_arr = np.append(CustomEnvironment.trdHighest_arr, self.trdHighest)
            # reward += 1
            self.sum += 1
            if self.sum > 6000:
                # mostChosen = nlargest(3, self.minSampling, key=self.minSampling.get)
                # CustomEnvironment.firstCount += self.minSampling[mostChosen[0]]
                # CustomEnvironment.secondCount += self.minSampling[mostChosen[1]]
                # CustomEnvironment.thirdCount += self.minSampling[mostChosen[2]]
                CustomEnvironment.sum += reward
            # bce = tf.keras.losses.BinaryCrossentropy()
            # print(bce(CustomEnvironment.actual, CustomEnvironment.expected).numpy())
            # CustomEnvironment.losses = np.append(CustomEnvironment.losses, bce(CustomEnvironment.actual, CustomEnvironment.expected))

            # y_true = [[0, 1], [0, 0]]
            # y_pred = [[0.6, 0.4], [0.4, 0.6]]
            loss = tf.keras.losses.binary_crossentropy(CustomEnvironment.actual, CustomEnvironment.expected)
            # assert loss.shape == (2,)
            print("LOSSSSSS", type(loss))
            # CustomEnvironment.losses = np.append(CustomEnvironment.losses, loss)

            print(self.minSampling)
        print("check", self.shape[1])

        returning = np.array([maxStdDev[0]]).astype(np.float32), reward, done

        return returning

def runEnv():
    environment = Environment.create(
        environment=CustomEnvironment, max_episode_timesteps=500
    )
    agent = Agent.create(agent='a2c', environment=environment, batch_size=10, learning_rate=1e-3)

    # Train for 200 episodes
    for _ in range(6000):
        states = environment.reset()
        terminal = False
        while CustomEnvironment.extraCounter != 100:
            actions = agent.act(states=states)
            # print(actions)
            # print(states)
            states, reward, terminal = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)

    # Evaluate for 100 episodes
    sum_rewards = 0.0
    for _ in range(4000):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        while CustomEnvironment.extraCounter != 100:
            actions, internals = agent.act(states=states, internals=internals, independent=True)
            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward

    # print('Mean episode reward:', sum_rewards / 100)
    # print(CustomEnvironment.firstCount, ",", CustomEnvironment.secondCount, ",", CustomEnvironment.thirdCount)
    print(CustomEnvironment.sum)


    # Close agent and environment
    agent.close()
    environment.close()


def lossPlot():
    # Data for plotting
    x = CustomEnvironment.losses
    s = CustomEnvironment.allEpisodes


    fig, ax = plt.subplots()
    ax.plot(s, x)


    ax.set(xlabel='Episode', ylabel='Loss',
           title='Loss per Episode')
    ax.grid()

    fig.savefig("loss.png")
    plt.show()


def makePlot1():
    # Data for plotting
    x = CustomEnvironment.highest_arr
    s = CustomEnvironment.allEpisodes


    fig, ax = plt.subplots()
    ax.plot(s, x)


    ax.set(xlabel='Episode', ylabel='Number of Highest Standard Deviations Chosen',
           title='Standard Deviation Choices per Episode')
    ax.grid()

    fig.savefig("test1.png")
    plt.show()

def makePlot2():
    # Data for plotting
    y = CustomEnvironment.sndHighest_arr
    s = CustomEnvironment.allEpisodes


    fig, ax = plt.subplots()
    ax.plot(s, y)


    ax.set(xlabel='Episode', ylabel='Number of Second Highest Standard Deviations Chosen',
           title='Standard Deviation Choices per Episode')
    ax.grid()

    fig.savefig("test2.png")
    plt.show()

def makePlot3():
    # Data for plotting
    z = CustomEnvironment.trdHighest_arr
    s = CustomEnvironment.allEpisodes


    fig, ax = plt.subplots()
    ax.plot(s, z)


    ax.set(xlabel='Episode', ylabel='Number of Third Highest Standard Deviations Chosen',
           title='Standard Deviation Choices per Episode')
    ax.grid()

    fig.savefig("test3.png")
    plt.show()

if __name__ == "__main__":
    runEnv()
    makePlot1()
    makePlot2()
    makePlot3()
    lossPlot()
    # logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    # with tf.Session() as sess:
    #     print(sess.run(runEnv()))


# lossPlot()