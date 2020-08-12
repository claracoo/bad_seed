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
import scipy.stats as st

#finds the standard deviation but takes into account that the 0s are null
def stdDeviaiton(array):
    cleanedUp = np.array([])
    for elem in array:
        if elem != 0:
            cleanedUp = np.append(cleanedUp, elem)
    return np.std(cleanedUp)

#the null value is the sampleCount, meaning we want to put the new value at the first one that is not the auto measured ones
#not used in end product, simulation dissolved
def findLastNonLen(action_arr, sampleCount):
    for i in range(len(action_arr)):
        if action_arr[i] == sampleCount and i > 2:
            return i - 1
    return len(action_arr) - 1

#a way to calculate what makes a standard deviaiton unusually high
#parameterizes what is "unusual" in terms of percent
#percent must be under 1
# returns the upper bound of unusual
def confidenceIntervalMax(array, percent, n):
    average = np.average(array)
    sd = np.std(array)
    zValue = st.norm.ppf(percent)
    conHigh = average + zValue * (sd / np.sqrt(n))
    return conHigh


# fakeArr = np.array([1, 1, 1, 1, 1, 1, 1, 2, 1, 5, 5, 5])
# print(np.array([fakeArr[findLastNonLen(fakeArr, 5)], fakeArr[findLastNonLen(fakeArr, 5) - 1]]))
# print(np.array([2, 1]).shape)


class CustomEnvironment(Environment):
    #while the standard number of bad seeds is 3, it can be channged to anything between 0 and the number of samples
    badSeedCount = 3
    # LEFT = 0
    # RIGHT = 1
    #sum of all rewards during testing
    sum = 0
    #represents the index of where the game actually starts, since the first 3 measurements are automatic
    #redefined in env
    extraCounter = 3
    firstCount = 0
    secondCount = 0
    thirdCount = 0
    testingEps = 100
    trainingEps = 5
    trials = 100
    highest_arr = np.array([])
    sndHighest_arr = np.array([])
    trdHighest_arr = np.array([])
    repeatCounter_arr = np.array([])
    skippedRepeat_arr = np.array([])
    # allActions = np.array([])
    allEpisodes = np.array([])
    expected = np.array([])
    actual = np.array([])
    rewards = np.array([])
    violinGrid = []
    badseedsFinal= []

    #creates x axis for most graphs, for episode count
    for i in range(testingEps + trainingEps):
        allEpisodes = np.append(allEpisodes, i)

    #when env first initialized
    def __init__(self):
            super().__init__()
            #index of true start of gamified measurements
            self.startingPoint = 3
            CustomEnvironment.extraCounter = self.startingPoint
            # Initialize the agent at the left of the grid, keeps track of where we are
            self.agent_pos = self.startingPoint
            self._max_episode_timesteps = 500
            self.TRIALS = CustomEnvironment.trials
            self.SAMPLES = 10
            self.confidence = 0.95
            #min nnumber of measurements necessary to make a judgement on if the std dev is unusually high
            #true min measurements is this number plus 3, since 3 auto measurements taken
            self.minMeasurements = 5
            self.GRID = []
            gridCopy = []
            #how many times (not including auto measurements) each sample has been chosen per round
            #keys are samples, times chosen are values
            self.minSampling = {}
            # keys are samples, associated std devs are values
            self.stdDev = {}
            # self.stdDevSim = {}
            self.sum = 0
            self.reward = 0
            self.highest = 0
            self.sndHighest = 0
            self.trdHighest = 0
            self.state = [0, 0, 0]
            self.repeatCounter = 0
            self.skippedRepeat = 0
            #gives the agent extra points for choosing a sample that they haven't chosen in a while
            self.bank = {}
            #null values for which samples are bad seeds
            self.badseeds = []
            for i in range(CustomEnvironment.badSeedCount):
                self.badseeds.append(self.SAMPLES)

            # self.simulation = []
            #simulation use only
            self.worstSeed = randint(0, self.SAMPLES - 1)


            for i in range(self.SAMPLES):
                col = []
                for j in range(self.TRIALS):
                    if j < self.startingPoint:
                        col.append(randint(0, 1000000))
                    else:
                        col.append(0)
                self.GRID.append(col)
                gridCopy.append(col)
                self.bank[i] = 0

            # second length will be the actions, but actions is not in scope in init
            # the width should always be the number of trials
            # simulation use only
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
        #shape is [highest std dev index, last chosen index, penultimate chosen index, badseed1, badseed2, badseed3]
        stateArr = [0]
        for i in range(CustomEnvironment.badSeedCount - 1):
            stateArr.append(0)
        for i in range(CustomEnvironment.badSeedCount):
            stateArr.append(0)
        return dict(type='float', shape=(len(stateArr),))

    def actions(self):
        return dict(type='int', num_values=self.SAMPLES)

    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()


    def reset(self):
        # self.extraCounter = self.startingPoint
        CustomEnvironment.extraCounter = self.startingPoint
        self.badseeds = []
        for i in range(CustomEnvironment.badSeedCount):
            self.badseeds.append(self.SAMPLES)
        self.reward = 0
        self.highest = 0
        self.sndHighest = 0
        self.trdHighest = 0
        self.agent_pos = self.startingPoint
        self.worstSeed = randint(0, self.SAMPLES - 1)
        self.state = [0, 0, 0]
        self.repeatCounter = 0
        self.skippedRepeat = 0
        CustomEnvironment.actual = np.array([])
        CustomEnvironment.expected = np.array([])
        for i in range(self.SAMPLES):
            for j in range(self.TRIALS):
                if j < self.startingPoint:
                    #automatic measuring
                    self.GRID[i][j] = randint(0, 1000000)
                else:
                    self.GRID[i][j] = 0
            self.bank[i] = 0
        #not used inn final version (shape simplified)
        for i in range(self.TRIALS):
            self.shape[0][i] = self.TRIALS
            self.shape[1][i] = self.SAMPLES
            self.shape[2][i] = 0
            for j in range(self.SAMPLES):
                self.shape[j + 3] = self.GRID[j]

        for i in range(self.SAMPLES):
            self.minSampling[i] = 0
            self.bank[i] = 0
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        stateArr = [0]
        for i in range(CustomEnvironment.badSeedCount - 1):
            stateArr.append(0)
        for i in range(CustomEnvironment.badSeedCount):
            stateArr.append(0)
        return np.array(stateArr).astype(np.float32)

    def execute(self, actions):
        # self.extraCounter += 1
        CustomEnvironment.extraCounter += 1
        maxStdDev = []
        reward = 0
        #variable ratio schedule of 2/3
        randomizer = randint(0, 2)

        #making sure this is a valid action
        if (actions >= 0 and actions < self.SAMPLES):
            # CustomEnvironment.allActions = np.append(CustomEnvironment.allActions, actions)
            # for i in range(self.SAMPLES):
            #     self.stdDev[i] = stdDeviaiton(array=self.GRID[i])
            # maxStdDev = nlargest(3, self.stdDev, key=self.stdDev.get)
            if len(maxStdDev) == 0:
                for i in range(self.SAMPLES):
                    self.stdDev[i] = stdDeviaiton(array=self.GRID[i])
                maxStdDev = nlargest(CustomEnvironment.badSeedCount, self.stdDev, key=self.stdDev.get)
            # finding the threshold for being an unusually high standard dev
            conHigh = confidenceIntervalMax(list(self.stdDev.values()), self.confidence, self.SAMPLES)
            #once we are confident in the fact that is a bad seed, we do not need many adidtional measurements
            if actions in self.badseeds:
                self.reward -= 2
            # if it is unusually high, and we have the minimum amount of measuremnts we to define this:
            if self.stdDev[actions] > conHigh and self.minSampling[actions] > self.minMeasurements:
                changed = False
                for i in range(len(self.badseeds)):
                    if changed == False and self.badseeds[i] == self.SAMPLES and actions not in self.badseeds:
                        self.badseeds[i] = actions
                        changed = True
            print(actions, maxStdDev)
            # print(actions, maxStdDev)
            # print(actions, "vs.", self.shape[1][self.agent_pos - 1])

            #penalizing for repeats
            for i in range(CustomEnvironment.badSeedCount - 1):
                if actions == self.shape[1][self.agent_pos - (i + 1)]:
                    self.reward -= 1
                    self.repeatCounter += 1
                else:
                    self.reward += 1
            # if actions == self.shape[1][self.agent_pos - 1]:
            #     self.reward -= 1
            #     self.repeatCounter += 1
            # elif actions == self.shape[1][self.agent_pos - 2]:
            #     self.reward -= 1
            #     self.skippedRepeat += 1
            # else:
            #     self.reward += 1
            CustomEnvironment.expected = np.append(CustomEnvironment.expected, maxStdDev[0])
            CustomEnvironment.actual = np.append(CustomEnvironment.actual, actions)
            #rewarding for getting the highest std dev
            if actions == maxStdDev[0]:
                self.reward += CustomEnvironment.badSeedCount
                self.highest += 1
            if actions == maxStdDev[1]:
                # self.reward += 2
                self.sndHighest += 1
            if actions == maxStdDev[2]:
                # self.reward += 1
                self.trdHighest += 1
            #rewarding if chosen onne that has not been chosen in a while
            if actions == nlargest(1, self.bank, key=self.bank.get)[0]:
                self.reward += self.bank[actions]
            #buildinng up a greater reward, the longer it is not chosen
            for i in range(self.SAMPLES):
                if actions != i:
                    self.bank[i] += 1
                else:
                    self.bank[i] = 0

            self.shape[0][self.agent_pos] = self.agent_pos
            self.shape[1][self.agent_pos] = actions
            self.shape[2][self.agent_pos] = self.reward

            #how the measurements get made, not necessary to differentiate now that simulation is over
            if actions == self.worstSeed:
                self.GRID[actions][self.agent_pos] = randint(0, 1000000)
            else:
                self.GRID[actions][self.agent_pos] = randint(0, 1000000)

            #keeping track of how many times we have chosen it
            self.minSampling[actions] += 1
            #moving where we are in the game
            self.agent_pos += 1
            # print(list(self.stdDev.values()))

            #we want to remeasure the std devs at the end to feed to the next round
            for i in range(self.SAMPLES):
                self.stdDev[i] = stdDeviaiton(array=self.GRID[i])
                #if it is already a bad seed we do not want it to contribute the standard devs because it will always be one of the highest
                if i in self.badseeds:
                    self.stdDev[i] = 0
            maxStdDev = nlargest(CustomEnvironment.badSeedCount, self.stdDev, key=self.stdDev.get)
            if randomizer == 0:
                maxStdDev[0] == randint(0, self.SAMPLES - 1)
            self.state = [maxStdDev[0]]
            for i in range(CustomEnvironment.badSeedCount - 2):
                self.state = np.append(self.state, self.shape[1][self.agent_pos - (i + 2)])
            self.state = np.append(self.state, actions)
            for seed in self.badseeds:
                self.state = np.append(self.state, seed)
            # self.state = np.append(self.state, nlargest(1, self.bank, key=self.bank.get)[0])
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(actions))
            # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.TRIALS)
        # Are we at the right of the grid?
        done = bool(self.agent_pos == self.TRIALS)

        reward = self.reward
        if done:
            #all for graphing purposes
            CustomEnvironment.repeatCounter_arr = np.append(CustomEnvironment.repeatCounter_arr, self.repeatCounter)
            CustomEnvironment.skippedRepeat_arr = np.append(CustomEnvironment.skippedRepeat_arr, self.skippedRepeat)
            CustomEnvironment.highest_arr = np.append(CustomEnvironment.highest_arr, self.highest)
            CustomEnvironment.sndHighest_arr = np.append(CustomEnvironment.sndHighest_arr, self.sndHighest)
            CustomEnvironment.trdHighest_arr = np.append(CustomEnvironment.trdHighest_arr, self.trdHighest)
            CustomEnvironment.badseedsFinal = self.badseeds
            self.sum += 1
            CustomEnvironment.rewards = np.append(CustomEnvironment.rewards, reward)
            CustomEnvironment.violinGrid = self.GRID
            if self.sum > CustomEnvironment.trainingEps:
                CustomEnvironment.sum += reward

        returning = np.array(self.state).astype(np.float32), reward, done

        return returning

#runnning the environment
def runEnv():
    environment = Environment.create(
        environment=CustomEnvironment, max_episode_timesteps=500
    )
    agent = Agent.create(
        agent='a2c', environment=environment, batch_size=10, learning_rate=1e-3,

        exploration=0.01,  # tried without this at first
        variable_noise=0.05,
        # variable_noise=0.01 bad?
        l2_regularization=0.1,
        entropy_regularization=0.2,

        summarizer = dict(
            directory='data/summaries',
            # list of labels, or 'all'
            labels=['graph', 'entropy', 'kl-divergence', 'losses', 'rewards'],
            frequency=100,  # store values every 100 timesteps
        )
    )

    # Train for 200 episodes
    for _ in range(CustomEnvironment.trainingEps):
        print("Episode:  ", _)
        states = environment.reset()
        terminal = False
        while CustomEnvironment.extraCounter != CustomEnvironment.trials:
            actions = agent.act(states=states)
            # print(actions)
            # print(states)
            states, reward, terminal = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
        print("bad seeds: ", CustomEnvironment.badseedsFinal)

    # Evaluate for 100 episodes
    sum_rewards = 0.0
    for _ in range(CustomEnvironment.testingEps):
        print("Episode:  ", _ + CustomEnvironment.trainingEps)
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        while CustomEnvironment.extraCounter != CustomEnvironment.trials:
            actions, internals = agent.act(states=states, internals=internals, independent=True)
            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward
        print("bad seeds: ", CustomEnvironment.badseedsFinal)
    # print('Mean episode reward:', sum_rewards / 100)
    # print(CustomEnvironment.firstCount, ",", CustomEnvironment.secondCount, ",", CustomEnvironment.thirdCount)
    print(CustomEnvironment.sum)


    # Close agent and environment
    agent.close()
    environment.close()


def lossPlot():
    # Data for plotting
    x = CustomEnvironment.rewards
    s = CustomEnvironment.allEpisodes


    fig, ax = plt.subplots()
    ax.plot(s, x)


    ax.set(xlabel='Episode', ylabel='Rewards',
           title='Rewards per Episode')
    ax.grid()

    fig.savefig("rewards.png")
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

def badSeedsPlot():
    badSeeds = CustomEnvironment.highest_arr
    total1 = 0
    total2 = 0
    total3 = 0
    for i in range(len(CustomEnvironment.highest_arr)):
        badSeeds[i] += (CustomEnvironment.sndHighest_arr[i] + CustomEnvironment.trdHighest_arr[i])
        total1 += CustomEnvironment.highest_arr[i]
        total2 += CustomEnvironment.sndHighest_arr[i]
        total3 += CustomEnvironment.trdHighest_arr[i]
    z = badSeeds
    s = CustomEnvironment.allEpisodes


    fig, ax = plt.subplots()
    ax.plot(s, z)


    ax.set(xlabel='Episode', ylabel='Number of Third Highest Standard Deviations Chosen',
           title='Standard Deviation Choices per Episode')
    ax.grid()

    fig.savefig("total.png")
    plt.show()

def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample number')

def violinPlots():
    data = CustomEnvironment.violinGrid

    fig, ax1 = plt.subplots()

    ax1.set_title('Sample Distributions')
    ax1.set_ylabel('Observed values')
    ax1.violinplot(data)

    # set style for the axes
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for ax in [ax1]:
        set_axis_style(ax, labels)

    plt.subplots_adjust(bottom=0.15, wspace=0.55)
    fig.savefig("violin.png")
    plt.show()


def makePlot():
    # Data for plotting
    x = CustomEnvironment.skippedRepeat_arr
    t = CustomEnvironment.repeatCounter_arr
    s = CustomEnvironment.allEpisodes


    fig, ax = plt.subplots()
    ax.plot(s, t)
    ax.plot(s, x)


    ax.set(xlabel='Episode', ylabel='Number of Repeated Actions',
           title='Repeats per Episode')
    ax.grid()

    fig.savefig("test.png")
    plt.show()

if __name__ == "__main__":
    runEnv()
    makePlot()
    makePlot1()
    makePlot2()
    makePlot3()
    badSeedsPlot()
    lossPlot()
    violinPlots()
    # logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    # with tf.Session() as sess:
    #     print(sess.run(runEnv()))


# lossPlot()
