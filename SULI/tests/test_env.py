import pprint

from SULI.src.tensorForceEnv import CustomEnvironment


def test_done():
    env = CustomEnvironment()


    actions = env.actions()
    print(f"actions: {actions}")

    assert env.extraCounter == 3
    print(f"extra Count: {env.extraCounter}")

    states, reward, done = env.execute(actions=0)
    print(f"states: {states}")
    print(f"reward: {reward}")
    print(f"done: {done}")

    assert done is False

    states, reward, done = env.execute(actions=0)
    assert done is False

    states, reward, done = env.execute(actions=0)
    assert done is False

    states, reward, done = env.execute(actions=0)
    assert done is False

    states, reward, done = env.execute(actions=0)
    assert done is False

    states, reward, done = env.execute(actions=0)
    assert done is False

    states, reward, done = env.execute(actions=0)
    assert done is False

    # states, reward, done = env.execute(actions=0)
    print(f"states: {states}")
    print(f"reward: {reward}")
    print(f"done: {done}")

    # assert done is False


def test_reset():
    env = CustomEnvironment()
    assert env.extraCounter == 3
    assert env.agent_pos == 3
    assert len(env.GRID) == env.SAMPLES
    assert len(env.GRID[0]) == env.TRIALS


    env.execute(actions=0)
    env.reset()

    assert env.extraCounter == 3
    assert env.agent_pos == 3
    assert len(env.GRID) == env.SAMPLES
    assert len(env.GRID[0]) == env.TRIALS

    env.execute(actions=0)
    env.reset()

    assert env.extraCounter == 3
    assert env.agent_pos == 3
    assert len(env.GRID) == env.SAMPLES
    assert len(env.GRID[0]) == env.TRIALS


def test_seven_steps():
    env = CustomEnvironment()

    state_reward_done = []
    for step in range(97):
        state_reward_done.append(env.execute(actions=0))

    pprint.pprint(state_reward_done)

    assert env.extraCounter == 100
    assert state_reward_done[96][2] is True


def test_stepthru_reset():
    env = CustomEnvironment()

    assert env.agent_pos == env.startingPoint
    assert env.extraCounter == env.startingPoint

    state_reward_done = []
    for step in range(97):
        state_reward_done.append(env.execute(actions=0))

    env.reset()

    for step in range(97):
        state_reward_done.append(env.execute(actions=0))

    pprint.pprint(state_reward_done)

    assert env.extraCounter == 100
    assert state_reward_done[96][2] is True
    assert state_reward_done[-1][2] is True

def test_shape_setup():
    env = CustomEnvironment()

    assert len(env.shape[0]) == env.TRIALS
    assert len(env.shape) == env.shapeHeight

    for i in range(env.SAMPLES):
        for j in range(env.TRIALS):
            assert(env.GRID[i][j] == env.shape[i + 3][j])

    for i in range(env.TRIALS):
        assert env.shape[0][i] == env.TRIALS
        assert env.shape[1][i] == env.SAMPLES

    env.reset()

    assert len(env.shape[0]) == env.TRIALS
    assert len(env.shape) == env.shapeHeight

    for i in range(env.SAMPLES):
        for j in range(env.TRIALS):
            assert (env.GRID[i][j] == env.shape[i + 3][j])

    for i in range(env.TRIALS):
        assert env.shape[0][i] == env.TRIALS
        assert env.shape[1][i] == env.SAMPLES

def test_shape_execute_once():
    env = CustomEnvironment()

    env.execute(actions=0)

    assert len(env.shape[0]) == env.TRIALS
    assert len(env.shape) == env.shapeHeight

    assert env.shape[0][env.startingPoint] == env.startingPoint
    assert env.shape[1][env.startingPoint] >= 0
    assert env.shape[1][env.startingPoint] < env.SAMPLES


    for i in range(env.SAMPLES):
        for j in range(env.TRIALS):
            assert (env.GRID[i][j] == env.shape[i + 3][j])


def test_shape_execute_all():
    env = CustomEnvironment()

    state_reward_done = []
    for step in range(97):
        state_reward_done.append(env.execute(actions=0))

    assert len(env.shape[0]) == env.TRIALS
    assert len(env.shape) == env.shapeHeight

    for i in range(env.TRIALS):
        if i >= env.startingPoint:
            assert env.shape[0][i] == i
            assert env.shape[1][i] >= 0
            assert env.shape[1][i] < env.SAMPLES

    for i in range(env.SAMPLES):
        for j in range(env.TRIALS):
            assert (env.GRID[i][j] == env.shape[i + 3][j])


    print(env.GRID)
    print(env.shape)
