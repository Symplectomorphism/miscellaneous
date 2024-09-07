import numpy as np
from tqdm import tqdm


def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi

def rmse(x, y, dp=4):
    return np.round(np.sqrt(np.mean((x - y)**2)), dp)    

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def ewma(past_value, current_value, alpha):
    return (1-alpha) * past_value + alpha * current_value

def calc_ewma(values, period):
    alpha = 2 / (period + 1)
    result = []
    for v in values:
        try:
            prev_value = result[-1]
        except IndexError:
            prev_value = 0
        
        new_value = ewma(prev_value, v, alpha)
        result.append(new_value)
    return np.array(result)

    
def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps
    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values

def q_learning(env, 
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float63)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float63)
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
        if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, 
                           min_alpha, 
                           alpha_decay_ratio, 
                           n_episodes)
    epsilons = decay_schedule(init_epsilon, 
                              min_epsilon, 
                              epsilon_decay_ratio, 
                              n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False
        while not done:
            action = select_action(state, Q, epsilons[e])
            next_state, reward, done, _ = env.step(action)
            td_target = reward + gamma * Q[next_state].max() * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[e] * td_error
            state = next_state

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=0))

    V = np.max(Q, axis=0)        
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=0))}[s]
    return Q, V, pi, Q_track, pi_track



    
"""
Define P[s][a] for the vanilla GridWorld
0: right, 1: up, 2: left, 3: down

Has row-major ordering so 
    k -> (i, j) = (k % size, k // size)
    (i, j) -> k = j * size + i
"""    
P_gridworld = {
    0: {
        0: [(1.0, 1, -1, False)],
        1: [(1.0, 0, -1, False)],
        2: [(1.0, 0, -1, False)],
        3: [(1.0, 5, -1, False)],
    },
    1: {
        0: [(1.0, 2, -1, False)],
        1: [(1.0, 1, -1, False)],
        2: [(1.0, 0, -1, False)],
        3: [(1.0, 6, -1, False)],
    },
    2: {
        0: [(1.0, 3, -1, False)],
        1: [(1.0, 2, -1, False)],
        2: [(1.0, 1, -1, False)],
        3: [(1.0, 7, -1, False)],
    },
    3: {
        0: [(1.0, 4, -1, False)],
        1: [(1.0, 3, -1, False)],
        2: [(1.0, 2, -1, False)],
        3: [(1.0, 8, -1, False)],
    },
    4: {
        0: [(1.0, 4, -1, False)],
        1: [(1.0, 4, -1, False)],
        2: [(1.0, 3, -1, False)],
        3: [(1.0, 9, -1, False)],
    },
    5: {
        0: [(1.0, 6, -1, False)],
        1: [(1.0, 0, -1, False)],
        2: [(1.0, 5, -1, False)],
        3: [(1.0, 10, -1, False)],
    },
    6: {
        0: [(1.0, 7, -1, False)],
        1: [(1.0, 1, -1, False)],
        2: [(1.0, 5, -1, False)],
        3: [(1.0, 11, -1, False)],
    },
    7: {
        0: [(1.0, 8, -1, False)],
        1: [(1.0, 2, -1, False)],
        2: [(1.0, 6, -1, False)],
        3: [(1.0, 12, -1, False)],
    },
    8: {
        0: [(1.0, 9, -1, False)],
        1: [(1.0, 3, -1, False)],
        2: [(1.0, 7, -1, False)],
        3: [(1.0, 13, -1, False)],
    },
    9: {
        0: [(1.0, 9, -1, False)],
        1: [(1.0, 4, -1, False)],
        2: [(1.0, 8, -1, False)],
        3: [(1.0, 14, -1, False)],
    },
    10: {
        0: [(1.0, 11, -1, False)],
        1: [(1.0, 5, -1, False)],
        2: [(1.0, 10, -1, False)],
        3: [(1.0, 15, -1, False)],
    },
    11: {
        0: [(1.0, 12, -1, False)],
        1: [(1.0, 6, -1, False)],
        2: [(1.0, 10, -1, False)],
        3: [(1.0, 16, -1, False)],
    },
    12: {
        0: [(1.0, 13, -1, False)],
        1: [(1.0, 7, -1, False)],
        2: [(1.0, 11, -1, False)],
        3: [(1.0, 17, -1, False)],
    },
    13: {
        0: [(1.0, 14, -1, False)],
        1: [(1.0, 8, -1, False)],
        2: [(1.0, 12, -1, False)],
        3: [(1.0, 18, -1, False)],
    },
    14: {
        0: [(1.0, 14, -1, False)],
        1: [(1.0, 9, -1, False)],
        2: [(1.0, 13, -1, False)],
        3: [(1.0, 19, -1, False)],
    },
    15: {
        0: [(1.0, 16, -1, False)],
        1: [(1.0, 10, -1, False)],
        2: [(1.0, 15, -1, False)],
        3: [(1.0, 20, -1, False)],
    },
    16: {
        0: [(1.0, 17, -1, False)],
        1: [(1.0, 11, -1, False)],
        2: [(1.0, 15, -1, False)],
        3: [(1.0, 21, -1, False)],
    },
    17: {
        0: [(1.0, 18, -1, False)],
        1: [(1.0, 12, -1, False)],
        2: [(1.0, 16, -1, False)],
        3: [(1.0, 22, -1, False)],
    },
    18: {
        0: [(1.0, 19, -1, False)],
        1: [(1.0, 13, -1, False)],
        2: [(1.0, 17, -1, False)],
        3: [(1.0, 23, -1, False)],
    },
    19: {
        0: [(1.0, 19, -1, False)],
        1: [(1.0, 14, -1, False)],
        2: [(1.0, 18, -1, False)],
        3: [(1.0, 24, -1, False)],
    },
    20: {
        0: [(1.0, 21, -1, False)],
        1: [(1.0, 15, -1, False)],
        2: [(1.0, 20, -1, False)],
        3: [(1.0, 20, -1, False)],
    },
    21: {
        0: [(1.0, 22, -1, False)],
        1: [(1.0, 16, -1, False)],
        2: [(1.0, 20, -1, False)],
        3: [(1.0, 21, -1, False)],
    },
    22: {
        0: [(1.0, 23, -1, False)],
        1: [(1.0, 17, -1, False)],
        2: [(1.0, 21, -1, False)],
        3: [(1.0, 22, -1, False)],
    },
    23: {
        0: [(1.0, 24, -1, True)],
        1: [(1.0, 18, -1, False)],
        2: [(1.0, 22, -1, False)],
        3: [(1.0, 23, -1, False)],
    },
    24: {
        0: [(1.0, 24, 0, True)],
        1: [(1.0, 24, 0, True)],
        2: [(1.0, 24, 0, True)],
        3: [(1.0, 24, 0, True)],
    },
}