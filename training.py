import numpy as np
import gymnasium as gym
from gym import spaces

# epsilon greedy policy function to choose a policy
def epsilon_greedy(q_values, epsilon, state, action_space, nu=1e-5):
    Q = q_values[state]
    q_normalized = Q / (nu + np.sum(np.abs(Q)))  
    exp_q = np.exp(q_normalized - np.max(q_normalized))  
    softmax_probs = exp_q / np.sum(exp_q)

    if np.random.rand() < epsilon:
        action = np.random.choice(action_space, p=softmax_probs) 
    else:
        action = np.argmax(Q) 

    return action, softmax_probs


def truncate_state(state):
    truncated = []
    for i in range(len(state)):
        if i == 0 or i == 1:
            truncated.append(int(min(state[i], 20)))
        else:
            truncated.append(int(state[i]))
    return tuple(truncated)


def SARSA(env, beta, Nepisodes, alpha):
    state_space_size = env.observation_space.nvec
    action_space_size = env.action_space.n
    q_values = np.random.uniform(low=-1, high=1, size=tuple(state_space_size) + (action_space_size,))

    epsilon = 1.0
    decay_rate = 0.997

    for episode in range(Nepisodes):
        state, _ = env.reset()
        state = truncate_state(state)
        done = False
        truncated = False
        timestep = 0

        action, _ = epsilon_greedy(q_values, epsilon, state, range(action_space_size))

        while not (done or truncated):
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = truncate_state(next_state)

            next_action, _ = epsilon_greedy(q_values, epsilon, next_state, range(action_space_size))

            q_values[state][action] += alpha * (reward + beta * q_values[next_state][next_action] - q_values[state][action])

            state, action = next_state, next_action
            timestep += 1

        epsilon = max(epsilon * decay_rate, 0.05)

    optimal_policy = np.argmax(q_values, axis=-1)
    return optimal_policy


def ExpectedSARSA(env, beta, Nepisodes, alpha):
    state_space_shape = tuple(env.observation_space.nvec)
    action_space_size = env.action_space.n
    nu = 1e-5
    epsilon = 1.0
    decay_rate = 0.997

    q_values = np.random.uniform(low=-1, high=1, size=state_space_shape + (action_space_size,))

    for episode in range(Nepisodes):
        state, _ = env.reset()
        state = truncate_state(state)
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = epsilon_greedy(q_values, epsilon, state, range(action_space_size), nu)

            next_state, reward, done, truncated, _ = env.step(action)
            next_state = truncate_state(next_state)

            _, next_policy_probs = epsilon_greedy(q_values, 1.0, next_state, range(action_space_size), nu)

            expected_q = np.dot(next_policy_probs, q_values[next_state])

            q_values[state][action] += alpha * (reward + beta * expected_q - q_values[state][action])

            state = next_state

        epsilon = max(epsilon * decay_rate, 0.05)

    optimal_policy = np.argmax(q_values, axis=-1)
    return optimal_policy

# Computing q-values from value function
def compute_q_from_v_array(V, env, state, beta):
    q_vals = []
    for action in range(env.action_space.n):
        saved = (env.q1, env.q2, env.light, env.delta, env.last_green, env.time)
        env.q1, env.q2, env.light, env.delta, env.last_green, env.time = \
            state[0], state[1], state[2], state[3], state[2], env.time

        next_state, reward, _, _, _ = env.step(action)
        next_state = truncate_state(next_state)
        q_vals.append(reward + beta * V[next_state])

        env.q1, env.q2, env.light, env.delta, env.last_green, env.time = saved

    return np.array(q_vals)


def ValueFunctionSARSA(env, beta, Nepisodes, alpha):
    state_space_shape = tuple(env.observation_space.nvec)
    V = np.random.uniform(low=-0.5, high=0.5, size=state_space_shape)
    epsilon = 1.0
    decay_rate = 0.997

    for episode in range(Nepisodes):
        state, _ = env.reset()
        state = truncate_state(state)
        done = False
        truncated_flag = False

        q_vals = compute_q_from_v_array(V, env, state, beta)
        action, _ = epsilon_greedy({state: q_vals}, epsilon, state, range(env.action_space.n))

        while not (done or truncated_flag):
            next_state, reward, done, truncated_flag, _ = env.step(action)
            next_state = truncate_state(next_state)

            v_current = V[state]
            v_next = V[next_state]
            V[state] = v_current + alpha * (reward + beta * v_next - v_current)

            state = next_state
            q_vals = compute_q_from_v_array(V, env, state, beta)
            action, _ = epsilon_greedy({state: q_vals}, epsilon, state, range(env.action_space.n))

        epsilon = max(epsilon * decay_rate, 0.05)

    optimal_policy = np.zeros(state_space_shape, dtype=int)
    for q1 in range(21):
        for q2 in range(21):
            for light in range(3):
                for delta in range(11):
                    s = (q1, q2, light, delta)
                    q_vals = compute_q_from_v_array(V, env, s, beta)
                    optimal_policy[s] = int(np.argmax(q_vals))

    return optimal_policy

# --- Main ---
env = GymTrafficEnv()

Nepisodes = 2000
alpha = 0.1
beta = 0.997

policy1 = SARSA(env, beta, Nepisodes, alpha)
policy2 = ExpectedSARSA(env, beta, Nepisodes, alpha)
policy3 = ValueFunctionSARSA(env, beta, Nepisodes, alpha)

np.save('policy1.npy', policy1)
np.save('policy2.npy', policy2)
np.save('policy3.npy', policy3)

env.close()
