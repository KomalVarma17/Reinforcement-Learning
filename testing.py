import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym



def truncate_state(state):
    truncated = []
    for i in range(len(state)):
        if i == 0 or i == 1:
            truncated.append(min(state[i], 20))
        else:
            truncated.append(state[i])
    return tuple(truncated)

def TestPolicy(env, policy):

    state, _ = env.reset()
    state = truncate_state(state)

    done = False
    truncated_flag = False
    timestep = 0

    q1_list = []
    q2_list = []
    actions_taken = []
    total_queue_sum = 0

    while not (done or truncated_flag):
        action = policy[state]

        next_state, reward, done, truncated_flag, _ = env.step(action)
        next_state = truncate_state(next_state)

        q1_list.append(env.q1)
        q2_list.append(env.q2)
        actions_taken.append(action)
        total_queue_sum += (env.q1 + env.q2)

        state = next_state
        timestep += 1


    plt.figure(figsize=(10, 6))
    plt.plot(range(timestep), q1_list, label='Queue 1 (Road 1)')
    plt.plot(range(timestep), q2_list, label='Queue 2 (Road 2)')
    plt.xlabel('Time Steps')
    plt.ylabel('Queue Length')
    plt.title('Queue Lengths vs Time')
    plt.legend()
    plt.grid(True)
    plt.show()


    avg_queue_sum = total_queue_sum / timestep
    print(f"\n Average Sum of Queue Lengths over 1 Episode: {avg_queue_sum:.2f}")
    print(f"\n Actions Taken Over Time:\n{actions_taken}")



env = GymTrafficEnv() # Create and instance of the traffic controller environment.

#policy = np.load('policy1.npy') # To load SARSA policy.
policy = np.load('policy2.npy') # To load ExpectedSARSA policy.
#policy = np.load('policy3.npy') # To load ValueFunctionSARSA policy.

TestPolicy(env, policy)

env.close() # Close the environment

