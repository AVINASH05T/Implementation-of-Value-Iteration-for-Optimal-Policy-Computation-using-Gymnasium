# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

## NAME : AVINASH T
## REG NO : 212223230026
---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement

Develop a Python program that applies the Value Iteration algorithm to the FrozenLake-v1 environment provided by Gymnasium. The algorithm should iteratively update the value of each state until convergence and then derive the optimal policy that maximizes the expected cumulative reward.


## Software Requirements

Python 3.x
Gymnasium
NumPy
Jupyter Notebook / Google Colab / VS Code


## Environment Description

The FrozenLake-v1 environment is a grid-world problem in which an agent must move from the Start (S) state to the Goal (G) while avoiding Holes (H).

Grid Used:

F F S F
F H H F
F F G H
F F F H
Where:

S – Start State
F – Frozen Surface (Safe)
H – Hole (Terminal State)
G – Goal State (Reward = 1)
The environment is stochastic (is_slippery=True), meaning the intended action may not always be executed.


## MDP Representation

An MDP is represented as:

MDP = (S, A, P, R, γ)

Where:

S = Set of states (16 states)
A = {Left, Down, Right, Up}
P(s'|s,a) = Transition probability
R(s,a,s') = Reward function
γ = 0.99 = Discount factor



## Theory

Value Iteration is a Dynamic Programming algorithm used to compute the optimal value function of an MDP.

It repeatedly updates the value of each state using the Bellman Optimality Equation:

[ V(s)=\max_a\sum_{s'}P(s'|s,a)\left[R(s,a,s')+\gamma V(s')\right] ]

The iterations continue until the maximum change in the value function is smaller than a predefined threshold.

After convergence, the optimal policy is obtained by selecting the action that gives the highest expected value.


## Algorithm

1. Create the FrozenLake environment.
2. Initialize the value function of all states to zero.
3. Repeat until convergence:
4. Compute the value for every possible action.
5. Update each state's value using the Bellman Optimality Equation.
6. Calculate the maximum difference between old and new values.
7. Stop when the difference becomes less than the threshold.
8. Extract the optimal policy by selecting the action with the highest value for every state.
9. Display the optimal value function and policy.



## Python Program

```python



# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------
env_desc = [
    "SFHF",
    "FHFH",
    "HFFG",
    "HHFF"
]

env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)
env = env.unwrapped

def value_iteration(env, gamma=0.99, theta=1e-8):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    V = np.zeros(num_states) 
    iterations = 0

    while True:
        delta = 0
        V_new = np.copy(V)
        for s in range(num_states):
            q_values = np.zeros(num_actions)
            for a in range(num_actions):
                for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                    q_values[a] += prob * (reward + gamma * V[next_state])
            
            V_new[s] = np.max(q_values)
            delta = max(delta, np.abs(V_new[s] - V[s]))
        
        V = V_new
        iterations += 1

        if delta < theta:
            break

    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        q_values = np.zeros(num_actions)
        for a in range(num_actions):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                q_values[a] += prob * (reward + gamma * V[next_state])
        policy[s] = np.argmax(q_values)
    return V, policy, iterations

# -------------------------------------------------
# Run Value Iteration
# -------------------------------------------------

V, policy, iteration = value_iteration(env)

# -------------------------------------------------
# Display Output
# -------------------------------------------------
print("Name: AVINASH T")
print("Register Number: 212223230026")
print("Value Iteration Completed")
print("Number of Iterations:", iteration)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))


action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

env.close()


```

---

## Output

<img width="581" height="292" alt="image" src="https://github.com/user-attachments/assets/169ac401-5d8a-47c3-bb2b-2c9b334ed9a6" />



## Result

The Value Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The optimal state-value function and optimal policy were computed after convergence using the Bellman Optimality Equation.


## Inference

From this experiment, it is observed that the Value Iteration algorithm efficiently computes the optimal value of every state by repeatedly applying the Bellman Optimality Equation. Once the value function converges, the optimal policy is extracted by selecting the action with the highest expected return. This demonstrates how Dynamic Programming can solve finite Markov Decision Processes and determine the best sequence of actions for an agent.

