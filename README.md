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

import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
env = env.unwrapped

n_states = env.observation_space.n
n_actions = env.action_space.n

gamma = 0.99
theta = 1e-8

policy = np.ones((n_states, n_actions)) / n_actions


def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):

    V = np.zeros(n_states)

    while True:

        delta = 0

        for s in range(n_states):

            v = V[s]
            value = 0

            for a in range(n_actions):

                action_prob = policy[s][a]

                for prob, next_state, reward, done in env.P[s][a]:
                    value += action_prob * prob * (
                        reward + gamma * V[next_state]
                    )

            V[s] = value
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V


V = policy_evaluation(env, policy, gamma, theta)

print("Policy Evaluation - Value Function")
print("")
print(np.round(V.reshape(4,4),4))

def policy_improvement(env, V, gamma=0.99):

    policy = np.zeros((n_states,n_actions))

    for s in range(n_states):

        action_values = np.zeros(n_actions)

        for a in range(n_actions):

            for prob,next_state,reward,done in env.P[s][a]:
                action_values[a] += prob*(reward+gamma*V[next_state])

        best_action=np.argmax(action_values)

        policy[s][best_action]=1

    return policy

policy = policy_improvement(env,V,gamma)

action_symbols={
    0:"←",
    1:"↓",
    2:"→",
    3:"↑"
}

best_actions=np.argmax(policy,axis=1)

policy_grid=np.array(
    [action_symbols[a] for a in best_actions]
).reshape(4,4)

print("Policy Improvement")
print("")
print(policy_grid)

def policy_iteration(env,policy,gamma=0.99,theta=1e-8):

    while True:

        V=policy_evaluation(env,policy,gamma,theta)

        new_policy=policy_improvement(env,V,gamma)

        if np.array_equal(policy,new_policy):
            break

        policy=new_policy

    return policy,V

 optimal_policy,optimal_value_function=policy_iteration(
    env,
    policy,
    gamma,
    theta
)

# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(V):
    print("\nOptimal State-Value Function:")
    print(np.round(V.reshape(4, 4), 4))


def print_policy(policy):

    action_symbols = {
        0: "←",
        1: "↓",
        2: "→",
        3: "↑"
    }

    best_actions = np.argmax(policy, axis=1)

    policy_grid = np.array(
        [action_symbols[action] for action in best_actions]
    ).reshape(4, 4)

    print("\nOptimal Policy:")
    print("")
    print(policy_grid)


# -------------------------------------------------
# Run Policy Iteration
# -------------------------------------------------

optimal_policy, optimal_value_function = policy_iteration(
    env,
    policy,
    gamma,
    theta
)

print("Name: AVINASH T")
print("Register Number: 212223230026")

print_value_function(optimal_value_function)
print_policy(optimal_policy)

env.close()


```

---

## Output


<img width="644" height="123" alt="image" src="https://github.com/user-attachments/assets/d66e44f2-64f8-4cfd-aa1f-549b7e2a00c9" />

<img width="685" height="113" alt="image" src="https://github.com/user-attachments/assets/a475cf7f-19e0-4522-8e21-bf2381311fc3" />


<img width="814" height="272" alt="image" src="https://github.com/user-attachments/assets/b37e5a72-c818-4eb5-825b-4066144283e3" />


## Result

The Value Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The optimal state-value function and optimal policy were computed after convergence using the Bellman Optimality Equation.


## Inference

From this experiment, it is observed that the Value Iteration algorithm efficiently computes the optimal value of every state by repeatedly applying the Bellman Optimality Equation. Once the value function converges, the optimal policy is extracted by selecting the action with the highest expected return. This demonstrates how Dynamic Programming can solve finite Markov Decision Processes and determine the best sequence of actions for an agent.

