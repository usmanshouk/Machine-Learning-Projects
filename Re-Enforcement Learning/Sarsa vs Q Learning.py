"""
COMP 532 - REINFORCEMENT LEARNING ASSIGNMENT 1
SUBMITTED BY : GROUP 1
USMAN SHOUKAT   					     STUDENT ID: 201537600
JONE CHONG							     STUDENT ID: 201533109
SAHIB BIR SINGH BHATIA 			 	     STUDENT ID: 201547831
PARTH KHANDELWAL					     STUDENT ID: 201549274
"""

import numpy as np
from numpy.core.fromnumeric import argmax
from tqdm import tqdm
import matplotlib.pyplot as plt

# initialization of environment
heightOfGrid = 4
widthOfGrid = 12
start = [3, 0]
end = [3, 11]
left = 2
up = 0
right = 3
down = 1
actions = [left, up, right, down]
eps = 0.1
gamma = 1
alpha = 0.5


# alpha = 0.1


def decidingAction(Q_Value, state):
    '''
    This Function takes the current Q values and current state and depending upon the epsilon 
    greedy policy, recommends the action

    Parameters:
        Current Q Values
        Current State

    return: 
        Action


    '''
    # Choosing the random value to decide whether to explore or exploit while comparing with epsilon
    # value.
    prob = np.random.rand()
    if prob < eps:  # If chosen random value is less than epsilon then explore
        return int(np.random.choice(actions))
    else:  # And if chosen random value is less than epsilon then exploit
        values = Q_Value[:, state[0], state[1]]
        return int(np.random.choice([action for action, value in enumerate(values) if value == np.max(values)]))


def takingStep(state, action):
    '''
    This function takes the state and decided action, and depending upon that return the next state
    and reward
    Parameters:
        Current State
        Decided Action
    Return:
        Next State
        Reward for taking the step 
    '''
    # max and min function is used in below if statements, so that agent doesn't fall off the grid
    i, j = state
    if action == up:
        nextState = [max(0, i - 1), j]
    elif action == down:
        nextState = [min(heightOfGrid - 1, i + 1), j]
    elif action == left:
        nextState = [i, max(0, (j - 1))]
    elif action == right:
        nextState = [i, min(widthOfGrid - 1, (j + 1))]

    reward = -1
    # This if statement takes care of penalizing the agent with -100 if it falls off the grid
    if (action == down and i == 2 and 1 <= j and j <= 10) or (action == right and i == 3 and j == 0):
        reward = -100
        nextState = start

    return nextState, reward


def sarsa(Q_Sarsa, episodes, rewardsSarsa):
    '''
    This function guide the agent to move around in the grid and then update the Q Values depending on 
    SARSA approach for given number of episodes. It also takes count of cumulative reward which agent
    gets  for each episode
    Parameter:
        Q Values initialised with zero
        No. of episodes
        Empty array for rewards
    '''
    for i in range(episodes):
        state = start
        # This loop  end  when agent has completed the episode and has reached the goal state
        while state != end:
            action = decidingAction(Q_Sarsa, state)
            nextState, reward = takingStep(state, action)
            rewardsSarsa[i] += reward
            nextAction = decidingAction(Q_Sarsa, nextState)
            # calculating Q values
            Q_Sarsa[action, state[0], state[1]] += alpha * (
                        reward + gamma * Q_Sarsa[nextAction, nextState[0], nextState[1],] - Q_Sarsa[
                    action, state[0], state[1]])
            state = nextState


def qLearning(Q_QLearnig, episodes, rewardsQLearning):
    '''
    This function guide the agent to move around in the grid and then update the Q Values depending on 
    Q-Learning approach for given number of episodes. It also takes count of cumulative reward which agent
    gets  for each episode
    Parameter:
        Q Values initialised with zero
        No. of episodes
        Empty array for rewards
    '''
    for i in range(episodes):
        state = start
        # This loop  end  when agent has completed the episode and has reached the goal state
        while state != end:
            action = decidingAction(Q_QLearnig, state)
            nextState, reward = takingStep(state, action)
            rewardsQLearning[i] += reward
            # Computing the Q values using QQ learning approach
            Q_QLearnig[action, state[0], state[1]] += alpha * (
                        reward + (gamma * np.max(Q_QLearnig[:, nextState[0], nextState[1]])) - Q_QLearnig[
                    action, state[0], state[1],])
            state = nextState


iterations = 100
episodes = 500
# Initialising of rewards arrays
rewardsSarsa = np.zeros(episodes)
rewardsQLearning = np.zeros(episodes)
for i in tqdm(range(iterations)):
    # Initializing of Q Values for each iteration
    Q_Sarsa = np.zeros((len(actions), heightOfGrid, widthOfGrid))
    Q_QLearnig = np.zeros((len(actions), heightOfGrid, widthOfGrid))
    # Calling of SARSA and QLearning function
    sarsa(Q_Sarsa, episodes, rewardsSarsa)
    qLearning(Q_QLearnig, episodes, rewardsQLearning)

# summing the rewards on number of iterations
rewardsSarsa /= iterations
rewardsQLearning /= iterations
smoothenedSasrsa = []
smoothenedQLearning = []

# smoothing the rewards on moving average of 10
for i in range(0, 500, 10):
    smoothenedSasrsa.append(sum(rewardsSarsa[i:i + 10]) / 10)
    smoothenedQLearning.append(sum(rewardsQLearning[i:i + 10]) / 10)


def optimal_policy(QValue):
    '''
    This function takes the Q Values and then print the optimal policy depending upon the given 
    Q values
    Parameter:
        Q values
    '''
    policy = []
    for i in range(QValue.shape[1]):
        row = []
        for j in range(QValue.shape[2]):
            if (i == 3 and j == 11):
                row.append("G")
            elif (i == 3 and 0 < j and j < 11):
                row.append("-")
            else:
                max = argmax(QValue[:, i, j])
                if max == 0:
                    row.append("U")
                elif max == 1:
                    row.append("D")
                elif max == 2:
                    row.append("L")
                elif max == 3:
                    row.append("R")

        policy.append(row)

    for row in policy:
        print(row)


# Plotting the graph
x = range(0, 500, 10)
plt.plot(x, smoothenedSasrsa, color="r", label="Sarsa")
plt.plot(x, smoothenedQLearning, color="b", label="Q learning")
plt.yticks([-25, -50, -75, -100])
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards during episode')
plt.xlim([0, 500])
plt.ylim([-100, -20])
plt.show()
plt.close()
