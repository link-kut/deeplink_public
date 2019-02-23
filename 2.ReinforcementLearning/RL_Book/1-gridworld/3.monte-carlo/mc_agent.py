# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
from environment import Env
import time


MIN_INT = -100000000000

# Monte Carlo Agent which learns every episodes from the sample
class MCAgent:
    def __init__(self, env, actions):
        self.width = 5
        self.height = 5
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.samples = []
        self.value_table = defaultdict(float)

        print("**Env - States**")
        for i in range(env.height):
            for j in range(env.width):
                print(env.all_states[i * env.width + j], end=" ")
            print()

        # 2-d list for the value function
        print("\n**Initial Value Table**")
        print(self.value_table)

    # append sample to memory(state, reward, done)
    def save_sample(self, state, reward, done):
        self.samples.append([state, reward, done])

    # for every episode, the agent updates values of visited states
    def update(self):
        G_t = 0
        visit_state = []
        for sample in reversed(self.samples):
            state = str(sample[0])
            if state not in visit_state:
                visit_state.append(state)
                G_t = self.discount_factor * (sample[1] + G_t)
                value = self.value_table[state]
                self.value_table[state] = value + self.learning_rate * (G_t - value)
                env.print_value_table(self.value_table, self.samples)

    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # take random action
            action = np.random.choice(self.actions)
        else:
            # take action according to the q function table
            next_state_values = self.possible_next_state_values(state)
            action = self.arg_max(next_state_values)
        return int(action)

    # compute arg_max if multiple candidates exit, pick one randomly
    @staticmethod
    def arg_max(next_state):
        max_index_list = []
        max_value = next_state[0]
        for index, value in enumerate(next_state):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)

        return random.choice(max_index_list)

    # get the possible next states
    def possible_next_state_values(self, state):
        col, row = state
        next_state_values = [0.0] * 4

        if row != 0:
            next_state_values[0] = self.value_table[str([col, row - 1])]
        else:
            next_state_values[0] = MIN_INT

        if row != self.height - 1:
            next_state_values[1] = self.value_table[str([col, row + 1])]
        else:
            next_state_values[1] = MIN_INT

        if col != 0:
            next_state_values[2] = self.value_table[str([col - 1, row])]
        else:
            next_state_values[2] = MIN_INT

        if col != self.width - 1:
            next_state_values[3] = self.value_table[str([col + 1, row])]
        else:
            next_state_values[3] = MIN_INT

        return next_state_values


# main loop
if __name__ == "__main__":
    env = Env()
    agent = MCAgent(env, actions=list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()
        action = agent.get_action(state)
        accumulated_reward = 0
        while True:
            env.render()

            # forward to next state. reward is number and done is boolean
            next_state, reward, done = env.step(action)
            print(action, "->", next_state, reward, done)

            accumulated_reward += reward
            agent.save_sample(next_state, reward, done)

            # get next action
            action = agent.get_action(next_state)

            # at the end of each episode, update the q function table
            if done:
                print("episode: {0}, accumulated reward: {1}".format(episode, accumulated_reward))
                agent.update()
                agent.samples.clear()
                accumulated_reward = 0
                break

