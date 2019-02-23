# -*- coding: utf-8 -*-
import sys
from environment import GraphicDisplay, Env

class ValueIteration:
    def __init__(self, env):
        self.env = env

        print("**Env - States**")
        for i in range(env.height):
            for j in range(env.width):
                print(self.env.all_states[i * env.width + j], end=" ")
            print()

        print("\n**Env - Rewards**")
        for i in range(env.height):
            print(self.env.reward[i])

        # 2-d list for the value function
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        print("\n**Initial Value Table**")
        for i in range(env.height):
            print(self.value_table[i])

        self.discount_factor = 0.9

    # get next value function table from the current value function table
    def value_iteration(self):
        next_value_table = [[0.0] * self.env.width for _ in range(self.env.height)]

        for state in self.env.all_states:
            if state == [2, 2]:
                next_value_table[2][2] = 0.0
                continue

            value_list = []
            for action in self.env.possible_actions:
                next_state, reward = self.env.get_next_state_and_reward(state, action)
                next_value = self.get_value(next_state)
                value_list.append((reward + self.discount_factor * next_value))

            # return the maximum value (it is the optimality equation!!)
            next_value_table[state[0]][state[1]] = round(max(value_list), 2)

        self.value_table = next_value_table

        print("\n**Value Table**")
        for i in range(env.height):
            print(self.value_table[i])

    # get action according to the current value function table
    def get_action(self, state):
        action_list = []
        max_value = -99999

        if state == [2, 2]:
            return []

        # calculating q values for the all actions and
        # append the action to action list which has maximum q value
        for action in self.env.possible_actions:
            next_state, reward = self.env.get_next_state_and_reward(state, action)
            next_value = self.get_value(next_state)
            value = (reward + self.discount_factor * next_value)

            if value > max_value:
                action_list.clear()
                action_list.append(action)
                max_value = value
            elif value == max_value:
                action_list.append(action)

        return action_list

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)

if __name__ == "__main__":
    env = Env()
    value_iteration = ValueIteration(env)
    grid_world = GraphicDisplay(value_iteration, "Value Iteration")
    grid_world.mainloop()
