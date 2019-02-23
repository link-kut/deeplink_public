# -*- coding: utf-8 -*-
import random
from environment import GraphicDisplay, Env
import copy

class PolicyIteration:
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

        # list of random policy (same probability of up, down, left, right)
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width for _ in range(env.height)]

        # setting terminal state
        self.policy_table[2][2] = []

        print("\n**Initial Policy Table**")
        for i in range(env.height):
            print(self.policy_table[i])

        self.discount_factor = 0.9

    def policy_evaluation(self):
        next_value_table = [[0.00] * self.env.width for _ in range(self.env.height)]

        # Bellman Expectation Equation for the every states
        for state in self.env.all_states:
            value = 0.0

            # keep the value function of terminal states as 0
            if state == [2, 2]:
                next_value_table[2][2] = value
                continue

            for action in self.env.possible_actions:
                next_state, reward = self.env.get_next_state_and_reward(state, action)
                next_value = self.get_value(next_state)
                value += (self.get_policy(state)[action] * (reward + self.discount_factor * next_value))

            next_value_table[state[0]][state[1]] = round(value, 2)

        self.value_table = next_value_table

        print("\n**Value Table**")
        for i in range(env.height):
            print(self.value_table[i])

    def policy_improvement(self):
        next_policy = copy.deepcopy(self.policy_table)

        for state in self.env.all_states:
            if state == [2, 2]:
                continue
            max_value = -99999
            max_index = []
            result = [0.0, 0.0, 0.0, 0.0]  # initialize the policy

            # for every actions, calculate
            # [reward + (discount factor) * (next state value function)]
            for index, action in enumerate(self.env.possible_actions):
                next_state, reward = self.env.get_next_state_and_reward(state, action)
                next_value = self.get_value(next_state)
                temp = reward + self.discount_factor * next_value

                # We normally can't pick multiple actions in greedy policy.
                # but here we allow multiple actions with same max values
                if temp == max_value:
                    max_index.append(index)
                elif temp > max_value:
                    max_value = temp
                    max_index.clear()
                    max_index.append(index)

            # probability of action
            prob = 1.0 / len(max_index)

            for index in max_index:
                result[index] = prob

            next_policy[state[0]][state[1]] = result

        self.policy_table = next_policy

        print("\n**Policy Table**")
        for i in range(env.height):
            print(self.policy_table[i])

    # get action according to the current policy
    def get_action(self, state):
        random_pick = random.randrange(100) / 100

        policy = self.get_policy(state)
        policy_sum = 0.0

        # return the action in the index
        for index, value in enumerate(policy):
            policy_sum += value
            if policy_sum > random_pick:
                return index

    # get policy of specific state
    def get_policy(self, state):
        if state == [2, 2]:
            return 0.0
        return self.policy_table[state[0]][state[1]]

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)


if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration, "Policy Iteration")
    grid_world.mainloop()
