from environment import *
import random

class ValueIterationGraphicDisplay(GraphicDisplay):
    def __init__(self, agent, title):
        self.btn_1_text = "Calculate"
        self.btn_2_text = "Print Policy"
        self.btn_1_func = self.calculate_value
        self.btn_2_func = self.print_optimal_policy
        self.btn_3_func = self.move_by_value_iteration
        GraphicDisplay.__init__(self, agent, title)

    def move_by_value_iteration(self):
        if self.improvement_count != 0 and self.is_moving != 1:
            self.is_moving = 1

            x, y = self.canvas.coords(self.rectangle)
            self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)

            x, y = self.find_rectangle()
            while len(self.agent.get_action([x, y])) != 0:
                action = random.sample(self.agent.get_action([x, y]), 1)[0]
                self.after(100, self.rectangle_move(action))
                x, y = self.find_rectangle()
            self.is_moving = 0

    def draw_one_arrow(self, col, row, action):
        if col == 2 and row == 2:
            return
        if action == 0:  # up
            origin_x, origin_y = 50 + (UNIT * row), 10 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.up))

        elif action == 1:  # down
            origin_x, origin_y = 50 + (UNIT * row), 90 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.down))

        elif action == 3:  # right
            origin_x, origin_y = 90 + (UNIT * row), 50 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.right))

        elif action == 2:  # left
            origin_x, origin_y = 10 + (UNIT * row), 50 + (UNIT * col)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.left))

    def draw_from_values(self, state, action_list):
        i = state[0]
        j = state[1]
        for action in action_list:
            self.draw_one_arrow(i, j, action)

    def calculate_value(self):
        self.iter_count += 1
        for i in self.texts:
            self.canvas.delete(i)
        self.agent.value_iteration()
        self.print_value_table(self.agent.value_table)

    def print_optimal_policy(self):
        self.improvement_count += 1
        for i in self.arrows:
            self.canvas.delete(i)
        for state in self.env.all_states:
            action = self.agent.get_action(state)
            self.draw_from_values(state, action)