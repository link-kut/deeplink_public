from sys import maxsize

class Node(object):
    def __init__(self, depth, player_num, state, value=0, select=-1):
        self.depth = depth
        self.player_num = player_num
        self.state = state
        self.value = value
        self.select = select
        self.children = []
        self.create_children()

    def create_children(self):
        if abs(self.value) == maxsize:
            return

        if self.depth >= 0:
            for i in range(9):
                if self.state[i] == 0:
                    child_state = self.state[:]
                    child_state[i] = self.player_num
                    self.children.append(
                        Node(
                            depth=self.depth - 1,
                            player_num=-self.player_num,
                            state=child_state,
                            value=self.check_val(child_state),
                            select=i
                        )
                    )

    def check_val(self, state):
        for (x, y, z) in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            if state[x] == state[y] == state[z]:
                if state[x] == 1:
                    return maxsize
                elif state[x] == -1:
                    return -maxsize

        if [i for i in range(9) if state[i] == 0] == []:
            return -1

        return 0


def MinMax(node, depth, player_num):
    if depth == 0 or abs(node.value) == maxsize:
        return node.value

    exist_empty = False
    for i in range(9):
        if node.state[i] == 0:
            exist_empty = True
            break

    if not exist_empty:
        return -1

    best_value = maxsize * -player_num

    for i in range(len(node.children)):
        child = node.children[i]
        value = MinMax(child, depth - 1, -player_num)
        if abs(maxsize * player_num - value) < abs(maxsize * player_num - best_value):
            best_value = value

    return best_value


def win_check(state):
    for (x, y, z) in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
        if state[x] == state[y] == state[z]:
            if state[x] == 1:
                print("You Win!!")
                return maxsize
            elif state[x] == -1:
                print("Computer Win!!")
                return -maxsize

    if len([i for i in range(9) if state[i] == 0]) == 0:
        print("Draw!!")
        return -1

    return 0


def print_game(state):
    a = ""
    for i in range(3):
        for j in range(3):
            if state[i * 3 + j] == -1:
                a += "2"
            else:
                a += str(state[i * 3 + j])
        print(a)
        a = ""


if __name__ == "__main__":
    state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    depth = 9
    cur_player = 1

    while win_check(state) == 0:
        print_game(state)
        while True:
            choice = int(input("\nWhich do you want to place? (0 ~ 8) : "))
            if 0 <= choice < 9 and state[choice] == 0:
                state[choice] = cur_player
                break
            else:
                print("You do wrong!!")

        if win_check(state) != 0:
            break

        cur_player *= -1

        node = Node(depth, cur_player, state)

        best_choice = -100
        best_value = -cur_player * maxsize

        for i in range(len(node.children)):
            child = node.children[i]
            value = MinMax(child, depth, -cur_player)
            if abs(cur_player * maxsize - value) <= abs(cur_player * maxsize - best_value):
                best_value = value
                best_choice = child.select

        state[best_choice] = cur_player
        print("computer choose " + str(best_choice + 1) + " with value " + str(best_value))
        cur_player *= -1





