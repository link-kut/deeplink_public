# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai
# http://mcts.ai/about/index.html

from math import *
import random


def upper_confidence_bounds(node_value, num_parent_visits, num_node_visits):
    """ the UCB1 formula """
    return node_value + sqrt(2 * log(num_parent_visits) / num_node_visits)


class OXOEnv:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """
    def __init__(self):
        self.current_player = 1  # At the root pretend the current player 'Player 1'
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0 = empty, 1 = player 1, 2 = player 2

    def clone(self):
        """ Create a deep clone of this game state.
        """
        env = OXOEnv()
        env.current_player = self.current_player
        env.board = self.board[:]
        return env

    def do_move(self, location):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert 8 >= location >= 0 == self.board[location] and location == int(location)

        self.board[location] = self.current_player

        if self.current_player == 1:
            self.current_player = 2
        elif self.current_player == 2:
            self.current_player = 1
        else:
            assert False

    def get_possible_locations(self):
        """ Get all possible moves from this state.
        """
        return [i for i in range(9) if self.board[i] == 0]

    def get_result(self, player_just_moved):
        """ Get the game result from the viewpoint of playerjm.
            Case == 1.0: player 1 win,
            Case == 2.0: player 2 win,
            Case == 3.0: both player 1 and 2 win (draw)
        """
        case = 0.0
        for (x, y, z) in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            if self.board[x] == self.board[y] == self.board[z] == player_just_moved:
                case += 1
            elif self.board[x] == self.board[y] == self.board[z] == (3 - player_just_moved):
                case += 2

        if case == 1:
            return 1        # player 1 win
        elif case == 2:
            return 2        # player 2 win
        elif case == 3 or not self.get_possible_locations():
            print("draw - ", case)
            return 0        # draw
        else:
            return -1.0     # continue

        assert False  # Should not be possible to get here

    def __repr__(self):
        s = ""
        for i in range(9):
            s += ".XO"[self.board[i]]
            if i % 3 == 2:
                s += "\n"
        return s[0:-1]


class Node:
    """ A node in the game tree.
        Note wins is always from the viewpoint of current_player.
        Crashes if state not specified.
    """

    def __init__(self, location=None, parent=None, env=None):
        self.location = location    # "None" for the root node
        self.parent_node = parent   # "None" for the root node
        self.child_nodes = []
        self.wins = 0
        self.visits = 0
        self.unvisited_locations = env.get_possible_locations()  # future child nodes

    def uct_select_child(self):
        """ Use the UCB1 formula to select a child node.
            Often a constant UCTK is applied so we have UCB1 to vary the amount of exploration versus exploitation.
        """
        s = sorted(
            self.child_nodes,
            key=lambda c: upper_confidence_bounds(c.wins / c.visits,  self.visits, c.visits)
        )[-1]
        return s

    def add_child(self, loc, env):
        """ Remove loc from untried_locations and add a new child node for the location loc.
            Return the added child node
        """
        n = Node(location=loc, parent=self, env=env)
        self.unvisited_locations.remove(loc)
        self.child_nodes.append(n)
        return n

    def update(self, result):
        """ Update this node - one additional visit and result additional wins.
            result must be from the viewpoint of current_player.
        """
        self.visits += 1
        self.wins += result

    def to_tree_string(self, indent):
        s = self.indent_string(indent) + str(self)
        for c in self.child_nodes:
            s += c.to_tree_string(indent + 1)
        return s

    @staticmethod
    def indent_string(indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def to_children_string(self):
        s = ""
        for c in self.child_nodes:
            s += str(c) + "\n"
        return s

    def __repr__(self):
        return "[Location: {0}, W/V: {1}/{2}, U: {3}, Child Nodes: {4}]".format(
            self.location,
            self.wins,
            self.visits,
            str(self.unvisited_locations),
            str([x.location for x in self.child_nodes])
        )


def search_by_uct(env, iter_max, verbose=False):
    """ Conduct a UCT (Upper Confidence Bounds for Trees) search for itermax iterations starting from rootstate.
        Return the best move from the root_node.
        Assumes 2 alternating players (player 1 starts), with game results in the range [1, 2, 0, -1].
    """

    root_node = Node(location=None, parent=None, env=env)

    print("[Search By UCT]")
    for i in range(iter_max):
        node = root_node
        env2 = env.clone()

        # Select
        while node.unvisited_locations == [] and node.child_nodes != []:  # node is fully expanded and non-terminal
            node = node.uct_select_child()
            env2.do_move(node.location)
            print("Iter: {0}, Player {1} selects the best child node {2}".format(
                i,
                env2.current_player,
                node.location
            ))

        # Expand
        if node.unvisited_locations:  # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.unvisited_locations)
            env2.do_move(m)
            print("Iter: {0}, Player {1} expands to an arbitrary location {2}".format(
                i,
                env2.current_player,
                m
            ))
            node = node.add_child(m, env2)  # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        j = 0
        while env2.get_possible_locations():  # while state is non-terminal
            m = random.choice(env2.get_possible_locations())
            env2.do_move(m)
            print("Iter: {0} and {1}, Player {2} rolls out to the location {3}".format(
                i,
                j,
                env2.current_player,
                m
            ))
            j += 1

        # Backpropagate
        j = 0
        print("{0} - Cloned Env:\n{1}".format(3 - env.current_player, env2))
        while node:  # backpropagate from the expanded node and work back to the root node
            node.update(env2.get_result(3 - env.current_player))  # state is terminal. Update node with result from
            # point of view of 3 - env.current_player
            print("Iter: {0}, {1}, Evaluate the node {2}: Wins/Visits - {3}/{4}".format(
                i,
                j,
                node.location,
                node.wins,
                node.visits
            ))
            node = node.parent_node
            j += 1
        print()

    # Output some information about the tree - can be omitted
    if verbose:
        print(root_node.to_tree_string(0))
    else:
        print(root_node.to_children_string())

    return sorted(root_node.child_nodes, key=lambda c: c.visits)[-1].location  # return the move that was most visited


def play_game(verbose=True):
    """ Play a sample game between two UCT players where each player gets a different number
        of UCT iterations (= simulations = tree nodes).
    """
    # state = OthelloState(4) # uncomment to play Othello on a square board of the given size
    env = OXOEnv() # uncomment to play OXO
    # state = NimState(15)  # uncomment to play Nim with the given number of starting chips

    while env.get_possible_locations():
        print("Original Env:\n{0}".format(env))
        if env.current_player == 1:
            m = search_by_uct(env=env, iter_max=2, verbose=verbose) # Player 1
        else:
            m = search_by_uct(env=env, iter_max=2, verbose=verbose) # Player 2
        print("Best Move: " + str(m) + "\n")
        env.do_move(m)

        print("Original Env:\n{0}".format(env))
        if env.get_result(env.current_player) == 1:
            print("Player " + str(env.current_player) + " wins!")
            break
        elif env.get_result(env.current_player) == 2:
            print("Player " + str(3 - env.current_player) + " wins!")
            break
        elif env.get_result(env.current_player) == 0:
            print("Nobody wins!")
            break
        elif env.get_result(env.current_player) == -1:
            print("Continue...\n")
        else:
            assert False


if __name__ == "__main__":
    play_game()