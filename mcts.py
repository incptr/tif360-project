import math
import numpy as np
import random
import time
import copy
import connect4encoders

class mcts_node :
    def __init__(self, game, nn, parent = None, p=0):
        self.parent = parent
        self.game = game
        self.nn = nn
        self.W = 0
        self.N = 0
        self.Q = 0
        self.P = p
        self.children = []


    def is_leaf (self):
        if self.children == []:
            return True
        else :
            return False

    def choose_child(self):
        moves = []
        for i in range(self.game.num_of_columns):
            if self.children[i] != None and self.children[i].N == 0:
                moves.append(i)
        return self.children[random.sample(moves, 1)[0]]



    def add_children (self):
        encoder = connect4encoders.encoder()
        policy, _  = self.nn(np.array([encoder.encode_state(self.game.board, self.game.player)]))
        for i in range(self.game.num_of_columns):
            if i in (self.game.get_valid_moves()):
                new_game = copy.deepcopy(self.game)
                new_game.make_move(i)
                new_child = mcts_node(new_game, self.nn, self, policy[0][i])
                self.children.append(new_child)
            else:
                self.children.append(None)

    def fully_expanded(self) :
        if self.children == [] :
                return False
        for child in self.children :
            if child != None and child.N == 0 :
                return False
        return True

    def uct_best_child (self, c=1) :
        val = -100000000
        best_child = None
        for i in range(self.game.num_of_columns):
            if self.children[i] != None:
                c_val = self.children[i].Q + c*self.children[i].P * (math.sqrt(self.N)/(1+self.children[i].N))
                if c_val > val :
                    val = c_val
                    best_child = self.children[i]           
        return best_child
    

    def back_prop (self, result) :
        self.N += 1
        self.W += result
        self.Q = self.W/self.N
        if self.parent != None :
            return self.parent.back_prop(result)
        else:
            return self 


    def get_policy(self):
        policy = []
        for i in range(7) :
            if self.children[i] != None:
                policy.append(self.children[i].N)
            else:
                policy.append(0)
        policy = np.array(policy)/self.N #Normalize
        return(policy)


def monte_carlo_tree_search(root) :
    thinktime = 1
    starttime = time.time()
    player =root.game.player
    while (time.time() < starttime + thinktime):
        while (root.fully_expanded() and not root.game.terminal_state):
            root = root.uct_best_child()

        if root.is_leaf() and not root.game.terminal_state:
            root.add_children()

        if not root.game.terminal_state: 
            root = root.choose_child()

        if root.game.terminal_state:
            if root.game.player == player:
                result = 1
            else:
                result = -1
        else:
            encoder = connect4encoders.encoder()
            _, v = root.nn(np.array([encoder.encode_state(root.game.board, root.game.player)]))
            result = -v
        root = root.back_prop(result)
    return root.get_policy()