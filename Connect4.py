import numpy as np
import random
import copy



############## Simple Connect4 game environment ##################

# The board is represented as an 2d array. If a position is 0 the position is empty.
# The players are represented as 1 or 2.
# A move is a number between 0 and 6 representing each column.




class Game :
    def __init__(self, player):
        self.state = np.zeros((6,7))
        self.player = player

    def new_game(self, player):
        self.state = np.zeros((6,7))
        self.player = player


    # Function to print the board if one wants to see what is happening or use it for manuallay play 
    def print_board(self) :
        print('0  1  2  3  4  5  6')
        print('-------------------', end ='')        
        for i in self.state :
            print('')
            for j in i :
                if j == 0 :
                    print('-', ' ', end = '')
                elif j == 1:
                    print('X', ' ', end = '')
                else :
                    print('O', ' ', end = '')


    # Check if a m ove is valid
    def valid_move(self, move) :
        if self.state[0][move] == 0 :
            return True
        else :
            return False 

    # Make a move, if the column of the move is full it does nothing, if move is larger than 6 it crashes. Easy to make the validation here if one wants to.   
    def make_move(self, move) :
        done = False
        for i in range(5, -1, -1) :
            if self.state[i][move] == 0 and not done :      
                self.state[i][move] = self.player
                if self.player == 1 :
                    self.player = 2
                else :
                    self.player = 1
                done = True

    # get all valid moves
    def get_valid_moves(self) :
        vm = []
        for i in range (0, 7) :
            if self.valid_move(i) :
                vm.append(i)
        return vm

    # Check if a position as a winning one, returns True if player has won, False otherwise.     
    def winning_position (self, player) :                    
        # Check rows
        for i in range(5, -1, -1) :
            r = self.state[i]
            for j in range (0, 4) :
                if r[j] != 0 and r[j] == player and r[j] == r[j+1] and r[j+1] == r[j+2] and r[j+2] == r[j+3] :
                    return True       
        # Check columns
        for i in range (0, 7) :
            for j in range(5, 2, -1) :
                if self.state[j][i] !=0 and self.state[j][i] == player and self.state[j][i] == self.state[j-1][i] and self.state[j-1][i] == self.state[j-2][i] and self.state[j-2][i] == self.state[j-3][i] :
                    return True
        # Check diagonals
        for i in range(5, 2, -1) :
            # Diagonals to the right
            for j in range(0, 4) :
                if self.state[i][j] != 0 and self.state[i][j] == player and self.state[i][j] == self.state[i-1][j+1] and self.state[i-1][j+1] == self.state[i-2][j+2] and self.state[i-2][j+2] == self.state[i-3][j+3] :
                    return True
            # Diagonals to the left
            for j in range(6, 2, -1) :
                if self.state[i][j] != 0 and self.state[i][j]== player and self.state[i][j] == self.state[i-1][j-1] and self.state[i-1][j-1] == self.state[i-2][j-2] and self.state[i-2][j-2] == self.state[i-3][j-3] :
                    return True  
        return False

    # Check if Tie, returns True if it is a Tie, False otherwise 
    def is_tie(self):
        if np.all((self.state[0] != 0)):
            return True
        else:
            return False


class randomagent:
    def __init__(self, game):
        self.game = game
        self.num_of_games = 0
    
    def choose_move(self):               
        action = random.choice(self.game.get_valid_moves())
        return int(action)

#Instanciate a game
game = Game(random.choice([1,2]))

game.print_board()

# Instanciate the agent
agent1 = randomagent(game)



# A simple gameloop, an agent plays as player 1 and a human as player 2 
while True:
    if game.player == 1 :
        move = agent1.choose_move()
        game.make_move(move)
        print('')
        game.print_board()
        print('')
        game.player = 2
    elif game.player == 2 :
        move = int(input('Make your move, (0-6): '))
        game.make_move(move)
        print('')
        game.print_board()
        print('')
        game.player = 1 

    if game.winning_position(1) :
        print('The agent won')
        break

    if game.winning_position(2) :
        print('You won')
        break

    if game.is_tie():
        print('It was a tie')
        break
