import numpy as np
import random



############## Simple Connect4 game environment ##################

# The board is represented as an 2d array. If a position is 0 the position is empty.
# The players are represented as 1 or 2.
# A move is a number between 0 and 6 representing each column.




class Game :
    def __init__(self, player):
        self.player = player
        self.num_of_rows = 6
        self.num_of_columns = 7
        self.in_row = 4
        self.board = np.zeros((self.num_of_rows,self.num_of_columns))
        self.terminal_state = False
        self.winner = -1
        self.reward = 0

    def new_game(self, player):
        self.board = np.zeros((self.num_of_rows,self.num_of_columns))
        self.player = player
        self.terminal_state = False
        self.winner = -1


    # Function to print the board if one wants to see what is happening or use it for manuallay play only works for the standard setup 
    def print_board(self) :
        print('0  1  2  3  4  5  6')
        print('-------------------', end ='')        
        for i in self.board :
            print('')
            for j in i :
                if j == 0 :
                    print('-', ' ', end = '')
                elif j == 1:
                    print('X', ' ', end = '')
                else :
                    print('O', ' ', end = '')


    # Check if a move is valid
    def valid_move(self, move) :
        if self.board[0][move] == 0 :
            return True
        else :
            return False 

    # Make a move, if the column of the move is full it does nothing, if move is larger than 6 it crashes. Easy to make the validation here if one wants to.   
    def make_move(self, move) :
        for i in range(self.num_of_rows-1, -1, -1) :
            if self.board[i][move] == 0:      
                self.board[i][move] = self.player
                if self.check_win():
                    self.terminal_state = True
                    self.winner = self.player
                    self.reward = 1
                elif self.is_tie():
                    self.terminal_state = True
                    self.winner = 0
                    self.reward = 0
                else:
                    if self.player == 1 :
                        self.player = 2
                    else :
                        self.player = 1
                break

    # get all valid moves
    def get_valid_moves(self) :
        vm = []
        for i in range (0, self.num_of_columns) :
            if self.valid_move(i) :
                vm.append(i)
        return vm

    # Check if a position as a winning one, returns True if player has won, False otherwise.     

    def check_win(self):
        # horizontal
        for row in range(self.num_of_rows):
            for col in range(self.num_of_columns - self.in_row + 1):
                window = self.board[row, col:col + self.in_row]
                if np.count_nonzero(window == self.player) == self.in_row:
                    return True
        # vertical
        for row in range(self.num_of_rows - self.in_row + 1):
            for col in range(self.num_of_columns):
                window = self.board[row:row + self.in_row, col]
                if np.count_nonzero(window == self.player) == self.in_row:
                    return True
        # diagonal
        for row in range(self.num_of_rows - self.in_row + 1):
            for col in range(self.num_of_columns - self.in_row + 1):
                window = self.board[range(row, row + self.in_row), range(col, col + self.in_row)]
                if np.count_nonzero(window == self.player) == self.in_row:
                    return True
        # off-diagonal
        for row in range(self.in_row - 1, self.num_of_rows):
            for col in range(self.num_of_columns - self.in_row + 1):
                window = self.board[range(row, row - self.in_row, -1), range(col, col + self.in_row)]
                if np.count_nonzero(window == self.player) == self.in_row:
                    return True
        return False


    # Check if Tie, returns True if it is a Tie, False otherwise 
    def is_tie(self):
        if np.all((self.board[0] != 0)):
            return True
        else:
            return False

"""
class randomagent:
    def __init__(self, game):
        self.game = game
        self.num_of_games = 0
    
    def choose_move(self):               
        move = random.choice(self.game.get_valid_moves())
        return int(move)

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
    elif game.player == 2 :
        move = int(input('Make your move, (0-6): '))
        game.make_move(move)
        print('')
        game.print_board()
        print('')

    if game.terminal_state:
        if game.winner == 0:
            print('It was a tie')
        else:
            print(f'Player {game.winner} won')
        print('***** New game *****')
        game.new_game(random.choice([1,2]))
        game.print_board()
"""