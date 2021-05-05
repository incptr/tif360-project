import numpy as np
import sys
from functions import get_valid_moves, print_summary, print_grid

class Obs:
    def __init__(self, rows=6, cols=7, x=4, agents=[None, None]):
        self.rows = rows
        self.cols = cols
        self.x = x
        self.agents = agents
        self.max_turns = rows * cols
        self.reset()

    def reset(self):
        self.board = np.zeros([self.rows, self.cols], dtype=int)
        self.turn = 1
        self.mark = 1
        self.agent_index = 0
        self.gameover = False
        self.score = None

    def drop(self, col):
        if not (0 <= col < self.cols):
            # print(f'Invalid move. Selected column ({col+1}) out of bounds.')
            return True
        for row in range(self.rows-1, -1, -1):
            if self.board[row, col] == 0:
                break
        if self.board[row, col] != 0:
            # print(f'Invalid move. Column {col+1} full.')
            return True
        self.board[row, col] = self.mark
        # print(f'Player {self.mark} picks column {col+1}.')
        return False

    def check_win(self):
        # horizontal
        for row in range(self.rows):
            for col in range(self.cols - self.x + 1):
                window = [val for val in self.board[row, col:col + self.x]]
                if window.count(self.mark) == self.x:
                    return True
        # vertical
        for row in range(self.rows - self.x + 1):
            for col in range(self.cols):
                window = [val for val in self.board[row:row + self.x, col]]
                if window.count(self.mark) == self.x:
                    return True
        # diagonal
        for row in range(self.rows - self.x + 1):
            for col in range(self.cols - self.x + 1):
                window = [val for val in self.board[range(row, row + self.x), range(col, col + self.x)]]
                if window.count(self.mark) == self.x:
                    return True
        # off-diagonal
        for row in range(self.x - 1, self.rows):
            for col in range(self.cols - self.x + 1):
                window = [val for val in self.board[range(row, row - self.x, -1), range(col, col + self.x)]]
                if window.count(self.mark) == self.x:
                    return True

    def get_human_player_col(self):
        # print_grid(self.board)
        while True:
            try:
                col = int(input(f"\nTurn {self.turn}. Player {self.mark}'s move. Select column (1-7): ")) - 1
            except:
                print(f'That is not an integer.')
            else:
                if 0 <= col < self.cols:
                    if col in get_valid_moves(self.board):
                        return col
                    else:
                        print(f'Invalid move. Column {col+1} full.')
                else:
                    print(f'Number must be between 1 and {self.cols}.')

    def next_turn(self):
        self.turn += 1
        self.mark = 3 - self.mark  # 1 --> 2 and 2 --> 1
        self.agent_index = self.mark // 2

    def check_draw(self):
        return self.board.all()

    def play_turn(self):
        print_grid(self.board)
        player = self.agents[self.agent_index]
        if player == None:
            col = self.get_human_player_col()
            self.drop(col)
        else:
            col = player(self)
            if self.drop(col):  # if invalid move
                # print(f'\nInvalid move by AI agent - Player {self.mark}.\nGAME OVER.\n')
                self.gameover = True
                if self.mark == 1:
                    self.score = [None, 0]
                else:
                    self.score = [0, None]
                return
        if self.check_win():
            # print(f'\nPlayer {self.mark} wins after {self.turn} turns.\nGAME OVER.')
            # print(f'\n{self.board}\n')
            self.gameover = True
            if self.mark == 1:
                self.score = [1, 0]
            else:
                self.score = [0, 1]
            return
        if self.check_draw():
            # print('\nDraw.\nGAME OVER.')
            # print(f'\n{self.board}\n')
            self.gameover = True
            self.score = [0, 0]
            return
        self.next_turn()

    def play_game(self):
        self.reset()
        while not self.gameover:
            self.play_turn()
        return self.score

    def play_n_games(self, num_games):
        game = 1
        scores = []
        while game <= num_games:
            score = self.play_game()
            scores.append(score)
            print_grid(self.board)
            if game % (num_games // 10) == 0:
                print(f'\nProgress: {100*game/num_games}%. {game}/{num_games} played.')
                print_summary(scores)
            game += 1
