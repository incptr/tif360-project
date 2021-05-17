import random
from functions import *
import numpy as np

class Leftmost:
    def __call__(self, obs):
        valid_moves = get_valid_moves(obs.board)
        return valid_moves[0]


class Random:
    def __init__(self, valid=True):
        self.valid = valid

    def __call__(self, obs):
        if self.valid:
            valid_moves = get_valid_moves(obs.board)
            return random.choice(valid_moves)
        else:
            columns = [i for i in range(obs.cols)]
            return random.choice(columns)


class One_step_ahead:
    def __call__(self, obs):
        columns = get_winning_moves(obs.board, obs.x, obs.mark)
        if columns:
            return random.choice(columns)
        columns = get_blocking_moves(obs.board, obs.x, obs.mark)
        if columns:
            return random.choice(columns)
        columns = get_valid_moves(obs.board)
        return random.choice(columns)


class N_steps_look_ahead_agent:
    def __init__(self, max_steps=3):
        self.max_steps = max_steps

    def __call__(self, obs):
        columns = get_winning_moves(obs.board, obs.x, obs.mark)
        if columns:  # if there is a winning move
            return random.choice(columns)
        columns = get_blocking_moves(obs.board, obs.x, obs.mark)
        if columns:  # if there is a blocking move
            return random.choice(columns)
        columns = n_look_ahead(board=obs.board, mark=obs.mark, max_steps=self.max_steps, x=obs.x)
        if columns:  # if there are max_steps ahead (i.e. step t+max is calculable). this wont be true when close to a draw
            return random.choice(columns)
        valid_moves = get_valid_moves()  # if all of the above fails. only gets called when close to a draw
        return random.choice(valid_moves)


class AB_agent():
    def __init__(self, max_steps=3):
        self.max_steps = max_steps

    def __call__(self, obs):
        winning_moves = get_winning_moves(obs.board, obs.x, obs.mark)
        if winning_moves:  # if there is a winning move
            return random.choice(winning_moves)
        blocking_moves = get_blocking_moves(obs.board, obs.x, obs.mark)
        if blocking_moves:  # if there is a blocking move
            return random.choice(blocking_moves)
        enabling_moves = get_enabling_moves(obs.board, obs.x, obs.mark)
        columns = alpha_beta(board=obs.board, mark=obs.mark, max_steps=self.max_steps, x=obs.x)
        if columns:  # if there are max_steps ahead (i.e. step t+max is calculable). this wont be true when close to a draw
            revised_columns = [i for i in columns if i not in enabling_moves]
            if revised_columns:  # if there are non-enabling moves from alpha-beta pruning suggestions
                return random.choice(revised_columns)
        valid_moves = get_valid_moves()  # if all of the above fails. only should get called when close to a draw
        revised_valid_moves = [i for i in valid_moves if i not in enabling_moves]
        if revised_valid_moves:  # if there are non-enabling moves
            return random.choice(revised_valid_moves)
        return random.choice(valid_moves)
        
        
class Alphabeta_agent():
    def __init(self):    
        print("Step {}: AphaBeta Agent moving".format(obs.step))


    def __call__(self,obs):

        # Get list of valid moves
        valid_moves = get_valid_moves(obs.board)#[c for c in range(obs.cols) if obs.board[c] == 0]
        # Convert the board to a 2D grid
        grid = np.asarray(obs.board).reshape(obs.rows, obs.cols)

        board_array = np.ndarray.tolist(obs.board)

        n_steps = 4 if board_array.count(0)>len(board_array)*2/3 else 5 if board_array.count(0)>len(board_array)/3 else 6
        # Use the heuristic to assign a score to each possible board in the next step
        scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, n_steps,obs) for col in valid_moves]))
        print("Scores:", scores, end=' - ')
        # Get a list of columns (moves) that maximize the heuristic
        max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
        # Select at random from the maximizing columns
        move = random.choice(max_cols)
        print("Selected move:", move)
        return move



