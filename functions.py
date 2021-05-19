import numpy as np

def get_valid_moves(board):
# returns a list of columns that are not yet full
    valid_moves = []
    cols = board.shape[1]
    for col in range(cols):
        if board[0, col] == 0:
            valid_moves.append(col)
    return valid_moves

def get_next_board(board, col, mark):
# returns the state of the next board if player mark drops a piece in column col
    next_board = board.copy()
    rows = len(next_board)
    for row in range(rows - 1, -1, -1):
        if next_board[row, col] == 0:
            break
    next_board[row, col] = mark
    return next_board

def count_window(board, window_size, num_discs, mark):
# for the entire game board, checks horizontal, vertical, and diagonal windows of window_size tiles.
# returns the number of total windows in a board that have num_discs pieces belonging to player 'mark' and the rest of the window is empty
    def check_window(window, window_size, num_discs, mark):
        # slightly more efficient "and" check of both conditions
        if not ((window == mark).sum() == num_discs):  # condition 1: number of discs in the window belong to the player
            return False
        return ((window == 0).sum() == (window_size - num_discs))  # condition 2: the rest of the tiles in the window are empty
    rows, cols = board.shape
    count = 0
    # horizontal
    for row in range(rows):
        for col in range(cols - window_size + 1):
            window = board[row, col:col + window_size]
            if check_window(window, window_size, num_discs, mark):
                count += 1
    # vertical
    for row in range(rows - window_size + 1):
        for col in range(cols):
            window = board[row:row + window_size, col]
            if check_window(window, window_size, num_discs, mark):
                count += 1
    # diagonal
    for row in range(rows - window_size + 1):
        for col in range(cols - window_size + 1):
            window = board[range(row, row + window_size), range(col, col + window_size)]
            if check_window(window, window_size, num_discs, mark):
                count += 1
    # off-diagonal
    for row in range(window_size - 1, rows):
        for col in range(cols - window_size + 1):
            window = board[range(row, row - window_size, -1), range(col, col + window_size)]
            if check_window(window, window_size, num_discs, mark):
                count += 1
    return count

def get_winning_moves(board, inarow, mark):
# returns a list of columns that will result in a game win if they exist
    valid_moves = get_valid_moves(board=board)
    winning_moves = []
    for col in valid_moves:
        next_board = get_next_board(board=board, col=col, mark=mark)
        if count_window(board=next_board, window_size=inarow, num_discs=inarow, mark=mark):
            winning_moves.append(col)
    return winning_moves

def get_blocking_moves(board, inarow, mark):
# returns a list of columns that block an opponent's imminent win in the next turn
    valid_moves = get_valid_moves(board=board)
    blocking_moves = []
    opp = 3 - mark  # opponents mark. 1 --> 2, and 2 --> 1
    for col in valid_moves:
        next_board = get_next_board(board=board, col=col, mark=opp)
        if count_window(board=next_board, window_size=inarow, num_discs=inarow, mark=opp):
            blocking_moves.append(col)
    return blocking_moves

def get_enabling_moves(board, inarow, mark):  # this might make get_blocking_moves redundant
# returns a list of columns that enable the opponent to make a winning move in the next turn
# the elements in this list later get removed from the list of potential valid moves, to avoid enabling an opponent win
    valid_moves = get_valid_moves(board=board)
    enabling_moves = []
    opp = 3 - mark  # opponents mark. 1 --> 2, and 2 --> 1
    for col in valid_moves:
        next_board = get_next_board(board=board, col=col, mark=mark)
        if get_winning_moves(board=next_board, inarow=inarow, mark=opp):
            enabling_moves.append(col)
    return enabling_moves

def evaluate_board(board, mark, inarow):
# assigns a numerical score to a board, depending on the number of beneficial windows to each player
# beneficial moves to the player get a positive score
# beneficial moves to the opponent get a negative score
# the magnitude of the score of each individual window depends on its urgency
    score = 0
    score -= 1e6 * count_window(board=board, window_size=inarow, num_discs=inarow, mark=3 - mark)
    score += 1e4 * count_window(board=board, window_size=inarow, num_discs=inarow, mark=mark)
    score -= 1e2 * count_window(board=board, window_size=inarow, num_discs=inarow - 1, mark=3 - mark)
    score += 1e0 * count_window(board=board, window_size=inarow, num_discs=inarow - 1, mark=mark)
    return score

def n_look_ahead(board, mark, max_steps, inarow, step=1, player=None):
    if player == None:
        player = mark  # initializes the player according to the mark at step 1
    valid_moves = get_valid_moves(board)
    if step == max_steps:
        scores = []
        for col in valid_moves:
            next_board = get_next_board(board=board, col=col, mark=mark)
            score = evaluate_board(board=next_board, mark=player, inarow=inarow)
            scores.append(score)
        return max(scores)  # return the col that corresponds to the max(scores)
    else:
        next_mark = 3 - mark
        next_step = step + 1
        scores = []
        for col in valid_moves:
            next_board = get_next_board(board=board, col=col, mark=mark)
            score = n_look_ahead(board=next_board, mark=next_mark, max_steps=max_steps, inarow=inarow, step=next_step, player=player)
            scores.append(score)
        if step == 1:
            max_indices = [index for index, val in enumerate(scores) if val == max(scores)]
            return [valid_moves[i] for i in max_indices]  # returns list of columns with max score
        elif mark == player:
            return max(scores)
        else:
            return min(scores)

def alpha_beta(board, mark, max_steps, inarow, step=0, player=None, alpha=float('-inf'), beta=float('inf')):
    if player == None:
        player = mark
    if step == max_steps or get_terminal_state(board=board, inarow=inarow):
        return evaluate_board(board=board, mark=player, inarow=inarow)  # evaluate board from the player's perspective
    valid_moves = get_valid_moves(board=board)
    scores = []
    for col in valid_moves:
        next_board = get_next_board(board=board, col=col, mark=mark)
        score = alpha_beta(board=next_board, mark=3-mark, max_steps=max_steps, inarow=inarow, step=step + 1, player=player, alpha=alpha, beta=beta)
        scores.append(score)
        if mark == player:  # if maximizing agent
            alpha = max(alpha, score)
        else:  # if minimizing agent
            beta = min(beta, score)
        if alpha >= beta:
            break
    if scores:  # if not empty list
        if mark == player:
            next_score = max(scores)
        else:
            next_score = min(scores)
    else:  # if empty list (happens when close to a draw)
        next_score = 0
    if step == 0:  # if origin state: return the column that leads to the best score
        return [valid_moves[i] for i in [j for j, s in enumerate(scores) if s == next_score]]
    return next_score

def print_summary(scores):
    win1 = scores.count([1, 0])
    win2 = scores.count([0, 1])
    inv1 = scores.count([None, 0])
    inv2 = scores.count([0, None])
    draw = scores.count([0, 0])
    tot = len(scores)
    other = tot - win1 - win2 - inv1 - inv2 - draw
    print('\nSummary\n' + '=' * 30)
    print(f'Player 1 wins: {win1}')
    print(f'Player 2 wins: {win2}')
    print(f'Player 1 invalid moves: {inv1}')
    print(f'Player 2 invalid moves: {inv2}')
    print(f'Draws: {draw}')
    print(f'Other: {other}')
    print(f'Total games played: {tot}')

def print_grid(grid):
    def get_char(val):
        chars = [' ', 'X', 'O']
        return chars[val]
    rows, cols = grid.shape
    indent = 10
    # column headers
    print('\n'+' '*indent, end='')
    for col in range(cols):
        print(f'  {col+1} ', end='')
    # top border
    print('\n'+' '*indent, end='')
    print('-'*(4*cols+1), end='')
    # board
    for row in range(rows):
        print('\n'+' ' * indent, end='')
        for col in range(cols):
            piece = get_char(grid[row, col])
            print(f'| {piece} ', end='')
        print('|', end='')
    # bottom border
    print('\n'+' '*indent, end='')
    print(f'{"-" * (4 * cols + 1)}')

def get_terminal_state(board, inarow):
    if (count_window(board=board, window_size=inarow, num_discs=inarow, mark=1) > 0):  # condition 1: player 1 win
        return True
    if (count_window(board=board, window_size=inarow, num_discs=inarow, mark=2) > 0):  # condition2: player 2 win
        return True
    if board[0].all():  # condition 3: draw
        return True
    
    
    
# ALPHABETA

# Gets board at next step if agent drops piece in selected column
def drop_piece(grid, col, mark):
    board_arrary = np.ndarray.tolist(grid)
    next_grid = grid.copy()
    for row in range(len(grid[:,0])-1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return next_grid

# Get the number of pieces of the same mark in a window
def pieces_in_window(window, piece,obs):
    return window.count(piece) * (window.count(piece) + window.count(0) == obs.inarow)

# Counts number of pieces for both players for every possible window
def count_windows(grid,obs):
    windows = {piece: [0 for i in range(obs.inarow+1)] for piece in [1, 2]}
    
    # horizontal
    for row in range(obs.rows):
        for col in range(obs.cols-(obs.inarow-1)):
            window = list(grid[row, col:col+obs.inarow])
            windows[1][pieces_in_window(window, 1,obs)]+=1
            windows[2][pieces_in_window(window, 2,obs)]+=1

    # vertical
    for row in range(obs.rows-(obs.inarow-1)):
        for col in range(obs.cols):
            window = list(grid[row:row+obs.inarow, col])
            windows[1][pieces_in_window(window, 1,obs)]+=1
            windows[2][pieces_in_window(window, 2,obs)]+=1

    # positive diagonal
    for row in range(obs.rows-(obs.inarow-1)):
        for col in range(obs.cols-(obs.inarow-1)):
            window = list(grid[range(row, row+obs.inarow), range(col, col+obs.inarow)])
            windows[1][pieces_in_window(window, 1,obs)]+=1
            windows[2][pieces_in_window(window, 2,obs)]+=1

    # negative diagonal
    for row in range(obs.inarow-1, obs.rows):
        for col in range(obs.cols-(obs.inarow-1)):
            window = list(grid[range(row, row-obs.inarow, -1), range(col, col+obs.inarow)])
            windows[1][pieces_in_window(window, 1,obs)]+=1
            windows[2][pieces_in_window(window, 2,obs)]+=1
    return windows
            
# Calculates value of heuristic for grid
def get_heuristic(grid, mark,obs):
    windows=count_windows(grid,obs)
    score =  windows[mark][1] + windows[mark][2]*3 + windows[mark][3]*9 + windows[mark][4]*81 - windows[mark%2+1][1] - windows[mark%2+1][2]*3 - windows[mark%2+1][3]*9 - windows[mark%2+1][4]*81
    return score

# Uses alphabeta to calculate value of dropping piece in selected column
def score_move(grid, col, mark, nsteps,obs):
    next_grid = drop_piece(grid, col, mark)
    score = alphabeta(next_grid, nsteps-1, -np.Inf, np.Inf, False, mark,obs)
    return score

# Checks if game has ended
def is_terminal_node(grid,obs):
    windows=count_windows(grid,obs)
    return windows[1][obs.inarow] + windows[2][obs.inarow] > 0

# Alpha Beta pruning implementation
def alphabeta(node, depth, a, b, maximizingPlayer, mark,obs):
    is_terminal = is_terminal_node(node,obs)
    valid_moves = [c for c in range(obs.cols) if node[0][c] == 0]
    if depth == 0 or is_terminal:
        return get_heuristic(node, mark,obs)
    if maximizingPlayer:
        value = -np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark)
            value = max(value, alphabeta(child, depth-1, a, b, False, mark,obs))
            a = max(a, value)
            if a >= b:
                break # β cutoff
        return value
    else:
        value = np.Inf
        for col in valid_moves:
            child = drop_piece(node, col, mark%2+1)
            value = min(value, alphabeta(child, depth-1, a, b, True, mark,obs))
            b = min(b, value)
            if b <= a:
                break # α cutoff
        return value