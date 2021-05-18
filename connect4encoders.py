import numpy as np


class encoder:
    def __init__(self):
        #for future use
        pass
    """
    def encode_state(self, board, player):
        channel1 = np.reshape(np.where(board == 2, 0, board), (6,7,1))
        channel2_tmp = np.where(board == 1, 0, board)
        channel2 = np.reshape(np.where(channel2_tmp == 2, 1, channel2_tmp), (6,7,1))
        if player == 1:
            channel3 = np.ones((6,7,1))
        else: 
            channel3 = np.zeros((6,7,1))

        encoded_state = np.dstack((channel1, channel2, channel3))
        return encoded_state
    """

    def encode_state(self, board, player):
        if player == 1:
            return np.reshape(np.where(board == 2, -1, board), (6,7,1))
        if player == 2:
            board_tmp = np.where(board == 1, -1, board)
            return np.reshape(np.where(board_tmp == 2, 1, board_tmp), (6,7,1))
            