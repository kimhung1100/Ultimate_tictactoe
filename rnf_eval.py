import keras
from state import *
import random
import numpy as np

from support import  toNetworkInput, getNetworkOutputIndex
model = keras.models.load_model("model_it10.keras")
#model = keras.models.load_model("common/random_model.keras")

X = 1
O = -1
#
# def fst(a):
#     return a[0]

# white == random player
# black == network
def net_vs_rand(board):
    record = []
    while(not board.game_over):
        if(board.player_to_move == X):
            moves = board.get_valid_moves
            m = moves[random.randint(0, len(moves)-1)]
            board.act_move(m)
            record.append(m)
            continue
        else:
            q = model.predict(np.array([toNetworkInput(board)]))
            masked_output = [ 0 for x in range(0,28)]
            for m in board.get_valid_moves:
                m_idx = getNetworkOutputIndex(m)
                masked_output[m_idx] = q[0][0][m_idx]
            best_idx = np.argmax(masked_output)
            sel_move = None
            for m in board.generateMoves():
                m_idx = getNetworkOutputIndex(m)
                if(best_idx == m_idx):
                    sel_move = m
            board.act_move(sel_move)
            record.append(sel_move)
            continue
    terminal, winner = board.game_over, board.game_result(board.global_cells)

    return winner

# white random player
# black random player
def rand_vs_rand(board):
    while(not board.game_over):
        moves = board.get_valid_moves

        m = moves[random.randint(0, len(moves)-1)]
        board.act_move(m)
        continue
    terminal, winner = board.game_over, board.game_result(board.global_cells)
    return winner


whiteWins = 0
blackWins = 0

for i in range(0,100):
    board = State_2()
    moves = board.get_valid_moves
    m = moves[random.randint(0, len(moves)-1)]
    board.act_move(m)
    winner = net_vs_rand(board)
    if(winner == X):
        whiteWins += 1
    if(winner == O):
        blackWins += 1

all = whiteWins + blackWins
print("Rand Network vs Reinforcement: "+str(whiteWins/all) + "/"+str(blackWins/all))


whiteWins = 0
blackWins = 0

for i in range(0,100):
    board = State_2()
    moves = board.get_valid_moves
    m = moves[random.randint(0, len(moves)-1)]
    board.applyMove(m)
    winner = rand_vs_rand(board)
    if(winner == X):
        whiteWins += 1
    if(winner == O):
        blackWins += 1

all = whiteWins + blackWins
print("Rand vs Rand Network: "+str(whiteWins/all) + "/"+str(blackWins/all))