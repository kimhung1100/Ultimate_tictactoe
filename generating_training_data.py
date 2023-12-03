from state import *
from minimax import minimax
import copy
import numpy as np

def getBestMoveRes(board):
    bestMove = None
    bestVal = 1000000000
    if(board.player_to_move == board.X):
        bestVal = -1000000000
    for m in board.get_valid_moves:
        tmp = State(board)        
        tmp.act_move(m)        
        mVal = minimax(tmp, 30, tmp.player_to_move == board.X)
        
        if(board.player_to_move == board.X and mVal > bestVal):
            bestVal = mVal
            bestMove = m
        if(board.player_to_move == board.O and mVal < bestVal):
            bestVal = mVal
            bestMove = m
    return bestMove, bestVal

positions = []
moveProbs = []
outcomes = []

terminals = []

def visitNodes(board):
    term = board.game_over
    if(term):
        terminals.append(1)
        return
    else:
        bestMove, bestVal = getBestMoveRes(board)
        positions.append(board.toNetworkInput())
        moveProb = [ 0 for x in range(0,28) ]
        idx = board.getNetworkOutputIndex(bestMove)
        moveProb[idx] = 1
        moveProbs.append(moveProb)
        if(bestVal > 0):
            outcomes.append(1)
        if(bestVal == 0):
            outcomes.append(0)
        if(bestVal < 0):
            outcomes.append(-1)
        for m in board.generateMoves():
            next = copy.deepcopy(board)
            next.applyMove(m)
            visitNodes(next)

board = State_2()
# board.setStartingPosition()
visitNodes(board)

np.save("positions", np.array(positions))
np.save("moveprobs", np.array(moveProbs))
np.save("outcomes", np.array(outcomes))
