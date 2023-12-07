import copy
from state import *
from misc import *


def minimax(board, depth, alpha, beta, maximize):
    isTerminal, winner = board.game_over, board.game_result
    if depth == 0 or isTerminal:
        if winner == board.X:
            return 1000
        elif winner == board.O:
            return -1000
        else:
            return 0
    moves = board.get_valid_moves
    print(moves)

    if maximize:
        bestVal = -999999999999
        for move in moves:
            if not board.is_valid_move(move):
                continue
            next_board = State_2(board)
            next_board.act_move(move)
            # print(next_board)
            bestVal = max(bestVal, minimax(next_board, depth - 1, alpha, beta, not maximize))
            alpha = max(alpha, bestVal)
            if beta <= alpha:
                break
        return bestVal
    else:
        bestVal = 9999999999999
        for move in moves:
            if not board.is_valid_move(move):
                continue
            next_board = State_2(board)
            next_board.act_move(move)
            # print(next_board)
            bestVal = min(bestVal, minimax(next_board, depth - 1, alpha, beta, not maximize))
            beta = min(beta, bestVal)
            if beta <= alpha:
                break
        return bestVal

# Assuming you have an instance of the State class
# if name == '__main__':
# initial_state = State_2()

# # Display the initial state
# print("Initial State:")
# print(initial_state)

# # Test minimax with maximizing player (X)
# maximize_result = minimax(initial_state, depth=5, alpha=-float('inf'), beta=float('inf'), maximize=True)
# print("Minimax Result (Maximizing Player):", maximize_result)

# # Test minimax with minimizing player (O)
# minimize_result = minimax(initial_state, depth=5, alpha=-float('inf'), beta=float('inf'), maximize=False)
# print("Minimax Result (Minimizing Player):", minimize_result)
