from state import *
def evaluate_board(board):
    weights = [
        [3, 2, 3],
        [2, 4, 2],
        [3, 2, 3]
    ]
    total_score = 0
    for block_index in range(len(board.blocks)):
        grid = board.blocks[block_index]
        for i in range(3):
            for j in range(3):
                if grid[i, j] == board.X:
                    total_score += weights[i][j]
                elif grid[i, j] == board.O:
                    total_score -= weights[i][j]
    return total_score

import copy

def minimax(board, depth, alpha, beta, maximize):
    is_terminal, winner = board.game_over, board.game_result
    if depth == 0 or is_terminal:
        if winner == board.X:
            return 1000 + evaluate_board(board)
        elif winner == board.O:
            return -1000 - evaluate_board(board)
        else:
            return evaluate_board(board)
    moves = board.get_valid_moves

    if maximize:
        best_val = -float('inf')
        for move in moves:
            if not board.is_valid_move(move):
                continue
            next_board = copy.deepcopy(board)  # Create a deep copy of the board
            next_board.act_move(move)
            best_val = max(best_val, minimax(next_board, depth - 1, alpha, beta, not maximize))
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        return best_val
    else:
        best_val = float('inf')
        for move in moves:
            if not board.is_valid_move(move):
                continue
            next_board = copy.deepcopy(board)  # Create a deep copy of the board
            next_board.act_move(move)
            best_val = min(best_val, minimax(next_board, depth - 1, alpha, beta, not maximize))
            beta = min(beta, best_val)
            if beta <= alpha:
                break
        return best_val

# Assuming you have an instance of the State class
# if __name__ == '__main__':
#     initial_state = State_2()

#     # Display the initial state
#     print("Initial State:")
#     print(initial_state)

#     # Test minimax with maximizing player (X)
#     maximize_result = minimax(initial_state, depth=5, alpha=-float('inf'), beta=float('inf'), maximize=True)
#     print("Minimax Result (Maximizing Player):", maximize_result)

#     # Test minimax with minimizing player (O)
#     minimize_result = minimax(initial_state, depth=5, alpha=-float('inf'), beta=float('inf'), maximize=False)
#     print("Minimax Result (Minimizing Player):", minimize_result)
