import numpy as np
from state import *
from minimax_eval import *
import json

import numpy as np
X = 1
O = -1
def heur2(state):
    score = 0

    # Heuristic 2: Evaluate the state based on features of the given board
    for i in range(9):
        small_board = state.blocks[i]
        row_sum = np.sum(small_board, axis=1)
        col_sum = np.sum(small_board, axis=0)
        diag_sum_topleft = np.trace(small_board)
        diag_sum_topright = np.trace(np.fliplr(small_board))

        # Check for small board wins (5 points)
        X_win = (any(row_sum == 3) or any(col_sum == 3) or diag_sum_topleft == 3 or diag_sum_topright == 3)
        O_win = (any(row_sum == -3) or any(col_sum == -3) or diag_sum_topleft == -3 or diag_sum_topright == -3)

        score += X_win * 5 - O_win * 5

        # center board win
        if (i == 4):
            score += X_win * 10 - O_win * 10

        # corner board
        if (i == 0 or i == 2 or i == 6 or i == 8):
            score += X_win * 3 - O_win * 3

        # center of each square
        score += small_board[1, 1] * 3

        # any in center board add 3
        if (i == 4):
            for j in range(9):
                row, col = divmod(j, 3)  # Convert 1D index to 2D indices
                score += small_board[row, col] * 3

        # Check any 2 sequences in small_board for all cases
        for i in range(3):
            for j in range(3):
                # Check two sequences in a row for X
                if (small_board[i, j] == X) and (j + 1 < 3) and (small_board[i, j + 1] == X):
                    score += 2

                # Check two sequences in a column for X
                if (small_board[j, i] == X) and (j + 1 < 3) and (small_board[j + 1, i] == X):
                    score += 2

                # Check two sequences in a row for O
                if (small_board[i, j] == O) and (j + 1 < 3) and (small_board[i, j + 1] == O):
                    score -= 2

                # Check two sequences in a column for O
                if (small_board[j, i] == O) and (j + 1 < 3) and (small_board[j + 1, i] == O):
                    score -= 2

        # Check two sequences in the main diagonal for X
        if ((small_board[0, 0] == X) or (small_board[2, 2] == X)) and (small_board[1, 1] == X):
            score += 2

        # Check two sequences in the secondary diagonal for X
        if ((small_board[0, 2] == X) or (small_board[2, 0] == X)) and (small_board[1, 1] == X):
            score += 2

        # Check two sequences in the main diagonal for O
        if ((small_board[0, 0] == O) or (small_board[2, 2] == O)) and (small_board[1, 1] == O):
            score -= 2

        # Check two sequences in the secondary diagonal for O
        if ((small_board[0, 2] == O) or (small_board[2, 0] == O)) and (small_board[1, 1] == O):
            score -= 2

    # check 2 sequence in global_board




    # Check two sequences in a row for X and  O in the global board
    for i in range(3):
        # Check for X
        if (state.global_cells[i * 3] + state.global_cells[i * 3 + 1] + state.global_cells[i * 3 + 2] == 2):
            score += 2
        # Check for O
        elif (state.global_cells[i * 3] + state.global_cells[i * 3 + 1] + state.global_cells[i * 3 + 2] == -2):
            score -= 2

    # Check two sequences in a column for X and O in the global board
    for j in range(3):
        if (state.global_cells[j + 0]) + (state.global_cells[j+ 3]) + (state.global_cells[j+6] == 2):
            score += 2
        elif (state.global_cells[j + 0]) + (state.global_cells[j + 3]) + (state.global_cells[j + 6] == -2):
            score -= 2


    # Check two sequences in the main diagonal for O in the global board
    if (state.global_cells[0]) + (state.global_cells[4]) + (state.global_cells[8]) == 2:
        score += 2

        # Check two sequences in the secondary diagonal for O in the global board
    if (state.global_cells[2]) + (state.global_cells[4]) + (state.global_cells[6]) == 2:
        score += 2

        # Repeat the above checks for O with appropriate adjustments to the score (subtract 2)

        # Check two sequences in the main diagonal for O in the global board
    if (state.global_cells[0]) + (state.global_cells[4]) + (state.global_cells[8])  == -2:
        score -= 2

        # Check two sequences in the secondary diagonal for O in the global board
    if (state.global_cells[2]) + (state.global_cells[4]) + (state.global_cells[6]) == -2:
        score -= 2

    return score

data_book_filename = "data_book.json"
data_book_cache = None

def heur3(state: State_2, data_book_filename: str):
    global data_book_cache

    memoized_values = {}

    def calculate_small_board_score(board):
        # print(board.flatten())

        # config_tuple = tuple(
        #     tuple(int(cell) for cell in row)
        #     for row in board)
        # #
        # # if config_tuple in memoized_values:
        # #     return memoized_values[config_tuple]
        # #
        # # # If state is not in the data book, evaluate using heur2 and save to file
        # # if not is_state_in_data_book(config_tuple, data_book_cache):
        # #     score = heur2(State_2(state))  # Create a new State_2 instance for evaluation
        # #     memoized_values[config_tuple] = score
        # #     save_to_data_book(config_tuple, score, data_book_cache)
        # #     return score
        #
        # score = heur2(State_2(state))  # Create a new State_2 instance for evaluation
        # memoized_values[config_tuple] = score
        # save_to_data_book(config_tuple, score, data_book_cache)
        # return score
        #
        # # If state is in the data book, retrieve the score
        # score = get_score_from_data_book(config_tuple, data_book_cache)
        # memoized_values[config_tuple] = score

        return 0
    # print("--------------->")
    # global_board_score = sum(calculate_small_board_score(board) for board in state.blocks.flatten())
    global_board_score = 0
    return global_board_score

# Use heur3 to evaluate the leaf nodes
# heur3_score = heur3(state, data_book_filename)
#
# def is_state_in_data_book(config_tuple, data_book_filename):
#     try:
#         with open(data_book_filename, 'r') as file:
#             data_book = json.load(file)
#             return str(config_tuple) in data_book
#     except FileNotFoundError:
#         return False

def save_to_data_book(config_tuple, score, data_book_filename):
    try:
        with open(data_book_filename, 'r') as file:
            data_book = json.load(file)
    except FileNotFoundError:
        data_book = {}

    data_book[str(config_tuple)] = score

    with open(data_book_filename, 'w') as file:
        json.dump(data_book, file)

# def get_score_from_data_book(config_tuple, data_book_filename):
#     with open(data_book_filename, 'r') as file:
#         data_book = json.load(file)
#         return data_book[str(config_tuple)]
import time
def select_move(cur_state, remain_time):
    global data_book_cache

    # Initialize data book if not already loaded
    if data_book_cache is None:
        data_book_filename = "data_book.json"
        data_book_cache = load_data_book(data_book_filename)

    valid_moves = cur_state.get_valid_moves
    print(valid_moves)
    if len(valid_moves) > 9:
        depth_minimax = 1
        return np.random.choice(valid_moves)
    else:
        depth_minimax = 5
    if len(valid_moves) != 0:
        start_time = time.time()
        time_limit = 3
        best_move = None
        best_score = -float('inf') if cur_state.player_to_move == cur_state.X else float('inf')

        for move in valid_moves:
            next_state = State_2(cur_state)
            if (not next_state.is_valid_move(move)):
                continue
            next_state.act_move(move)

            print("minimax call", depth_minimax)
            # Use minimax to evaluate the move
            score = minimax(next_state, depth=depth_minimax, alpha=-float('inf'), beta=float('inf'),
                            maximize=(cur_state.player_to_move == cur_state.X))

            # Update the best move if needed
            if cur_state.player_to_move == cur_state.X and score > best_score:
                best_score = score
                best_move = move
            elif cur_state.player_to_move == cur_state.O and score < best_score:
                best_score = score
                best_move = move

            if time.time() - start_time >= time_limit:
                valid_moves = cur_state.get_valid_moves
                if valid_moves:
                    choice = np.random.choice(valid_moves)
                    while len(valid_moves) and not cur_state.is_valid_move(choice):
                        choice = np.random.choice(valid_moves)
                    return choice

        return best_move

    valid_moves = cur_state.get_valid_moves
    if valid_moves:
        choice = np.random.choice(valid_moves)
        while len(valid_moves) and not cur_state.is_valid_move(choice):
            choice = np.random.choice(valid_moves)
        return choice

def minimax(state, depth, alpha, beta, maximize):
    print("minimax called", depth)
    if depth == 0 or state.game_over:
        return heur2(state)
        # return heur3(state, data_book_filename)  # Use heur3 to evaluate the leaf nodes

    valid_moves = state.get_valid_moves

    if maximize:
        max_eval = -float('inf')
        for move in valid_moves:
            next_state = State_2(state)
            if (not next_state.is_valid_move(move)):
                continue
            next_state.act_move(move)
            eval = minimax(next_state, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval
    else:
        min_eval = float('inf')
        for move in valid_moves:
            next_state = State_2(state)
            if (not next_state.is_valid_move(move)):
                continue
            next_state.act_move(move)
            eval = minimax(next_state, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval


def load_data_book(data_book_filename):
    try:
        with open(data_book_filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}



# import numpy as np
#
# class StateMock:
#     def __init__(self, blocks):
#         self.blocks = blocks
#
# # Test Case 1: Small board win for X
# test_case_1 = StateMock(np.array([[[-1, -1, -1], [0, 0, 0], [0, 0, 0]]] + [[[0, 0, 0], [-1, -1, -1], [0, 0, 0]]] + [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]] * 7))
# # Expected Score: 5
#
# # Test Case 2: Winning sequence for X
# test_case_2 = StateMock(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]] + [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]] * 8))
# # Expected Score: 4
#
# # Test Case 3: Center board win for X
# test_case_3 = StateMock(np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]] * 4 + [[[1, 1, 1], [0, 0, 0], [0, 0, 0]]] + [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]] * 4))
# # Expected Score: 10
#
# # Test Case 4: Corner board win for O
# test_case_4 = StateMock(np.array([[[0, 0, 0], [0, -1, 0], [0, 0, 0]]] + [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]] * 7 + [[[1, 1, 1], [0, 0, 0], [0, 0, 0]]]))
# # Expected Score: -3
#
# # Test Case 5: Center square occupied by X
# test_case_5 = StateMock(np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]] * 4 + [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]] + [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]] * 4))
# # Expected Score: 3
#
# # Test Case 6: Two squares occupied in a board
# test_case_6 = StateMock(np.array([[[0, 1, 0], [0, 0, 1], [0, 0, 0]]] + [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]] * 8))
# # Expected Score: 2
#
# # Test Case 7: Empty board
# test_case_7 = StateMock(np.zeros((9, 3, 3)))
# # Expected Score: 0
#
# # Test Case 8: Random configuration
# test_case_8 = StateMock(np.random.randint(-1, 2, size=(9, 3, 3)))
# # Expected Score: Varies based on the random configuration
#
# # Test Case 9: X and O wins
# test_case_9 = StateMock(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]] + [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]] * 3 +
#                                   [[[-1, 0, 0], [0, -1, 0], [0, 0, -1]]] + [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]] * 5))
# # Expected Score: Varies based on the specific wins
#
# # Test Case 10: Two consecutive wins for X
# test_case_10 = StateMock(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]] + [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]] + [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]] * 7))
# # Expected Score: 4
#
#
# # Assuming heur2 function is defined
#
# test_cases = [test_case_1, test_case_2, test_case_3, test_case_4, test_case_5,
#               test_case_6, test_case_7, test_case_8, test_case_9, test_case_10]
#
# for i, test_case in enumerate(test_cases, 1):
#     print(f"Test Case {i}:")
#     score = heur2(test_case)
#     print()
#     print(f"Heuristic Score: {score}\n")
