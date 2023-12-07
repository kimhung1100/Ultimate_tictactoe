from state import State_2, UltimateTTT_Move

def toNetworkInput(state: State_2):
    posVec = []

    # X position, Y position
    for i in range(9):
        for j in range(3):
            for k in range(3):
                posVec.append(state.blocks[i, j, k])

    # Turn
    for i in range(3):
        if state.player_to_move == state.X:
            posVec.append(1)
        else:
            posVec.append(-1)

    return posVec

def getNetworkOutputIndex(move):
    # Assuming move is an instance of UltimateTTT_Move
    index_local_board = move.index_local_board
    x_coordinate = move.x
    y_coordinate = move.y

    # Calculate the index based on move coordinates
    index = index_local_board * 9 + x_coordinate * 3 + y_coordinate
    return index


if __name__ == "__main__":
    # Example usage
    initial_state = State_2()
    network_input = toNetworkInput(initial_state)
    print("Network Input:", network_input)

    initial_state.act_move(initial_state.get_valid_moves[0])
    network_input = toNetworkInput(initial_state)
    print("Network Input:", network_input)


    initial_state.act_move(initial_state.get_valid_moves[0])
    network_input = toNetworkInput(initial_state)
    print("Network Input:", network_input)


    # Create an instance of the State class
    example_state = State_2()

    # Assuming you have an instance of UltimateTTT_Move
    example_move = UltimateTTT_Move(index_local_board=0, x_coordinate=0, y_coordinate=2, value=1)

    # Get the network output index for the given move in the current state
    output_index = getNetworkOutputIndex(example_move)

    print("Network Output Index:", output_index)