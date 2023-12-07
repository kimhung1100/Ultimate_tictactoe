from state import *

def print_blocks(State):
    blocks = State.blocks
    print(State.global_cells)
    print("----------------------------")
    for row in range(3):
        for col in range(3):
            print("Block [{}, {}]:".format(row, col))
            block = blocks[row * 3 + col]
            for i in range(3):
                for j in range(3):
                    print("{:2} ".format(int(block[i, j])), end="")
                print()
            print()