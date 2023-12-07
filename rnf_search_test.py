import keras
from state import *
from mcts import Edge, Node, MCTS
import numpy as np
from support import toNetworkInput
np.set_printoptions(precision=3, suppress=True)

model = keras.models.load_model("model_it10.keras")

mctsSearcher = MCTS(model)
g = State_2()
rootEdge = Edge(None, None)
rootEdge.N = 1
root = Node(g, rootEdge)
probs = mctsSearcher.search(root)
print(probs)
print(model.predict([toNetworkInput(g)])[0][0])