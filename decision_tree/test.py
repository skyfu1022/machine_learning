# from  trees import DecisionTree
from decision_tree import ID3
from tree_plotter import TreePlotter
import numpy as np

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lenses = np.array(lenses)
lenses_labels = ['age', 'prescript', 'astigmatic', 'tear_rate']
test = lenses[-1, :]
model = ID3(lenses, lenses_labels)
tree_plotter = TreePlotter()
# lenses_tree = model.train()
# model.save('decision_tree.txt')
lenses_tree = model.load('decision_tree.txt')
print(model.classify(lenses_tree, test))
tree_plotter.create_plot(lenses_tree)
print(lenses_tree)