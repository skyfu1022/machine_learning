from matplotlib import pyplot as plt


class TreePlotter:

    def __init__(self):
        self._decicion_node = dict(boxstyle='sawtooth', fc='0.8')
        self._leaf_ndoe = dict(boxstyle='round4', fc='0.8')
        self._arrow_args = dict(arrowstyle='<-')

    def plot_node(self, node_txt, center_pt, parent_pt, node_type):
        self._ax.annotate(node_txt, xy=parent_pt, xycoords='axes fraction', xytext=center_pt,
                          textcoords='axes fraction', va="center", ha="center", bbox=node_type, arrowprops=self._arrow_args)

    def get_num_leafs(self, decision_tree):
        num_leafs = 0
        first_str = list(decision_tree.keys())[0]
        second_dict = decision_tree[first_str]
        for key in second_dict.keys():
            if type(second_dict[key]).__name__ == 'dict':
                num_leafs += self.get_num_leafs(second_dict[key])
            else:
                num_leafs += 1
        return num_leafs

    def get_tree_depth(self, decision_tree):
        max_depth = 0
        first_str = list(decision_tree.keys())[0]
        second_dict = decision_tree[first_str]
        for key in second_dict.keys():
            if type(second_dict[key]).__name__ == 'dict':
                this_depth = 1 + self.get_tree_depth(second_dict[key])
            else:
                this_depth = 1
            if this_depth > max_depth:
                max_depth = this_depth
        return max_depth

    def retrieve_tree(self, idx):
        list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                         {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
        return list_of_trees[idx]

    def plot_mid_text(self, center_pt, parent_pt, txt_string):
        x_mid = (parent_pt[0] - center_pt[0]) / 2.0 + center_pt[0]
        y_mid = (parent_pt[1] - center_pt[1]) / 2.0 + center_pt[1]
        self._ax.text(x_mid, y_mid, txt_string)

    def plot_tree(self, decision_tree, parent_pt, node_txt):
        num_leafs = self.get_num_leafs(decision_tree)
        depth = self.get_tree_depth(decision_tree)
        first_str = list(decision_tree.keys())[0]
        center_pt = (self._x_off + (1.0 + float(num_leafs)) / 2.0 / self._total_w, self._y_off)
        self.plot_mid_text(center_pt, parent_pt, node_txt)
        self.plot_node(first_str, center_pt, parent_pt, self._decicion_node)
        second_dict = decision_tree[first_str]
        self._y_off -= 1.0 / self._total_d
        for key in second_dict.keys():
            if type(second_dict[key]).__name__ == 'dict':
                self.plot_tree(second_dict[key], center_pt, str(key))
            else:
                self._x_off += 1.0 / self._total_w
                self.plot_node(second_dict[key], (self._x_off, self._y_off), center_pt, self._leaf_ndoe)
                self.plot_mid_text((self._x_off, self._y_off), center_pt, str(key))
        self._y_off += 1.0 / self._total_d

    def create_plot(self, decision_tree):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        self._ax = plt.subplot(111, frameon=False)
        self._total_w = float(self.get_num_leafs(decision_tree))
        self._total_d = float(self.get_tree_depth(decision_tree))
        self._x_off = -0.5 / self._total_w
        self._y_off = 1.0
        self.plot_tree(decision_tree, (0.5, 1.0), '')
        plt.show()

# tree_plotter = TreePlotter()
# my_tree = tree_plotter.retrieve_tree(0)
# print(tree_plotter.get_num_leafs(my_tree))
# print(tree_plotter.get_tree_depth(my_tree))
# tree_plotter.create_plot(my_tree)