#!/usr/bin/env python3

import os
import random
import subprocess as sp
import sys


from skbio import TreeNode


# def set_color(self, color):
#     """
#     Set the color of the node.

#     Parameters
#     ----------
#     self : TreeNode
#         The node to set the color of.
#     color : str
#         The color to set the node to.

#     Returns
#     -------
#     None
#     """
#     self.color = color


# def get_color(self):
#     """
#     Get the color of the node.

#     Parameters
#     ----------
#     self : TreeNode
#         The node to get the color of.

#     Returns
#     -------
#     str
#         The color of the node.
#     """
#     return self.color


def get_timeline(self, timeline, depth):
    """
    Get the timeline of the tree.

    The timeline records the depth of each node in the tree along the time axis.

    Parameters
    ----------
    self : TreeNode
        The tree to get the timeline of.
    depth : float
        The depth of the tree.

    Returns
    -------
    list
        The timeline of the tree.
    """
    timeline.append(depth)
    if self.is_tip():
        pass
    else:
        for child in self.children:
            child.get_timeline(timeline, depth + child.length)


def make_nodes_compatible(self, timeline, comp_node_dic, depth):
    """
    Creates new nodes where fit and add relevant nodes to the dictionary of compatible nodes.
    The comp node dictionary is a dictionary of lists of compatible nodes.
    The keys are the depth.

    Parameters
    ----------
    self : TreeNode
        The tree to paint (monkey patched).
    timeline : list
        The timeline of the tree.
    colors : dict
        The dictionary of colors (keys : depth).
    comp_node_dic : dict
        The dictionary of compatible nodes.
    depth : float
        The depth of the node.

    Returns
    -------
    None
    """
    try:
        time_index = timeline.index(depth)
    except ValueError:
        print(f"Depth {depth} not in timeline")
        return None

    if self.is_tip():
        return

    else:
        # this is to prevent the self.children list from changing during the loop
        # which happens when a new node is created
        # I hate this behavior
        children_copy = self.children.copy()
        for child in children_copy:
            if child.length + depth > timeline[time_index + 1]:
                self.remove(child)
                new_node_length = timeline[time_index + 1] - depth
                new_node = TreeNode(f"nn_{child.name}_{time_index + 1}", new_node_length, children=[child])
                child.length -= new_node_length
                self.append(new_node)
                comp_node_dic[timeline[time_index + 1]].append(new_node)
                new_node.make_nodes_compatible(timeline, comp_node_dic, depth + new_node_length)
            else:
                comp_node_dic[timeline[time_index + 1]].append(child)
                child.make_nodes_compatible(timeline, comp_node_dic, depth + child.length)


def delete_branch(parent, child, comp_node_dic):
    """
    Delete the child branch : remove all single child nodes in the child branch.
    Take care of compatible node dic.
    Returns the root of the tree in case the root is deleted.

    Parameters
    ----------
    self : TreeNode
        The parent node.
    child : TreeNode
        The child node to prune.
    comp_node_dic : dict
        The dictionary of compatible nodes.

    Returns
    -------
    TreeNode
        The root of the tree.
    """
    branch_tips = list(child.tips())
    if not child in parent.children:
        print(f"{child.name} is not a child of {parent.name}")
    if len(branch_tips) > 1:
        print(f"Branch has more than one tip")
    else:
        # only to keep the same dictionary but maybe regenerate it each time
        # nodes_to_delete = [node for node in child.traverse()]
        # for _, comp_nodes in comp_node_dic.items():
            # for node in nodes_to_delete:
                # if node in comp_nodes:
                    # comp_nodes.remove(node)
        parent.remove(child)

    if parent.is_root():
        # case where the transfer has happened from the outgroup
        # the root has only one child and needs to be removed
        new_root = parent.children[0]
        new_root.parent = None
        new_root.length = 0
        # TODO self still exists ?
        parent.children = []
        return new_root

    return parent


def homologous_recombination(self, comp_node_dic):
    """
    Choose one of the compatible nodes to recombine with.
    Technically self is the node that will receive the transfer.
    Recombination in this case is subtree pruning and regrafting with a random compatible node.

    Parameters
    ----------
    self : TreeNode
        The node to recombine.
    comp_node_dic : dict
        The dictionary of compatible nodes.

    Returns
    -------
    TreeNode
        The root of the tree.
    """

    if self.is_root():
        print(f"Cannot recombine a root {self.name}")
        return self

    compatible_nodes = comp_node_dic.get(self.depth())
    # remove self from the list of compatible nodes

    compatible_nodes.remove(self)
    if compatible_nodes:
        # choose random compatible node
        recombination_branch = random.choice(compatible_nodes)

        # choose where to recombine along the branch
        # this is the distance from the tip of the branch (0 is the tip, branch.length is the root of the branch)
        recombination_point = random.uniform(0, recombination_branch.length)
        # create the new node
        # distance from the tip of the recombination point to the root of the branch
        recomb_node_length = recombination_branch.length - recombination_point
        recombination_node = TreeNode(f"hr_{self.name}_{recombination_branch.name}", recomb_node_length)
        # set relationships
        recomb_parent = recombination_branch.parent
        recombination_node.append(recombination_branch)
        recomb_parent.append(recombination_node)
        # distance from the tip of the recombination branch to the recombination point
        recombination_branch.length = recombination_point

        current_parent = self.parent
        current_child = self
        while len(current_parent.children) == 1:
            current_parent = current_parent.parent
            current_child = current_child.parent
        # current_parent = delete_branch(current_parent, current_child, comp_node_dic)
        current_parent.remove(current_child)
        if current_parent.is_root():
            current_parent = current_parent.children[0]


        recombination_node.append(self)
        self.length = recombination_point
    else:
        print(f"No compatible nodes for {self.name}")


def recombine_tree(self):
    """
    Recombine the tree by choosing a random node and recombining it with a random compatible node.

    Parameters
    ----------
    self : TreeNode
        The tree to recombine.

    Returns
    -------
    None
    """
    timeline = []
    self.get_timeline(timeline, 0)
    deduplicated_timeline = list(set(timeline))
    comp_node_dic = {i: [] for i in deduplicated_timeline}
    recomb_node = random.choice(list(self.traverse(include_self=False)))
    self.make_nodes_compatible(deduplicated_timeline, comp_node_dic, 0)
    # choose a random node
    # recombine the node
    recomb_node.homologous_recombination(comp_node_dic)
    self.prune()





TreeNode.get_timeline = get_timeline
TreeNode.make_nodes_compatible = make_nodes_compatible
TreeNode.homologous_recombination = homologous_recombination
TreeNode.recombine_tree = recombine_tree



if __name__ == "__main__":
    if False:
        # HGT test
        tree = TreeNode.read(["(((C:0.303,D:0.303)5:0.104,B:0.407)2:0.369,A:0.776);"])
        original_tree = tree.copy()
        print(tree.ascii_art())
        timeline = []
        tree.get_timeline(timeline, 0)
        deduplicated_timeline = list(set(timeline))
        print(deduplicated_timeline)

        comp_node_dic = {i: [] for i in deduplicated_timeline}
        tree.make_nodes_compatible(deduplicated_timeline, comp_node_dic, 0)
        print(tree.ascii_art())

        # recombination
        node_5 = tree.find("5")

        node_5.homologous_recombination(comp_node_dic)

        tree.prune()
        print(tree.ascii_art())

        # recombine_tree test
        tree = TreeNode.read(["(((C:0.303,D:0.303)5:0.104,B:0.407)2:0.369,A:0.776);"])
        original_tree = tree.copy()
        print(f"original height: {tree.height()[0]}")
        print(tree.ascii_art())
        tree.recombine_tree()
        print(tree.ascii_art())
        print(f"new height: {tree.height()[0]}")
