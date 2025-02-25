#!/usr/bin/env python3

# Trying to implement the same thing with monkey patching

import random

from skbio import TreeNode

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
    The keys are the timeline depths.

    Parameters
    ----------
    self : TreeNode
        The tree to paint (monkey patched).
    timeline : list
        The timeline of the tree.
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
                new_node = TreeNode(f"nn_{child.name}_{timeline[time_index + 1]:.0f}", new_node_length, children=[child])
                child.length -= new_node_length
                self.append(new_node)
                comp_node_dic[timeline[time_index + 1]].append(new_node)
                new_node.make_nodes_compatible(timeline, comp_node_dic, depth + new_node_length)
            else:
                comp_node_dic[timeline[time_index + 1]].append(child)
                child.make_nodes_compatible(timeline, comp_node_dic, depth + child.length)




def set_node_compatible(self, comp_node_dic, depth):
    """
    Recursively set the compatibility list of all the nodes of the tree.
    Assuming the tree has already been had had new nodes created to reflect compatibility.


    Parameters
    ----------
    self : TreeNode
        The tree to set the compatibility of.
    comp_node_dic : dict
        The dictionary of compatible nodes.
    depth : float
        The depth of the node.

    Returns
    -------
    None
    """
    self.compatible_nodes = [node for node in comp_node_dic[depth] if node != self]
    for child in self.children:
        child.set_node_compatible(comp_node_dic, depth + child.length)


def homologous_recombination(self):
    """
    Choose one of the compatible nodes to recombine with.
    Technically self is the node that will recieve the transfer.
    Recombination in this case is subtree pruning and regrafting with a random compatible node.

    Parameters
    ----------
    self : TreeNode
        The node to recombine.

    Returns
    -------
    None
    """
    import pdb; pdb.set_trace()
    if self.compatible_nodes:
        # choose random compatible node
        recombination_branch = random.choice(self.compatible_nodes)

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

        # TODO delete single child nodes above self so that fake nodes are not left in the tree
        # how, by name ? or prune the first parent that has more than one child (or the one before that)
        # move current node to the recombination node
        recombination_node.append(self)
        self.length = recombination_point
    else:
        print(f"No compatible nodes for {self.name}")





TreeNode.get_timeline = get_timeline
TreeNode.set_node_compatible = set_node_compatible
TreeNode.make_nodes_compatible = make_nodes_compatible
TreeNode.homologous_recombination = homologous_recombination



if __name__ == "__main__":
    tree = TreeNode.read(["((A:5,B:3):2,C:4)root;"])
    original_tree = tree.copy()
    print(tree.ascii_art())
    tree.set_color_tree("red")
    timeline = []
    tree.get_timeline(timeline, 0)
    deduplicated_timeline = list(set(timeline))
    print(deduplicated_timeline)

    comp_node_dic = {i: [] for i in deduplicated_timeline}
    tree.make_nodes_compatible(deduplicated_timeline, comp_node_dic, 0)
    print(tree.ascii_art())
    tree.set_node_compatible(comp_node_dic, 0)

    # recombination
    C = tree.find("C")
    C.homologous_recombination()
    tree.prune()
    print(tree.ascii_art())
