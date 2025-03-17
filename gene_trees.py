#!/usr/bin/env python3

# Trying to implement the same thing with monkey patching

import argparse
import concurrent.futures
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import random
import subprocess as sp
import sys

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
from scipy.stats import truncnorm, kstest, gmean, uniform
from skbio import TreeNode
import yaml


# DONE : write random walk function
# DONE : generate scaled gene trees from a species tree
# DONE : write lognormal function
# DONE : write function to make rates vary through time
# DONE : write random walk function
# DONE : generate genomes and infer divergences
# DONE : null model : rate changes after a constant time in the tree (Misha idea)
# TODO : add no muc defined in all trees
# TODO : the issue might be of correlation between A and B rates. But how ? there are enough steps in the random walk to make it uncorrelated
# TODO : uncorrelated : ideas : white noise process (special case of gamma, Drummond 2006), Cox - Ingersoll - Ross process (lepage 2007), mixed relaxed clock (lartillot 2016), lognormal

# Gene tree zone ---------------------------------------------------------------
def get_time_tree(tree, total_time):
    """
    Scale a tree to represent time.

    Parameters
    ----------
    tree : TreeNode
        The tree to scale.
    total_time : float
        The total time to scale the tree to.

    Returns
    -------
    tree : TreeNode
        The scaled tree.
    """
    if not tree.is_root():
        print("Tree is not rooted")
        return tree
    else:
        current_tree_height, _ = tree.height()
        scaling_factor = total_time / current_tree_height
        for node in tree.traverse(include_self=False):
            node.length *= scaling_factor
        return tree


def get_constant_rate_tree(tree, mu):
    """
    Get a "mutation rate" tree with a constant rate.

    Parameters
    ----------
    tree : TreeNode
        The tree to copy.
    mu : float
        The mutation rate.

    Returns
    -------
    mutation_rate_tree : TreeNode
        The mutation rate tree.
    """
    if not tree.is_root():
        print("Tree is not at root")
        return tree
    mutation_rate_tree = tree.copy()
    for node in mutation_rate_tree.traverse(include_self=False):
        node.length = mu
    return mutation_rate_tree


def get_lognormal_rate_tree(tree, beta, mu, muc, mus):
    """
    Get a "mutation rate" tree with an autocorrelated lognormal rate variation.

    Parameters
    ----------
    tree : TreeNode
        The tree to copy.
    beta : float
        The autocorrelation parameter.
    mu : float
        The "root" mutation rate.

    Returns
    -------
    mutation_rate_tree : TreeNode
        The mutation rate tree.
    """
    if not tree.is_root():
        print("Tree is not at root")
        return tree
    # initialize the mutation rate tree at mu
    mutation_rate_tree = tree.copy()
    mutation_rate_tree.length = mu
    for time_node, rate_node in zip(tree.traverse(include_self=False), mutation_rate_tree.traverse(include_self=False)):
        scale = beta * time_node.length
        muc_trunc = (np.log(muc) - np.log(rate_node.parent.length)) / scale
        mus_trunc = (np.log(mus) - np.log(rate_node.parent.length)) / scale
        log_new_mu = truncnorm.rvs(muc_trunc, mus_trunc, loc=np.log(rate_node.parent.length), scale=scale)
        rate_node.length = np.exp(log_new_mu)
    return mutation_rate_tree



def random_walk(mu, step, time, muc, mus, linear=False):
    """
    Generate a random walk with a given step size.

    Parameters
    ----------
    mu : float
        The initial value of the mutation rate.
    step : float
        The step size of the random walk.
    time : float
        The total time of the branch.
    muc : float
        The minimum value of the mutation rate.
    mus : float
        The maximum value of the mutation rate.

    Returns
    -------
    float
        The mutation rate at the end of the branch.
    """
    n_steps = int(time / step)
    rw = np.zeros(n_steps)

    lower_bound = True

    if muc == None:
        lower_bound = False
        muc = mu / 10
    if not linear:
        rw[0] = np.log(mu)
        range_mu = np.log(mus) - np.log(muc)
        mu_min = np.log(muc)
        mu_max = np.log(mus)
    else:
        rw[0] = mu
        range_mu = mus - muc
        mu_min = muc
        mu_max = mus

    sigma = range_mu / 100
    for i in range(1, n_steps):
        if np.random.random() > 0.5:
            rw[i] = rw[i - 1] + sigma
        else:
            rw[i] = rw[i-1] - sigma
        if rw[i] < mu_min and lower_bound:
            rw[i] = mu_min + sigma
        elif rw[i] > mu_max:
            rw[i] = mu_max - sigma

    if not linear:
        rw = np.exp(rw)
    return np.mean(rw), rw


def get_random_walk_tree(tree, rw_step, mu, muc, mus):
    """
    Get a "mutation rate" tree with a random walk rate variation.

    Parameters
    ----------
    tree : TreeNode
        The tree to copy.
    rw_step : float
        The step size of the random walk.
    mu : float
        The "root" mutation rate.

    Returns
    -------
    mutation_rate_tree : TreeNode
        The mutation rate tree.
    """
    if not tree.is_root():
        print("Tree is not at root")
        return tree
    # initialize the mutation rate tree at mu
    mutation_rate_tree = tree.copy()
    mutation_rate_tree.length = mu
    for time_node, rate_node in zip(tree.traverse(include_self=False), mutation_rate_tree.traverse(include_self=False)):
         rate_node.length, _ = random_walk(rate_node.parent.length, rw_step, time_node.length, muc, mus, linear=False)
    return mutation_rate_tree


def null_model_mu(time, timestep, mu_start, muc, mus):
    """
    Get the mutation rate after a given time in the null model.

    Parameters
    ----------
    time : float
        The time at which to get the mutation rate.
    timestep : float
        The time step at which the mutation rate changes.
    mu_start : float
        The initial mutation rate.
    muc : float
        The minimum mutation rate.
    mus : float
        The maximum mutation rate.


    Returns
    -------
    float
        The mutation rate after the given time.
    """
    if time < timestep:
        return mu_start
    else:
        steps = int(time / timestep)
        step_mus = np.zeros(steps)
        for i in range(steps):
            step_mus[i] = np.random.uniform(muc, mus)
        return gmean(step_mus)



def get_null_model_tree(time_tree, timestep, mu, muc, mus):
    """
    Get a "mutation rate" tree with a null model rate variation.

    Parameters
    ----------
    time_tree : TreeNode
        The time tree to copy.
    timestep : float
        The time step at which the mutation rate changes.
    mu : float
        The "root" mutation rate.
    muc: float
        The minimum value of the mutation rate.
    mus: float
        The maximum value of the mutation rate.

    Returns
    -------
    mutation_rate_tree : TreeNode
        The mutation rate tree.
    """
    if not time_tree.is_root():
        print("Tree is not at root")
        return time_tree
    # initialize the mutation rate tree at mu
    mutation_rate_tree = time_tree.copy()
    mutation_rate_tree.length = mu
    for time_node, rate_node in zip(time_tree.traverse(include_self=False), mutation_rate_tree.traverse(include_self=False)):
        rate_node.length = null_model_mu(time_node.length, timestep, rate_node.parent.length, muc, mus)

    return mutation_rate_tree


def generate_cherries(species_tree, n, muc, mus):
    """
    Generate cherries that respect Sheinman 2024 assumptions.

    Parameters
    ----------
    species_tree : TreeNode
        The 2-tipped species tree to generate cherries from.
    n : int
        The number of cherries to generate.
    muc : float
        The minimum mutation rate.
    mus : float
        The maximum mutation rate.

    Returns
    -------
    list
        The list of cherries.
    """
    cherries = []
    for _ in range(n):
        mu_a = np.random.uniform(muc, mus)
        mu_b = np.random.uniform(muc, mus)
        cherry = species_tree.copy()
        cherry.find("A").length = mu_a*cherry.find("A").length
        cherry.find("B").length = mu_b*cherry.find("B").length
        cherries.append(cherry)
    return cherries


def generate_gene_trees(species_tree, n, muc, mus, beta=None, rw_step=None, timestep=None, fixed_mu=None):
    """
    Generate gene trees from a species tree.

    Parameters
    ----------
    species_tree : TreeNode
        The species tree to generate gene trees from.
    n : int
        The number of gene trees to generate.
    muc : float
        The minimum mutation rate.
    mus : float
        The maximum mutation rate.

    Returns
    -------
    list
        The list of gene trees.
    """
    gene_trees = []
    if not species_tree.is_root():
        print("Species tree is not at root")
        return gene_trees
    for _ in range(n):
        gene_tree = species_tree.copy()
        if fixed_mu:
            mu = fixed_mu
        elif muc:
            mu = np.random.uniform(muc, mus)
        else:
            mu = np.random.uniform(mus/100, mus)
        if beta:
            mutation_rate_tree = get_lognormal_rate_tree(gene_tree, beta, mu, muc, mus)
        elif rw_step:
            mutation_rate_tree = get_random_walk_tree(gene_tree, rw_step, mu, muc, mus)
        elif timestep:
            mutation_rate_tree = get_null_model_tree(gene_tree, timestep, mu, muc, mus)
        else:
            mutation_rate_tree = get_constant_rate_tree(gene_tree, mu)

        # scale the gene tree with the mutation rate tree and the time species tree
        for gene_node, mutation_rate_node in zip(gene_tree.traverse(include_self=False), mutation_rate_tree.traverse(include_self=False)):
            gene_node.length = gene_node.length * mutation_rate_node.length

        gene_trees.append(gene_tree)
    return gene_trees



def linear_pdf(mu, muc, mus):
    """
    Generate a linear pdf.

    Parameters
    ----------
    mu : np.array
        The mutation rates.
    muc : float
        The minimum mutation rate.
    mus : float
        The maximum mutation rate.

    Returns
    -------
    np.array
        The pdf.
    """
    return np.where((mu >= muc) & (mu <= mus), 2 * mu/(mus**2 - muc**2), 0)


def linear_cdf(mu, muc, mus):
    """
    Generate a linear cdf.

    Parameters
    ----------
    mu : np.array
        The mutation rates.
    muc : float
        The minimum mutation rate.
    mus : float
        The maximum mutation rate.

    Returns
    -------
    np.array
        The cdf.
    """
    return np.where(mu < muc, 0, np.where(mu > mus, 1, (mu**2 - muc**2)/(mus**2 - muc**2)))


def get_all_distances(gene_trees):
    """
    Get all the distances between tips of gene trees.

    Parameters
    ----------
    gene_trees : list
        The list of gene trees to get the distances from.

    Returns
    -------
    dict
        The dictionary of distances.
    """
    tips = [tip.name for tip in gene_trees[0].tips()]
    distances = {}
    for t1, t2 in itertools.combinations(tips, 2):
        distances[(t1, t2)] = []
        for gene_tree in gene_trees:
            t1_node = gene_tree.find(t1)
            t2_node = gene_tree.find(t2)
            distances[(t1, t2)].append(t1_node.distance(t2_node))
    return distances

def get_average_mutation_rate(species_tree, tip_node):
    """
    Get the average mutation rate from the leaf to the root.

    Parameters
    ----------
    species_tree : TreeNode
        The species tree to get the time distances from.
    tip_node : TreeNode
        The tip node of the gene tree to get the mutation rate from.

    Returns
    -------
    float
        The average mutation rate.
    """
    if not tip_node.is_tip():
        print("Node is not a tip")
        return None
    current_node = tip_node
    mutation_rates = []
    while not current_node.is_root():
        mutation_rates.append(current_node.length)
        current_node = current_node.parent
    tree_height = species_tree.height()[0]
    return np.mean(mutation_rates) / tree_height


def plot_mutation_rate_distribution(ax, species_tree, tip_name, gene_trees, muc, mus, label="observed"):
    """
    Plot the distribution of the average of the mutation from one given leaf to the root.

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot on.
    species_tree : TreeNode
        The species tree to get the time distances from.
    tip_name : str
        The tip name to get the mutation rate from.
    gene_trees : list
        The list of gene trees to plot the distribution of mutation rates from.
    muc : float
        The minimum mutation rate.
    mus : float
        The maximum mutation rate.

    Returns
    -------
    None
    """

    rates = []
    for gene_tree in gene_trees:
        tip = gene_tree.find(tip_name)
        rates.append(get_average_mutation_rate(species_tree, tip))

    bins = np.logspace(np.log10(muc), np.log10(mus), 30)
    # hist, bins = np.histogram(rates, bins=bins, density=True)
    # bin_centers = (bins[1:] + bins[:-1]) / 2

    mu = np.linspace(muc, mus, 100)
    pdf = uniform.pdf(mu, muc, mus - muc)

    ax.hist(rates, bins, label=label, alpha=0.5, density=True)
    ax.set_xlabel("Mutation rate")
    ax.set_ylabel("Density")
    ax.set_title(f"Mutation rate distribution, tip : {tip_name}")
    ax.plot(mu, pdf, color="red")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(0.5, 0.1, f"{tip_name}", transform=ax.transAxes)
    ax.legend()



def plot_rate_correlation(ax, species_tree, gene_trees, tip_1, tip_2, muc, mus, label="observed"):
    """
    Plot the correlation of mutation rates between two tips.

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot on.
    species_tree : TreeNode
        The species tree to get the time distances from.
    gene_trees : list
        The list of gene trees to plot the distribution of mutation rates from.
    tip_1 : str
        The first tip name to get the mutation rate from.
    tip_2 : str
        The second tip name to get the mutation rate from.
    muc : float
        The minimum mutation rate.
    mus : float
        The maximum mutation rate.

    Returns
    -------
    None
    """
    rates_1 = []
    rates_2 = []
    for gene_tree in gene_trees:
        tip_1_node = gene_tree.find(tip_1)
        tip_2_node = gene_tree.find(tip_2)
        rates_1.append(get_average_mutation_rate(species_tree, tip_1_node))
        rates_2.append(get_average_mutation_rate(species_tree, tip_2_node))

    corr = np.corrcoef(rates_1, rates_2)
    label = f"{label} - corr : {corr[0, 1]:.2f}"
    ax.scatter(rates_1, rates_2, label=label)
    ax.plot([muc, mus], [muc, mus], color="red")
    ax.set_xlabel(f"Mutation rate {tip_1}")
    ax.set_ylabel(f"Mutation rate {tip_2}")
    ax.set_title(f"Mutation rate correlation, tips : {tip_1} - {tip_2}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()




def plot_distance_distribution(axs, species_tree, gene_trees, muc, mus, label="observed", tree_dir=None):
    """
    Plot the distribution of distances between tips of gene trees.

    Parameters
    ----------
    species_tree : TreeNode
        The species tree to get the time distances from.
    gene_trees : list
        The list of gene trees to plot the distribution of distances from.
    muc : float
        The minimum mutation rate.
    mus : float
        The maximum mutation rate.

    Returns
    -------
    None
    """
    distances = get_all_distances(gene_trees)
    n_combinations = len(distances)

    # plot the distribution of distances/time
    for i, (pair, dists) in enumerate(distances.items()):
        # compute time distance between tips
        t1_node = species_tree.find(pair[0])
        t2_node = species_tree.find(pair[1])
        total_time = t1_node.distance(t2_node)

        # log bin the distances
        bins = np.logspace(np.log10(muc*total_time), np.log10(mus*total_time), 30)
        hist, bins = np.histogram(dists, bins=bins, density=True)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        bin_widths = bins[1:] - bins[:-1]
        norm_hist = hist #/ bin_widths

        mu = np.linspace(muc*total_time, mus*total_time, 100)
        pdf = linear_pdf(mu, muc*total_time, mus*total_time)

        # compute the KS statistic
        # ks_res = kstest(dists, linear_cdf, args=(muc*total_time, mus*total_time))
        # ks_stat = ks_res.statistic
        # ks_where = ks_res.statistic_location

        if n_combinations == 1:
            ax = axs
        elif len(axs.shape) == 1:
            ax = axs[i]
        elif len(axs.shape) == 2:
            ax = axs[i, 1]
        # plot the distance distribution
        ax.plot(bin_centers, norm_hist, label=label)
        ax.set_xlabel("Evolutionary distance")
        ax.set_ylabel("Density")
        ax.plot(mu, pdf, color="red")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.text(0.5, 0.1, f"{pair[0]} - {pair[1]} - {total_time:.1e}y", transform=ax.transAxes)
        # plot corresponding tree
        if n_combinations != 1:
            if len(axs.shape) == 2 and tree_dir:
                tip_ordered = sorted(pair)
                tree_img = mpimg.imread(f"{tree_dir}/{tip_ordered[0]}{tip_ordered[1]}.png")
                axs[i, 0].imshow(tree_img)
                axs[i, 0].axis("off")

        # add ks_stat to plot
        # ax.text(0.5, 0.1, f"KS stat : {ks_stat:.2f}", transform=ax.transAxes)
        # ax.text(0.5, 0.15, f"KS where : {ks_where:.2f}", transform=ax.transAxes)


def run_alisim(gene_tree, outdir, length_gene, num):
    """
    Create sequences from Alisim for a gene tree.

    Parameters
    ----------
    gene_tree : TreeNode
        The gene tree to simulate.
    outdir : str
        The output directory.
    num : int
        The index of the gene tree.

    Returns
    -------
    """
    os.makedirs(outdir, exist_ok=True)
    output_prefix = f"{outdir}/gene_tree_{num}"
    tree_path = f"{outdir}/gene_tree_{num}.newick"
    gene_tree.write(tree_path)
    alisim_cmd = ["iqtree2", "--alisim", output_prefix, "-t", tree_path, "-m", "JC", "--out-format", "fasta", "--length", str(length_gene)]
    sp.run(alisim_cmd, check=True, capture_output=True)


def run_alisim_trees(gene_trees, outdir, length_gene, threads=1):
    """
    Run Alisim on all the gene trees in a parallel.

    Parameters
    ----------

    gene_trees : list
        A list of gene trees to generate sequences of.
    outdir: str
        The output directory.
    length_gene : int
        The length of a gene.
    threads : int
        The number of threads to use.

    Returns
    -------
    """
    # run alisim

    with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
        executor.map(run_alisim, gene_trees, itertools.repeat(outdir), itertools.repeat(length_gene), range(len(gene_trees)))

    # concatenate fasta
    # and create taxon_csv
    taxon_dic = {"genome": [], "clade": []}
    seq_dic = {node.name: Seq("") for node in gene_trees[0].tips()}
    for i in range(len(gene_trees)):
        seq_path = f"{outdir}/gene_tree_{i}.fa"
        seq_recs = list(SeqIO.parse(seq_path, "fasta"))
        # test all sequences correspond to tips
        assert set([seq.id for seq in seq_recs]) == set(seq_dic.keys())
        for seq in seq_recs:
            seq_dic[seq.id] += seq.seq
    for name, seq in seq_dic.items():
        rec = SeqRecord(seq, id=name, description="")
        SeqIO.write(rec, f"{outdir}/{name}.fasta", "fasta")
        taxon_dic["genome"].append(f"{name}.fasta")
        taxon_dic["clade"].append(name)
    taxon_df = pd.DataFrame(taxon_dic)
    taxon_df.to_csv(f"{outdir}/taxon.csv")






    
# HGT zone ---------------------------------------------------------------------

def set_color(self, color):
    """
    Set the color of the node.

    Parameters
    ----------
    self : TreeNode
        The node to set the color of.
    color : str
        The color to set the node to.

    Returns
    -------
    None
    """
    self.color = color


def get_color(self):
    """
    Get the color of the node.

    Parameters
    ----------
    self : TreeNode
        The node to get the color of.

    Returns
    -------
    str
        The color of the node.
    """
    return self.color


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


def make_nodes_compatible(self, timeline, colors, comp_node_dic, depth):
    """
    Creates new nodes where fit and add relevant nodes to the dictionary of compatible nodes.
    The comp node dictionary is a dictionary of lists of compatible nodes.
    The keys are the colors.

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
                new_node = TreeNode(f"nn_{child.name}_{timeline[time_index + 1]:.0f}", new_node_length, children=[child])
                child.length -= new_node_length
                self.append(new_node)
                comp_node_dic[timeline[time_index + 1]].append(new_node)
                new_node.make_nodes_compatible(timeline, comp_node_dic, depth + new_node_length)
            else:
                comp_node_dic[timeline[time_index + 1]].append(child)
                child.make_nodes_compatible(timeline, comp_node_dic, depth + child.length)


def delete_branch(self, child, comp_node_dic):
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
    branch_tips = child.tips()
    if not child in self.children:
        print(f"{child.name} is not a child of {self.name}")
        return None
    if len(branch_tips) > 1:
        print(f"Branch has more than one tip")
        return None
    else:
        nodes_to_delete = [node for node in child.traverse()]
        for color, comp_nodes in comp_node_dic.items():
            for node in nodes_to_delete:
                if node in comp_nodes:
                    comp_nodes.remove(node)
        self.remove(child)


    if self.is_root():
        # case where the transfer has happened from the outgroup
        # the root has only one child and needs to be removed
        new_root = self.children[0]
        new_root.parent = None
        new_root.length = 0
        # TODO self still exists ?
        self.children = []
        return new_root
    else:
        return self.root()

    
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

        # TODO delete single child nodes above self so that compatibility nodes are not left in the tree
        current_parent = self.parent
        current_child = self
        while len(current_parent.children) == 1:
            current_parent = current_parent.parent
            current_child = current_child.parent
        delete_branch(current_parent, current_child, comp_node_dic)


        recombination_node.append(self)
        self.length = recombination_point
        return self.root()
    else:
        print(f"No compatible nodes for {self.name}")
        return self.root()


def run_simulation(cfg):
    """
    Run a simulation of a species tree.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary.

    Returns
    -------
    float
        The minimum mutation rate.
    """
    mus = float(cfg["mus"])
    muc = cfg["muc"]
    try:
        muc = float(muc)
    except ValueError:
        muc = None
    if muc:
        range_mu = np.log(mus) - np.log(muc)
    else:
        range_mu = np.log(mus) - np.log(mus/100)
    tree_height = float(cfg["tree_height"])
    beta = range_mu/float(cfg["beta_fraction"])/tree_height
    rw_step = tree_height/float(cfg["rw_step_fraction"])
    os.makedirs(cfg["outdir"], exist_ok=True)

    species_tree = TreeNode.read([cfg["species_tree"]])
    time_tree = get_time_tree(species_tree, tree_height)
    if cfg["rate_evolution"] == "lognormal":
        gene_trees = generate_gene_trees(time_tree, cfg["n_gene_trees"], muc, mus, beta=beta)
    elif cfg["rate_evolution"] == "random_walk":
        gene_trees = generate_gene_trees(time_tree, cfg["n_gene_trees"], muc, mus, rw_step=rw_step)
    elif cfg["rate_evolution"] == "none":
        gene_trees = generate_gene_trees(time_tree, cfg["n_gene_trees"], muc, mus)
    elif len(species_tree.tips()) == 2 and cfg["rate_evolution"] == "none":
        gene_trees = generate_cherries(species_tree, cfg["n_gene_trees"], muc, mus)
    else:
        print("Unknown rate evolution")
        return None

    n_combinations = len(list(itertools.combinations([tip.name for tip in gene_trees[0].tips()], 2)))
    fig, axs = plt.subplots(n_combinations, 1, figsize=(10, 10))
    if not muc:
        all_mut_rates = []
        for gene_tree in gene_trees:
            mut_rate_tuple = (get_average_mutation_rate(time_tree, tip) for tip in gene_tree.tips())
            all_mut_rates.append(sum(mut_rate_tuple))
        if n_combinations == 1:
            muc = np.min(all_mut_rates)
            mus = np.max(all_mut_rates)
        else:
            sys.exit("Multiple combinations not implemented yet for undefined muc")

    print(f"muc : {muc:.2e}, mus : {mus:.2e}")
    plot_distance_distribution(axs, time_tree, gene_trees, muc, mus)
    fig.tight_layout()
    fig.savefig(f"{cfg['outdir']}/distance_distribution.png")
    run_alisim_trees(gene_trees, cfg["outdir"], cfg["length_gene"], threads=cfg["threads"])

    return muc, mus





TreeNode.set_color = set_color
TreeNode.get_color = get_color
TreeNode.get_timeline = get_timeline
TreeNode.make_nodes_compatible = make_nodes_compatible
TreeNode.homologous_recombination = homologous_recombination



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simulate gene trees.")
    parser.add_argument("config", help="The yaml configuration file.")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_simulation(cfg)


    if False:
        muc = None
        mus = 5e-9
        # simpler rate evolution test
        cherry_tau = 1e8
        # beta = (np.log(mus) - np.log(muc)) / cherry_tau
        rw_steps = [1e3, 1e4, 1e5]
        # fixed_mu = np.random.uniform(muc, mus)
        cherry_species_tree = TreeNode.read([f"(A:{cherry_tau/2},B:{cherry_tau/2});"])
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots(len(list(cherry_species_tree.tips())), 1, figsize=(10, 10))
        fig3, ax3 = plt.subplots()
        for rw in rw_steps:
            rw_step = cherry_tau / rw
            print(f"rw : {rw}")
            cherries = generate_gene_trees(cherry_species_tree, 1000, muc, mus, rw_step=rw_step)
            if not muc:
                muc= np.min([np.min([tip.length for tip in cherry.tips()]) for cherry in cherries])/(cherry_tau/2)
            print(f"empirical muc : {muc}")
            for i, tip in enumerate(cherry_species_tree.tips()):
                plot_mutation_rate_distribution(ax2[i], cherry_species_tree, tip.name, cherries, muc, mus, label=f"random walk steps ({rw:.0e})")
            plot_distance_distribution(ax, cherry_species_tree, cherries, muc, mus, label=f"random walk steps ({rw:.0e})")
            plot_rate_correlation(ax3, cherry_species_tree, cherries, "A", "B", muc, mus, label=f"random walk steps ({rw:.0e})")

        ax.legend()
        for ax in ax2:
            ax.legend()
        plt.show()



        # rate evolution test
        tree = TreeNode.read(["(((C:0.303,D:0.303)5:0.104,B:0.407)2:0.369,A:0.776);"])
        tree_height = 1e8
        beta_fraction = [10, 1, 1e-5]
        rw_step_fraction = [1e2, 1e3, 1e4]
        timestep_fraction = [5, 10, 20]

        time_tree = get_time_tree(tree, tree_height)

        tree_plot_dir = "test_tree_repr"
        n_combinations = len(list(itertools.combinations([tip.name for tip in time_tree.tips()], 2)))

        # beta fraction test
        fig, axs = plt.subplots(n_combinations, 2, figsize=(20, 20))
        for bf in beta_fraction:
            beta = (np.log(mus) - np.log(muc)) / bf / tree_height
            gene_trees = generate_gene_trees(time_tree, 5000, muc, mus, beta=beta)
            plot_distance_distribution(axs, time_tree, gene_trees, muc, mus, label=f"lognormal ({1/bf:.1f})", tree_dir=tree_plot_dir)
        for ax in axs:
            ax[1].legend()
        fig.tight_layout()
        fig.savefig("lognormal_rate_evolution.png", dpi=300)

        # random walk test
        fig, axs = plt.subplots(n_combinations, 2, figsize=(20, 20))
        for rwf in rw_step_fraction:
            rw_step = tree_height / rwf
            gene_trees = generate_gene_trees(time_tree, 5000, muc, mus, rw_step=rw_step)
            plot_distance_distribution(axs, time_tree, gene_trees, muc, mus, label=f"random walk ({rwf:.0e})", tree_dir=tree_plot_dir)
        for ax in axs:
            ax[1].legend()
        fig.tight_layout()
        fig.savefig("random_walk_rate_evolution.png", dpi=300)

        # timestep/null model test
        fig, axs = plt.subplots(n_combinations, 2, figsize=(20, 20))
        for tsf in timestep_fraction:
            timestep = tree_height / tsf
            gene_trees = generate_gene_trees(time_tree, 5000, muc, mus, timestep=timestep)
            plot_distance_distribution(axs, time_tree, gene_trees, muc, mus, label=f"null model ({tsf:.0f})", tree_dir=tree_plot_dir)
        for ax in axs:
            ax[1].legend()
        fig.tight_layout()
        fig.savefig("null_model_rate_evolution.png", dpi=300)



        # test lognormal truncated
        # plot the correlation between child and parent rates for different values of beta fraction
        # bf_gene_trees = {}
        # correlation = {}
        # for bf in beta_fraction:
        #     beta = (np.log(mus) - np.log(muc)) / bf / tree_height
        #     bf_gene_trees[bf] = generate_gene_trees(time_tree, 5000, muc, mus, beta=beta)
        #     for gene_tree in bf_gene_trees[bf]:
        #         for node in gene_tree.traverse(include_self=False):
        #             correlation[bf] =




        # test random walk
        n_rw = 10
        for _ in range(n_rw):
            mean, rw = random_walk(np.random.uniform(muc, mus), 1e4, 1e8, None, mus, linear=False)
            plt.plot(rw)
            plt.axhline(y=mean, color="green")
            plt.axhline(y=mus, color="blue")
            plt.axhline(y=muc, color="blue")
        plt.yscale("log")
        plt.show()





















        # HGT test
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
