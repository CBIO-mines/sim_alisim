#!/usr/bin/env python3



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
import seaborn as sns
from skbio import TreeNode
import yaml


# script_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(script_dir)
# import gene_trees as gt



def simple_random_walk(mu, rw_step, time, sigma):
    """
    Generate a random walk with a given step size.

    Parameters
    ----------
    mu : float
        The initial value of the mutation rate.
    rw_step : float
        The discrete time step size.
    time : float
        The total time of the branch.

    Returns
    -------
    float
        The mutation rate at the end of the branch.
    """
    n_steps = int(time / rw_step)
    rw = np.zeros(n_steps)
    rw[0] = mu
    for i in range(1, n_steps):
        if np.random.random() > 0.5:
            rw[i] = rw[i - 1] + sigma
        else:
            rw[i] = rw[i-1] - sigma
    return np.mean(rw), rw, rw[-1]



if __name__ == "__main__":
    # random walk steady state exploration
    muc = 6e-11
    mus = 5e-9
    n_rw = int(1e4)
    n_steps_list = [1e2, 1e3, 1e4]
    time = 1e8
    ncases = 15
    all_res = []
    plot_i = 0
    fig = plt.figure(layout="constrained", figsize=(len(n_steps_list)*5, ncases*5))
    subfigs = fig.subfigures(ncases, 1)


    # simple random walk, point start
    mu_start = [np.random.uniform(muc, mus)] * n_rw
    sigma = (mus - muc)/100
    res_dic = []
    for n_steps in n_steps_list:
        step_size = time / n_steps
        res_list = map(simple_random_walk, mu_start, itertools.repeat(step_size), itertools.repeat(time), itertools.repeat(sigma))
        res_dic += [{"mean": a,  "last": b, "n_steps": n_steps} for a, _, b in res_list]

    res_df = pd.DataFrame(res_dic)
    res_df = res_df.melt(id_vars=["n_steps"], value_vars=["mean", "last"], var_name="variable", value_name="Mutation rate")
    res_df["experience"] = "simple"
    res_df["space"] = "linear"
    res_df["start"] = "point"
    all_res.append(res_df)

    bins_dict = {}
    groups = res_df.groupby("n_steps")
    axs = subfigs[plot_i].subplots(ncols=len(groups), sharex=True)
    for (i, (nstep, step_df)) in enumerate(groups):
        rw_type = step_df["variable"].unique()

        for var in rw_type:
            sub_df = step_df[step_df["variable"] == var]
            axs[i].hist(x=sub_df["Mutation rate"], bins=30, density=True, alpha=0.5, label=var)

        axs[i].set_title(f"n_steps = {nstep}")
        axs[i].set_xlim([min(step_df["Mutation rate"]), max(step_df["Mutation rate"])])
        axs[i].legend()
    subfigs[plot_i].suptitle("Simple random walk, point start")



    plot_i+=1


    # simple log random walk, point start
    mu_start = [np.random.uniform(np.log(muc), np.log(mus))] * n_rw
    sigma = (np.log(mus) - np.log(muc))/100
    res_dic = []
    for n_steps in n_steps_list:
        step_size = time / n_steps
        res_list = map(simple_random_walk, mu_start, itertools.repeat(step_size), itertools.repeat(time), itertools.repeat(sigma))
        res_dic += [{"mean": a,  "last": b, "n_steps": n_steps} for a, _, b in res_list]

    res_df = pd.DataFrame(res_dic)
    res_df = res_df.melt(id_vars=["n_steps"], value_vars=["mean", "last"], var_name="variable", value_name="Mutation rate")
    res_df["Mutation rate"] = np.exp(res_df["Mutation rate"])
    res_df["experience"] = "simple"
    res_df["space"] = "log"
    res_df["start"] = "point"
    all_res.append(res_df)


    bins_dict = {}
    groups = res_df.groupby("n_steps")
    axs = subfigs[plot_i].subplots(ncols=len(groups), sharex=True)
    for (i, (nstep, step_df)) in enumerate(groups):
        rw_type = step_df["variable"].unique()

        for var in rw_type:
            sub_df = step_df[step_df["variable"] == var]
            axs[i].hist(x=sub_df["Mutation rate"], bins=30, density=True, alpha=0.5, label=var)

        axs[i].set_title(f"n_steps = {nstep}")
        axs[i].legend()
    subfigs[plot_i].suptitle("Simple log random walk, point start")



    plot_i += 1


    bins_dict = {}
    groups = res_df.groupby("n_steps")
    axs = subfigs[plot_i].subplots(ncols=len(groups), sharex=True)
    for (i, (nstep, step_df)) in enumerate(groups):
        rw_type = step_df["variable"].unique()

        for var in rw_type:
            sub_df = step_df[step_df["variable"] == var]
            min_mut_rate = sub_df["Mutation rate"].min()
            max_mut_rate = sub_df["Mutation rate"].max()


            axs[i].hist(x=sub_df["Mutation rate"], bins=40, density=True, alpha=0.5, label=var)

        axs[i].set_title(f"n_steps = {nstep}")
        axs[i].set_xlim([min(step_df["Mutation rate"]), max(step_df["Mutation rate"])])
        axs[i].set_xscale("log")
        axs[i].legend()
    subfigs[plot_i].suptitle("Simple log random walk, point start")



    plot_i += 1

    # simple log random walk, uniform start
    mu_start = np.random.uniform(np.log(muc), np.log(mus), n_rw)
    sigma = (np.log(mus) - np.log(muc))/100
    res_dic = []
    for n_steps in n_steps_list:
        step_size = time / n_steps
        res_list = map(simple_random_walk, mu_start, itertools.repeat(step_size), itertools.repeat(time), itertools.repeat(sigma))
        res_dic += [{"mean": a,  "last": b, "n_steps": n_steps} for a, _, b in res_list]

    res_df = pd.DataFrame(res_dic)
    res_df = res_df.melt(id_vars=["n_steps"], value_vars=["mean", "last"], var_name="variable", value_name="Mutation rate")
    res_df["Mutation rate"] = np.exp(res_df["Mutation rate"])
    res_df["experience"] = "simple"
    res_df["space"] = "log"
    res_df["start"] = "uniform"
    all_res.append(res_df)


    bins_dict = {}
    groups = res_df.groupby("n_steps")
    axs = subfigs[plot_i].subplots(ncols=len(groups), sharex=True)
    for (i, (nstep, step_df)) in enumerate(groups):
        rw_type = step_df["variable"].unique()

        for var in rw_type:
            sub_df = step_df[step_df["variable"] == var]
            min_mut_rate = sub_df["Mutation rate"].min()
            max_mut_rate = sub_df["Mutation rate"].max()

            axs[i].hist(x=sub_df["Mutation rate"], bins=40, density=True, alpha=0.5, label=var)

        axs[i].set_title(f"n_steps = {nstep}")
        axs[i].set_xlim([min(step_df["Mutation rate"]), max(step_df["Mutation rate"])])
        axs[i].set_xscale("log")
        axs[i].legend()
    subfigs[plot_i].suptitle("Simple log random walk, uniform start")


    plot_i += 1



    # linear point start
    mu_start = [np.random.uniform(muc, mus)] * n_rw
    res_dic = []
    for n_steps in n_steps_list:
        step_size = time / n_steps
        res_list = map(random_walk, mu_start, itertools.repeat(step_size), itertools.repeat(time), itertools.repeat(muc), itertools.repeat(mus), itertools.repeat(True))
        res_dic += [{"mean": a,  "last": b, "n_steps": n_steps} for a, _, b in res_list]

    res_df = pd.DataFrame(res_dic)
    res_df = res_df.melt(id_vars=["n_steps"], value_vars=["mean", "last"], var_name="variable", value_name="Mutation rate")
    res_df["experience"] = "bounded"
    res_df["space"] = "linear"
    res_df["start"] = "point"
    all_res.append(res_df)



    bins_dict = {}
    groups = res_df.groupby("n_steps")
    axs = subfigs[plot_i].subplots(ncols=len(groups), sharex=True)
    for (i, (nstep, step_df)) in enumerate(groups):
        rw_type = step_df["variable"].unique()

        for var in rw_type:
            sub_df = step_df[step_df["variable"] == var]
            axs[i].hist(x=sub_df["Mutation rate"], bins=30, density=True, alpha=0.5, label=var)

        axs[i].set_title(f"n_steps = {nstep}")
        axs[i].set_xlim([min(step_df["Mutation rate"]), max(step_df["Mutation rate"])])
        axs[i].legend()
    subfigs[plot_i].suptitle("Random walk, point start")


    plot_i += 1


    # linear uniform start
    mu_start = np.random.uniform(muc, mus, n_rw)
    res_dic = []
    for n_steps in n_steps_list:
        step_size = time / n_steps
        res_list = map(random_walk, mu_start, itertools.repeat(step_size), itertools.repeat(time), itertools.repeat(muc), itertools.repeat(mus), itertools.repeat(True))
        res_dic += [{"mean": a,  "last": b, "n_steps": n_steps} for a, _, b in res_list]

    res_df = pd.DataFrame(res_dic)
    res_df = res_df.melt(id_vars=["n_steps"], value_vars=["mean", "last"], var_name="variable", value_name="Mutation rate")
    res_df["experience"] = "bounded"
    res_df["space"] = "linear"
    res_df["start"] = "uniform"
    all_res.append(res_df)



    groups = res_df.groupby("n_steps")
    axs = subfigs[plot_i].subplots(ncols=len(groups), sharex=True)
    for (i, (nstep, step_df)) in enumerate(groups):
        rw_type = step_df["variable"].unique()

        for var in rw_type:
            sub_df = step_df[step_df["variable"] == var]
            axs[i].hist(x=sub_df["Mutation rate"], bins=30, density=True, alpha=0.5, label=var)

        axs[i].set_title(f"n_steps = {nstep}")
        axs[i].set_xlim([min(step_df["Mutation rate"]), max(step_df["Mutation rate"])])
        axs[i].legend()
    subfigs[plot_i].suptitle("Random walk, uniform start")


    plot_i += 1

    groups = res_df.groupby("n_steps")
    axs = subfigs[plot_i].subplots(ncols=len(groups), sharex=True)
    for (i, (nstep, step_df)) in enumerate(groups):
        rw_type = step_df["variable"].unique()

        for var in rw_type:
            sub_df = step_df[step_df["variable"] == var]
            min_mut_rate = sub_df["Mutation rate"].min()
            max_mut_rate = sub_df["Mutation rate"].max()

            axs[i].hist(x=sub_df["Mutation rate"], bins=40, density=True, alpha=0.5, label=var)

        axs[i].set_title(f"n_steps = {nstep}")
        axs[i].set_xlim([min(step_df["Mutation rate"]), max(step_df["Mutation rate"])])
        axs[i].set_xscale("log")
        axs[i].legend()
    subfigs[plot_i].suptitle("Random walk, uniform start")


    plot_i += 1




    # log point start
    mu_start = [np.random.uniform(muc, mus)] * n_rw
    res_dic = []
    for n_steps in n_steps_list:
        step_size = time / n_steps
        res_list = map(random_walk, mu_start, itertools.repeat(step_size), itertools.repeat(time), itertools.repeat(muc), itertools.repeat(mus), itertools.repeat(False))
        res_dic += [{"mean": a,  "last": b, "n_steps": n_steps} for a, _, b in res_list]

    res_df = pd.DataFrame(res_dic)
    res_df = res_df.melt(id_vars=["n_steps"], value_vars=["mean", "last"], var_name="variable", value_name="Mutation rate")
    res_df["Mutation rate"] = np.exp(res_df["Mutation rate"])
    res_df["experience"] = "bounded"
    res_df["space"] = "log"
    res_df["start"] = "point"
    all_res.append(res_df)


    bins_dict = {}
    groups = res_df.groupby("n_steps")
    axs = subfigs[plot_i].subplots(ncols=len(groups), sharex=True)
    for (i, (nstep, step_df)) in enumerate(groups):
        rw_type = step_df["variable"].unique()

        for var in rw_type:
            sub_df = step_df[step_df["variable"] == var]
            axs[i].hist(x=sub_df["Mutation rate"], bins=30, density=True, alpha=0.5, label=var)

        axs[i].set_title(f"n_steps = {nstep}")
        axs[i].set_xlim([min(step_df["Mutation rate"]), max(step_df["Mutation rate"])])
        axs[i].legend()
    subfigs[plot_i].suptitle("Log random walk, point start")


    plot_i += 1
    bins_dict = {}
    groups = res_df.groupby("n_steps")
    axs = subfigs[plot_i].subplots(ncols=len(groups), sharex=True)
    for (i, (nstep, step_df)) in enumerate(groups):
        rw_type = step_df["variable"].unique()

        for var in rw_type:
            sub_df = step_df[step_df["variable"] == var]
            min_mut_rate = sub_df["Mutation rate"].min()
            max_mut_rate = sub_df["Mutation rate"].max()

            axs[i].hist(x=sub_df["Mutation rate"], bins=40, density=True, alpha=0.5, label=var)

        axs[i].set_title(f"n_steps = {nstep}")
        axs[i].set_xlim([min(step_df["Mutation rate"]), max(step_df["Mutation rate"])])
        axs[i].set_xscale("log")
        axs[i].legend()
    subfigs[plot_i].suptitle("log random walk, point start")


    plot_i += 1




    # log, uniform start
    mu_start = np.random.uniform(muc, mus, n_rw)
    res_dic = []
    for n_steps in n_steps_list:
        step_size = time / n_steps
        res_list = map(random_walk, mu_start, itertools.repeat(step_size), itertools.repeat(time), itertools.repeat(muc), itertools.repeat(mus), itertools.repeat(False))
        res_dic += [{"mean": a,  "last": b, "n_steps": n_steps} for a, _, b in res_list]

    res_df = pd.DataFrame(res_dic)
    res_df = res_df.melt(id_vars=["n_steps"], value_vars=["mean", "last"], var_name="variable", value_name="Mutation rate")
    res_df["Mutation rate"] = np.exp(res_df["Mutation rate"])
    res_df["experience"] = "bounded"
    res_df["space"] = "log"
    res_df["start"] = "uniform"
    all_res.append(res_df)


    bins_dict = {}
    groups = res_df.groupby("n_steps")
    axs = subfigs[plot_i].subplots(ncols=len(groups), sharex=True)
    for (i, (nstep, step_df)) in enumerate(groups):
        rw_type = step_df["variable"].unique()

        for var in rw_type:
            sub_df = step_df[step_df["variable"] == var]
            axs[i].hist(x=sub_df["Mutation rate"], bins=30, density=True, alpha=0.5, label=var)

        axs[i].set_title(f"n_steps = {nstep}")
        axs[i].set_xlim([min(step_df["Mutation rate"]), max(step_df["Mutation rate"])])
        axs[i].legend()
    subfigs[plot_i].suptitle("Log random walk, uniform start")


    plot_i += 1

    bins_dict = {}
    groups = res_df.groupby("n_steps")
    axs = subfigs[plot_i].subplots(ncols=len(groups), sharex=True)
    for (i, (nstep, step_df)) in enumerate(groups):
        rw_type = step_df["variable"].unique()

        for var in rw_type:
            sub_df = step_df[step_df["variable"] == var]
            min_mut_rate = sub_df["Mutation rate"].min()
            max_mut_rate = sub_df["Mutation rate"].max()

            axs[i].hist(x=sub_df["Mutation rate"], bins=40, density=True, alpha=0.5, label=var)

        axs[i].set_title(f"n_steps = {nstep}")
        axs[i].set_xlim([min(step_df["Mutation rate"]), max(step_df["Mutation rate"])])
        axs[i].set_xscale("log")
        axs[i].legend()
    subfigs[plot_i].suptitle("Simple log random walk, uniform start")


    plot_i += 1


    # log, loguniform start
    mu_start = np.exp(np.random.uniform(np.log(muc), np.log(mus), n_rw))
    res_dic = []
    for n_steps in n_steps_list:
        step_size = time / n_steps
        res_list = map(random_walk, mu_start, itertools.repeat(step_size), itertools.repeat(time), itertools.repeat(muc), itertools.repeat(mus), itertools.repeat(False))
        res_dic += [{"mean": a,  "last": b, "n_steps": n_steps} for a, _, b in res_list]

    res_df = pd.DataFrame(res_dic)
    res_df = res_df.melt(id_vars=["n_steps"], value_vars=["mean", "last"], var_name="variable", value_name="Mutation rate")
    res_df["Mutation rate"] = np.exp(res_df["Mutation rate"])
    res_df["experience"] = "bounded"
    res_df["space"] = "log"
    res_df["start"] = "loguniform"
    all_res.append(res_df)


    bins_dict = {}
    groups = res_df.groupby("n_steps")
    axs = subfigs[plot_i].subplots(ncols=len(groups), sharex=True)
    for (i, (nstep, step_df)) in enumerate(groups):
        rw_type = step_df["variable"].unique()

        for var in rw_type:
            sub_df = step_df[step_df["variable"] == var]
            axs[i].hist(x=sub_df["Mutation rate"], bins=30, density=True, alpha=0.5, label=var)

        axs[i].set_title(f"n_steps = {nstep}")
        axs[i].set_xlim([min(step_df["Mutation rate"]), max(step_df["Mutation rate"])])
        axs[i].legend()
    subfigs[plot_i].suptitle("Log random walk, loguniform start")


    plot_i += 1

    bins_dict = {}
    groups = res_df.groupby("n_steps")
    axs = subfigs[plot_i].subplots(ncols=len(groups), sharex=True)
    for (i, (nstep, step_df)) in enumerate(groups):
        rw_type = step_df["variable"].unique()

        for var in rw_type:
            sub_df = step_df[step_df["variable"] == var]
            min_mut_rate = sub_df["Mutation rate"].min()
            max_mut_rate = sub_df["Mutation rate"].max()

            axs[i].hist(x=sub_df["Mutation rate"], bins=40, density=True, alpha=0.5, label=var)

        axs[i].set_title(f"n_steps = {nstep}")
        axs[i].set_xlim([min(step_df["Mutation rate"]), max(step_df["Mutation rate"])])
        axs[i].set_xscale("log")
        axs[i].legend()
    subfigs[plot_i].suptitle("Log random walk, loguniform start")


    plot_i += 1


    all_res_df = pd.concat(all_res)
    all_res_df.to_csv("all_experiments.csv.bz2", compression="bz2")


    # sum, log, uniform start
    mu_start_1 = np.random.uniform(muc, mus, n_rw)
    mu_start_2 = np.random.uniform(muc, mus, n_rw)
    res_dic = []
    for n_steps in n_steps_list:
        step_size = time / n_steps
        res_list_1 = map(random_walk, mu_start_1, itertools.repeat(step_size), itertools.repeat(time), itertools.repeat(muc), itertools.repeat(mus), itertools.repeat(False))
        res_list_2 = map(random_walk, mu_start_2, itertools.repeat(step_size), itertools.repeat(time), itertools.repeat(muc), itertools.repeat(mus), itertools.repeat(False))
        res_dic += [{"mean_1": a,  "last_1": b, "mean_2": c,  "last_2": d,  "n_steps": n_steps} for (a, _, b), (c, _, d) in zip(res_list_1, res_list_2)]

    res_df = pd.DataFrame(res_dic)
    res_df["mean_1"] = np.exp(res_df["mean_1"])
    res_df["mean_2"] = np.exp(res_df["mean_2"])
    res_df["last_1"] = np.exp(res_df["last_1"])
    res_df["last_2"] = np.exp(res_df["last_2"])
    res_df["mean"] = res_df["mean_1"] + res_df["mean_2"]
    res_df["last"] = res_df["last_1"] + res_df["last_2"]
    res_df = res_df[["mean", "last", "n_steps"]]
    res_df = res_df.melt(id_vars=["n_steps"], value_vars=["mean", "last"], var_name="variable", value_name="Mutation rate")

    bins_dict = {}
    groups = res_df.groupby("n_steps")
    axs = subfigs[plot_i].subplots(ncols=len(groups), sharex=True)
    for (i, (nstep, step_df)) in enumerate(groups):
        rw_type = step_df["variable"].unique()

        for var in rw_type:
            sub_df = step_df[step_df["variable"] == var]
            min_mut_rate = sub_df["Mutation rate"].min()
            max_mut_rate = sub_df["Mutation rate"].max()

            axs[i].hist(x=sub_df["Mutation rate"], bins=40, density=True, alpha=0.5, label=var)

        axs[i].set_title(f"n_steps = {nstep}")
        axs[i].set_xlim([min(step_df["Mutation rate"]), max(step_df["Mutation rate"])])
        axs[i].set_xscale("log")
        axs[i].legend()

    subfigs[plot_i].suptitle("Log random walk summed, uniform start")


    plot_i += 1


    # sum, log, loguniform start
    mu_start_1 = np.exp(np.random.uniform(np.log(muc), np.log(mus), n_rw))
    mu_start_2 = np.exp(np.random.uniform(np.log(muc), np.log(mus), n_rw))
    res_dic = []
    for n_steps in n_steps_list:
        step_size = time / n_steps
        res_list_1 = map(random_walk, mu_start_1, itertools.repeat(step_size), itertools.repeat(time), itertools.repeat(muc), itertools.repeat(mus), itertools.repeat(False))
        res_list_2 = map(random_walk, mu_start_2, itertools.repeat(step_size), itertools.repeat(time), itertools.repeat(muc), itertools.repeat(mus), itertools.repeat(False))
        res_dic += [{"mean_1": a,  "last_1": b, "mean_2": c,  "last_2": d,  "n_steps": n_steps} for (a, _, b), (c, _, d) in zip(res_list_1, res_list_2)]

    res_df = pd.DataFrame(res_dic)
    res_df["mean_1"] = np.exp(res_df["mean_1"])
    res_df["mean_2"] = np.exp(res_df["mean_2"])
    res_df["last_1"] = np.exp(res_df["last_1"])
    res_df["last_2"] = np.exp(res_df["last_2"])
    res_df["mean"] = res_df["mean_1"] + res_df["mean_2"]
    res_df["last"] = res_df["last_1"] + res_df["last_2"]
    res_df = res_df[["mean", "last", "n_steps"]]
    res_df = res_df.melt(id_vars=["n_steps"], value_vars=["mean", "last"], var_name="variable", value_name="Mutation rate")

    bins_dict = {}
    groups = res_df.groupby("n_steps")
    axs = subfigs[plot_i].subplots(ncols=len(groups), sharex=True)
    for (i, (nstep, step_df)) in enumerate(groups):
        rw_type = step_df["variable"].unique()

        for var in rw_type:
            sub_df = step_df[step_df["variable"] == var]
            min_mut_rate = sub_df["Mutation rate"].min()
            max_mut_rate = sub_df["Mutation rate"].max()

            axs[i].hist(x=sub_df["Mutation rate"], bins=40, density=True, alpha=0.5, label=var)

        axs[i].set_title(f"n_steps = {nstep}")
        axs[i].set_xlim([min(step_df["Mutation rate"]), max(step_df["Mutation rate"])])
        axs[i].set_xscale("log")
        axs[i].legend()

    subfigs[plot_i].suptitle("Log random walk summed, loguniform start")


    plot_i += 1

    fig.savefig("all_plt.png", dpi=300)

    # looking at log vs non log rw
    # simple case
    muc = 6e-11
    mus = 5e-9
    n_rw = 30
    time = 1e8
    nsteps = 1e3

    # simple random walk, point start
    mu_start = [1e-10] * n_rw
    sigma = (mus - muc)/100
    step_size = time / nsteps
    res_list = map(simple_random_walk, mu_start, itertools.repeat(step_size), itertools.repeat(time), itertools.repeat(sigma))
    res_lin = [pd.DataFrame({"x":np.arange(0, time, step_size), "rw": rw, "type": ["lin"] * int(nsteps), "which": f"{i+1}lin"}) for i, (_, rw, _) in enumerate(res_list)]

    # simple log rw, point start
    mu_start = [1e-10] * n_rw
    sigma = (np.log(mus) - np.log(muc))/100
    step_size = time / 1e3
    res_list = map(simple_random_walk, mu_start, itertools.repeat(step_size), itertools.repeat(time), itertools.repeat(sigma))
    res_log = [pd.DataFrame({"x":np.arange(0, time, step_size), "rw": rw, "type": ["log"] * int(nsteps), "which": f"{i+1}log"}) for i, (_, rw, _) in enumerate(res_list)]
    res_df = pd.concat([pd.concat(res_lin), pd.concat(res_log)])

    sns.lineplot(
        data=res_df,
        x="x", y="rw", hue="type", units="which",
        estimator=None,lw=1
    )
    plt.show()
