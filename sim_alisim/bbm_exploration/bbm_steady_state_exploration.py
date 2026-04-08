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
from scipy.stats import truncnorm, kstest, gmean, uniform, norm
import seaborn as sns
from skbio import TreeNode
import yaml


# script_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(script_dir)
import gene_trees as gt


def stupid_random_walk(n):
    cur = np.zeros(n)
    for i in range(1, n):
        draw = np.random.random()
        if draw > 0.5:
            cur[i] = cur[i-1] + 1
        else:
            cur[i] = cur[i-1] - 1
    return np.mean(cur) / np.sqrt(n), cur[-1] / np.sqrt(n)

def stupid_bounded_random_walk(n, a, b):
    cur = np.zeros(n)
    for i in range(1, n):
        draw = np.random.random()
        if draw > 0.5:
            if cur[i-1] + 1 <= b:
                cur[i] = cur[i-1] + 1
            else:
                cur[i] = cur[i-1] - 1
        else:
            if cur[i-1] - 1 >= a:
                cur[i] = cur[i-1] - 1
            else:
                cur[i] = cur[i-1] + 1
    return np.mean(cur) / np.sqrt(n), cur[-1] / np.sqrt(n)


def stupid_dropping_bounded_random_walk(n, start, a, b):
    cur = np.array([start for _ in range(n)])
    for i in range(1, n):
        draw = np.random.random()
        if draw > 0.5:
            if cur[i-1] + 1 <= b:
                cur[i] = cur[i-1] + 1
            else:
                return np.NaN, np.NaN
        else:
            if cur[i-1] - 1 >= a:
                cur[i] = cur[i-1] - 1
            else:
                return np.NaN, np.NaN
    return np.mean(cur) / np.sqrt(n), cur[-1] / np.sqrt(n)




def bbm(mu, sigma, muc, mus, t):
    """
    Generate a bounded brownian motion.

    Parameters
    ----------
    mu : float
        The initial value of the mutation rate.
    sigma : float
        Evolution of the bbm.
    muc: float
        Lower bound of the bbm
    mus: float
        Higher bound of the bbm
    t : float
        The total time of the branch.

    Returns
    -------
    float
        The mutation rate at the end of the branch.
    """
    scale = sigma*np.sqrt(t)
    a, b = (muc - mu) / scale, (mus - mu) / scale
    distr = truncnorm(a, b, loc=mu, scale=scale)
    end_mu = distr.rvs()
    return (mu + end_mu) / 2


if __name__ == "__main__":
    muc = 6e-11
    mus = 5e-9
    mu = (muc + mus ) /2
    t=[1e7, 5e7, 1e8, 5e8, 1e9]
    sigma=1e-13
    fig, axs = plt.subplots(len(t), 1, sharex=True)

    # testing trunc norm
    for i in range(len(t)):
        scale = sigma*np.sqrt(t[i])
        a, b = (muc - mu) / scale, (mus - mu) / scale

        # testing truncnorm
        rvs = truncnorm.rvs(a, b, loc=mu, scale=scale, size=10000)
        print(f"loc = {mu}, mean = {rvs.mean()}, 1st and 10th decile = {np.quantile(rvs, (0.1,  0.9 ))}, min = {rvs.min()}, max = {rvs.max()}")
        axs[i].hist(x=rvs, bins=50, density=True, label="last")
        rvs_prec = (rvs + np.repeat(mu, 10000)) / 2
        axs[i].hist(x=rvs_prec, bins=50, density=True, label="mean")

    plt.show()

    # norm vs unbounded rw
    n =10000
    n_rw = 10000
    res = np.zeros(n_rw)
    for i in range(n_rw):
        _, res[i] = stupid_random_walk(n)
    res_th = norm.rvs(size=n_rw)
    fig, ax = plt.subplots()
    ax.hist(x=res, density=True, alpha = 0.5, label="rw")
    ax.hist(x=res_th, density=True, alpha = 0.5, label="norm")
    ax.legend()
    plt.show()


    # trunc norm vs bounded rw
    n =1000
    n_rw = 1000
    res = np.zeros(n_rw)
    for i in range(n_rw):
        _, res[i] = stupid_bounded_random_walk(n, 0, 10)
    res_th = truncnorm.rvs(0, 10, size=n_rw)
    fig, ax = plt.subplots()
    ax.hist(x=res, density=True, alpha = 0.5, label="brw")
    ax.hist(x=res_th, density=True, alpha = 0.5, label="truncnorm")
    ax.legend()
    plt.show()

    # start end norm vs unbounded rw average
    n =10000
    n_rw = 10000
    avg = np.zeros(n_rw)
    last = np.zeros(n_rw)
    for i in range(n_rw):
        avg[i], last[i] = stupid_random_walk(n)
    res_th = norm.rvs(size=n_rw) / 2
    last = last / 2
    fig, ax = plt.subplots()
    ax.hist(x=avg, bins="auto", density=True, alpha = 0.5, label="rw")
    ax.hist(x=res_th, bins="auto", density=True, alpha = 0.5, label="norm")
    ax.hist(x=last, bins="auto", density=True, alpha = 0.5, label="rw_start_end")
    ax.legend()
    plt.show()

    n =10000
    n_rw = 1000
    avg = np.zeros(n_rw)
    last = np.zeros(n_rw)
    for i in range(n_rw):
        avg[i], last[i] = stupid_bounded_random_walk(n, 0, 100)
    res_th = truncnorm.rvs(0, 100, size=n_rw) / 2
    fig, ax = plt.subplots()
    ax.hist(x=avg, bins="auto", density=True, alpha = 0.5, label="brw")
    ax.hist(x=res_th, bins="auto", density=True, alpha = 0.5, label="truncnorm")
    ax.hist(x=last, bins="auto", density=True, alpha = 0.5, label="brw_start_end")
    ax.legend()
    plt.show()

    n =1000
    n_rw = 1000
    avg = np.zeros(n_rw)
    last = np.zeros(n_rw)
    for i in range(n_rw):
        avg[i], last[i] = stupid_dropping_bounded_random_walk(n, 5000, 0, 10000)
    avg = avg[~np.isnan(avg)]
    last = last[~np.isnan(last)]
    print(len(avg))
    print(len(last))
    res_th = truncnorm.rvs(0, 10000, loc=5000, size=n_rw) / 2
    fig, ax = plt.subplots()
    ax.hist(x=avg, bins="auto", density=True, alpha = 0.5, label="brw")
    ax.hist(x=res_th, bins="auto", density=True, alpha = 0.5, label="truncnorm")
    ax.hist(x=last, bins="auto", density=True, alpha = 0.5, label="brw_start_end")
    ax.legend()
    plt.show()
