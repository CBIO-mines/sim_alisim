#!/usr/bin/env python3

import argparse
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skbio import TreeNode
import yaml

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
mosaic_method_path = "/home/paulimer/Documents/test_florian/"
sys.path.append(mosaic_method_path)

import gene_trees as gt
from main_inference import run_inference
from fit.fit import theoretical_mld, fit_params
from parse.fun import bin_mld
from parse import fun as parse_fun
from lastz_parallel_db import utils as lastz_utils


import synthetic_mld as smld

def inf_add_suffix(inference_cfg, suffix):
    for key in inference_cfg:
        if key.endswith("dir"):
            inference_cfg[key] = inference_cfg[key] + suffix


def simplified_inference(cfg, genomes_dir=None):

    if not os.path.exists(cfg["taxon_csv"]):
        cfg["taxon_csv"] = os.path.join(cfg["genomes_dir"], cfg["taxon_csv"])
        if not os.path.exists(cfg["taxon_csv"]):
            print(f"Taxon csv not found at {cfg['taxon_csv']}")
            sys.exit(1)

    masked_genomes_dir = cfg["genomes_dir"]
    os.makedirs(cfg["results_dir"], exist_ok=True)
    # alignment
    print("Aligning genomes")
    database_path = os.path.join(cfg["results_dir"], cfg["database_name"])
    if os.path.exists(database_path):
        update_db = True
    else:
        update_db = False
    con = lastz_utils.create_lastz_db(
        cfg["taxon_csv"],
        masked_genomes_dir,
        cfg["cluster_name"],
        database_path,
        cfg["max_threads"],
        update_db,
        cfg["aligner"]
    )
    con.close()

    # mlds
    print("Computing MLDs")
    taxon_df = pd.read_csv(cfg["taxon_csv"])
    level_list = sorted(taxon_df[cfg["cluster_name"]].unique())
    levels = list(itertools.combinations(level_list, 2))
    binned_mlds = {}
    for level in levels:
        genome_comps = parse_fun.get_genome_comp(level, cfg["taxon_csv"], "", cfg["cluster_name"], output_csv=False)
        full_mld = parse_fun.get_all_mlds(genome_comps, database_path, threads=cfg["max_threads"])
        summed_mld = parse_fun.sum_mlds(full_mld)
        binned_mld = parse_fun.bin_mld(
            summed_mld,
            linear_bin_width=3,
            limit_size=30.5,
            power_increment=0.1,
            ncomp=full_mld.shape[0]
        )
        binned_mlds[level] = binned_mld

    os.makedirs(os.path.join(cfg["results_dir"], "binned_mlds"), exist_ok=True)
    for level in levels:
        binned_mlds[level].to_csv(
            os.path.join(cfg["results_dir"], "binned_mlds", f"{level[0]}_{level[1]}.csv"),
            index=False
        )





def plot_mld_fit_and_expected(ax, res_dir, empirical_muc, empirical_mus, delta, simulated_params, summed_mu_array, tips_mut_rate, genome_length, level, save_res=None, outfile=None, plotminus4=False):
    """Plots the fit of the mosaic model to the MLDs and the expected MLD given the simulation parameters."""

    # get expected and fitted MLDs
    binned_mld = pd.read_csv(os.path.join(res_dir, f"binned_mlds/{level[0]}_{level[1]}.csv"))

    # mus and muc are different for each comp, need to refit here
    res_opt, _, res_opt_4 = fit_params(
        "dual-annealing",
        np.array([8, -8]),
        binned_mld["freq"],
        0.1,
        binned_mld["match_length"],
        empirical_mus,
        empirical_muc,
        delta,
        genome_length
        )
    if plotminus4:
        fitted_params = res_opt_4.x
    else:
        fitted_params = res_opt.x

    if save_res:
        with open(save_res, "a") as f:
            f.write(f"{level[0]},{level[1]},{fitted_params[0]},{fitted_params[1]}\n")


    max_x = binned_mld["match_length"].max()
    min_y = binned_mld[binned_mld["freq"] > 0]["freq"].min()
    if binned_mld["freq"].sum() == 0:
        # mld is empty, do not plot
        return
    r = np.logspace(0, np.log10(max_x), 1000)
    if plotminus4:
        _, mc = theoretical_mld(fitted_params, 0.1, r, empirical_mus, empirical_muc, delta, genome_length, False)
    else:
        mh, mc = theoretical_mld(fitted_params, 0.1, r, empirical_mus, empirical_muc, delta, genome_length, False)
    _, mc_simulated = theoretical_mld(simulated_params, 0.1, r, empirical_mus, empirical_muc, delta, genome_length, False)

    # get synthetic MLD
    # synthetic_mld = smld.synthetic_mld(summed_mu_array, 10**simulated_params[0], 1000)
    # synthetic_df = pd.DataFrame({"match_length": np.arange(1, 1001), "freq": synthetic_mld})
    # binned_synthetic_df = bin_mld(synthetic_df, 3, 30.5, 0.1, 1)

    synthetic_mld_delta = smld.synthetic_mld(summed_mu_array, 10**simulated_params[0], 1000, 100, delta)
    synthetic_df_delta = pd.DataFrame({"match_length": np.arange(1, 1001), "freq": synthetic_mld_delta})
    binned_synthetic_df_delta = bin_mld(synthetic_df_delta, 3, 30.5, 0.1, 1)


    ax.plot(binned_mld["match_length"], binned_mld["freq"], 'o', label="Observed", color="black")
    # ax.plot(binned_synthetic_df_delta["match_length"], binned_synthetic_df_delta["freq"], 'o', label="Synthetic - delta", color="purple", alpha=0.5)
    # ax.plot(binned_synthetic_df["match_length"], binned_synthetic_df["freq"], 'o', label="Synthetic", color="grey", alpha=0.5)
    if not plotminus4:
        ax.plot(r, mh, label="mh - fit", color="red")
        ax.plot(r, mc, label="mc - fit", color="blue")
    else:
        ax.plot(r, mc, label="fit", color="blue")
    ax.plot(r, mc_simulated, label="mc - expected", color="green")
    ax.set_ylim(ymin=min_y/10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(0.1, 0.1, f"mus = {empirical_mus:.2e}\nmuc = {empirical_muc:.2e}", transform=ax.transAxes)
    ax.legend()
    ax.set_title(f"MLD fit for {level[0]} vs {level[1]}")
    ax.set_xlabel("Match length")
    ax.set_ylabel("Frequency")
    if outfile:
        plt.savefig(outfile)
    return {"genome_1": level[0], "genome_2": level[1], "fit_tau": fitted_params[0], "sim_tau": simulated_params[0]}


def simulate_infer(simulation_cfg, inference_cfg):
    if type(simulation_cfg["rw_step_fraction"]) is list and simulation_cfg["rate_evolution"] == "random_walk":
        rw_step_fraction_list = simulation_cfg["rw_step_fraction"].copy()
        if type(simulation_cfg["tree_height"]) is list:
            tree_heights = simulation_cfg["tree_height"].copy()
            tree_heights = [float(tree_height) for tree_height in tree_heights]
        else:
            tree_heights = [float(simulation_cfg["tree_height"])]
        all_res_df_list = []
        for tree_height in tree_heights:
            print(f"Tree height: {tree_height}")
            for rw_step_fraction in rw_step_fraction_list:
                res_fit = []
                simulation_cfg_copy = simulation_cfg.copy()
                inference_cfg_copy = inference_cfg.copy()
                simulation_cfg_copy["rw_step_fraction"] = rw_step_fraction
                suffix = f"_height_{tree_height:.0e}_rw_{float(rw_step_fraction):.0e}"
                inf_add_suffix(inference_cfg_copy, suffix)
                inf_add_suffix(simulation_cfg_copy, suffix)
                print("################################################")
                print("Simulating gene trees and sequences Random walk " + str(rw_step_fraction))
                print("################################################")
                species_tree = TreeNode.read([simulation_cfg_copy["species_tree"]])
                n_combinations = len(list(itertools.combinations(species_tree.tips(), 2)))
                figaa, axaa = plt.subplots(n_combinations, 2, figsize=(15, 5*n_combinations))
                simulation_cfg_copy["tree_height"] = tree_height
                tips_mut_rate = gt.run_simulation(simulation_cfg_copy, axaa)
                all_summed_mus = {}
                simplified_inference(inference_cfg_copy)
                for i, (pair, mut_rate) in enumerate(tips_mut_rate.items()):
                    all_summed_mus[pair] = [(a + b)/2 for a, b in mut_rate]
                    if simulation_cfg_copy["exp_mus"]:
                        muc = np.min(all_summed_mus[pair])
                        mus = np.max(all_summed_mus[pair])
                    else:
                        muc = float(simulation_cfg_copy["muc"])
                        mus = float(simulation_cfg_copy["mus"])
                    time_tree = gt.get_time_tree(species_tree, tree_height)
                    sim_tau = time_tree.find(pair[0]).distance(time_tree.find(pair[1]))
                    if n_combinations == 1:
                        res_fit.append(plot_mld_fit_and_expected(
                            axaa[1],
                            inference_cfg_copy["results_dir"],
                            muc, mus, inference_cfg_copy["delta"],
                            (np.log10(sim_tau), -20),
                            all_summed_mus[pair],
                            mut_rate,
                            simulation_cfg_copy["length_gene"]*simulation_cfg_copy["n_gene_trees"],
                            pair,
                            save_res=os.path.join(inference_cfg_copy["results_dir"], "fitted_params.csv"),
                            plotminus4=True
                        ))
                    else:
                        res_fit.append(plot_mld_fit_and_expected(
                            axaa[i, 1],
                            inference_cfg_copy["results_dir"],
                            muc, mus, inference_cfg_copy["delta"],
                            (np.log10(sim_tau), -20),
                            all_summed_mus[pair],
                            mut_rate,
                            simulation_cfg_copy["length_gene"]*simulation_cfg_copy["n_gene_trees"],
                            pair,
                            save_res=os.path.join(inference_cfg_copy["results_dir"], "fitted_params.csv"),
                            plotminus4=True
                        ))

                figaa.tight_layout()
                figaa.savefig(os.path.join(inference_cfg_copy["results_dir"], "distr_fit_and_expected.png"), dpi=300)
                res_fit_df = pd.DataFrame(res_fit)
                res_fit_df["rw_step_fraction"] = rw_step_fraction
                res_fit_df["tree_height"] = tree_height
                all_res_df_list.append(res_fit_df)

        all_res_fit_df = pd.concat(all_res_df_list).reset_index(drop=True)
        all_res_fit_df.to_csv(os.path.join(os.path.dirname(inference_cfg["results_dir"]), "all_res_fit.csv"), index=False)


    # elif type(simulation_cfg["beta_fraction"]) is list and simulation_cfg["rate_evolution"] == "lognormal":
    #     # TODO obsolete
    #     beta_fraction_list = simulation_cfg["beta_fraction"].copy()
    #     for beta_fraction in beta_fraction_list:
    #         simulation_cfg_copy = simulation_cfg.copy()
    #         inference_cfg_copy = inference_cfg.copy()
    #         simulation_cfg_copy["beta_fraction"] = beta_fraction
    #         suffix = f"_ln_{float(beta_fraction):.2e}"
    #         inf_add_suffix(inference_cfg_copy, suffix)
    #         inf_add_suffix(simulation_cfg_copy, suffix)
    #         print("################################################")
    #         print("Simulating gene trees and sequences Lognormal " + str(beta_fraction))
    #         print("################################################")
    #         figaa, axaa = plt.subplots(1, 2, figsize=(10, 5))
    #         tips_mut_rate = gt.run_simulation(simulation_cfg_copy, axaa[0])
    #         all_summed_mus = {}
    #         for pair, mut_rate in tips_mut_rate.items():
    #             all_summed_mus[pair] = [(a + b)/2 for a, b in mut_rate]

    #         muc = np.min(all_summed_mus[("A", "B")])
    #         mus = np.max(all_summed_mus[("A", "B")])

    #         print("Running inference")
    #         inference_cfg["muc"] = muc
    #         inference_cfg["mus"] = mus
    #         run_inference(inference_cfg_copy)
    #         plot_mld_fit_and_expected(
    #             axaa[1],
    #             inference_cfg_copy["results_dir"],
    #             muc, mus, inference_cfg_copy["delta"],
    #             (np.log10(2*float(simulation_cfg_copy["tree_height"])), -20),
    #             all_summed_mus,
    #             simulation_cfg_copy["length_gene"]*simulation_cfg_copy["n_gene_trees"],
    #         )
    #         figaa.tight_layout()
    #         figaa.savefig(os.path.join(inference_cfg_copy["results_dir"], "fit_and_sim.png"))

    else:
        if simulation_cfg["rate_evolution"] == "random_walk":
            print_info = "Random walk" + simulation_cfg["rw_step_fraction"]
        elif simulation_cfg["rate_evolution"] == "lognormal":
            print_info = "Lognormal" + simulation_cfg["beta_fraction"]
        else:
            print_info = simulation_cfg["rate_evolution"]
        print("################################################")
        print("Simulating gene trees and sequences " + print_info)
        print("################################################")
        figaa, axaa = plt.subplots(1, 2, figsize=(10, 5))
        all_summed_mus = gt.run_simulation(simulation_cfg, axaa[0])
        all_summed_mus = np.array(all_summed_mus)
        muc = np.min(all_summed_mus)
        mus = np.max(all_summed_mus)
        print("Running inference")
        inference_cfg["muc"] = muc
        inference_cfg["mus"] = mus
        run_inference(inference_cfg)
        plot_mld_fit_and_expected(
                axaa[1],
                inference_cfg["results_dir"],
                muc, mus, inference_cfg["delta"],
                (np.log10(2*float(simulation_cfg["tree_height"])), -20),
                all_summed_mus,
                simulation_cfg["length_gene"]*simulation_cfg["n_gene_trees"],
            )
        figaa.tight_layout()
        figaa.savefig(os.path.join(inference_cfg["results_dir"], "fit_and_sim.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the gene tree simulation and the inference")
    parser.add_argument("simulation_cfg", help="Path to the simulation config file")
    parser.add_argument("inference_cfg", help="Path to the inference config file")
    args = parser.parse_args()

    with open(args.simulation_cfg, "r") as f:
        simulation_cfg = yaml.safe_load(f)
    with open(args.inference_cfg, "r") as f:
        inference_cfg = yaml.safe_load(f)

    simulate_infer(simulation_cfg, inference_cfg)

    if False:
        # debug region
        # simulation data
        gene_trees = []
        gene_tree_dir = "/home/paulimer/Documents/CoreSimul_rewrite/CoreAliSim/full_tree_rw/rw_v_rw_1.00e+04"
        res_dir = "/home/paulimer/Documents/CoreSimul_rewrite/CoreAliSim/full_tree_rw/res_rw_v_rw_1.00e+04"
        for _ in range(5000):
            gene_trees.append(TreeNode.read(f"{gene_tree_dir}/gene_tree_{_}.newick"))
        species_tree = TreeNode.read([f"(A:3,(B:2,(C:1,D:1):1):1);"])
        level = ("A", "B")
        tree_height = 2e8
        time_tree = gt.get_time_tree(species_tree, tree_height)
        sim_tau = time_tree.find(level[0]).distance(time_tree.find(level[1]))

        all_paired_mut_rates_dic = gt.get_all_pair_mutation_rate(time_tree, gene_trees)
        all_summed_mus = [(a + b)/2 for a, b in all_paired_mut_rates_dic[level]]
        muc = np.min(all_summed_mus)
        mus = np.max(all_summed_mus)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plot_mld_fit_and_expected(ax, res_dir, muc, mus, 0.82, (np.log10(sim_tau), -20), all_summed_mus, all_paired_mut_rates_dic[level], 5e6, level)
        fig.show()



        # plot fit results vs input of simulation
        res_csv = "/home/paulimer/Documents/CoreSimul_rewrite/CoreAliSim/full_tree_rw_height/all_res_fit.csv"
        res_df = pd.read_csv(res_csv)

        res_df["fit_tau"] = res_df["fit_tau"].apply(lambda x: 10**x)
        res_df["sim_tau"] = res_df["sim_tau"].apply(lambda x: 10**x)
        min_tau = res_df["fit_tau"].min()
        max_tau = res_df["fit_tau"].max()

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for rw_step_fraction, df in res_df.groupby("rw_step_fraction"):
            ax.plot(df["sim_tau"], df["fit_tau"], "o", label=rw_step_fraction)
        ax.plot([min_tau, max_tau], [min_tau, max_tau], "k--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Fitted tau")
        ax.set_ylabel("Simulated tau")
        ax.legend()
        ax.set_title("Fitted tau vs simulated tau")
        fig.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(res_csv), "fit_vs_sim.png"), dpi=300)
