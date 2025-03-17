#!/usr/bin/env python3

import argparse
import os
import sys

import yaml

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
mosaic_method_path = "/home/paulimer/Documents/test_florian/"
sys.path.append(mosaic_method_path)

import gene_trees as gt
from main_inference import run_inference


def inf_add_suffix(inference_cfg, suffix):
    for key in inference_cfg:
        if key.endswith("dir"):
            inference_cfg[key] = inference_cfg[key] + suffix


def simulate_infer(simulation_cfg, inference_cfg):
    if type(simulation_cfg["rw_step_fraction"]) is list:
        rw_step_fraction_list = simulation_cfg["rw_step_fraction"].copy()
        for rw_step_fraction in rw_step_fraction_list:
            simulation_cfg_copy = simulation_cfg.copy()
            inference_cfg_copy = inference_cfg.copy()
            simulation_cfg_copy["rw_step_fraction"] = rw_step_fraction
            suffix = f"_rw_{float(rw_step_fraction):.2e}"
            inf_add_suffix(inference_cfg_copy, suffix)
            inf_add_suffix(simulation_cfg_copy, suffix)
            print("################################################")
            print("Simulating gene trees and sequences Random walk " + str(rw_step_fraction))
            print("################################################")
            muc, mus = gt.run_simulation(simulation_cfg_copy)
            print("Running inference")
            inference_cfg["muc"] = muc
            inference_cfg["mus"] = mus
            run_inference(inference_cfg_copy)

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
        muc, mus = gt.run_simulation(simulation_cfg)
        print("Running inference")
        inference_cfg["muc"] = muc
        inference_cfg["mus"] = mus
        run_inference(inference_cfg)


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
