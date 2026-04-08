#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import pandas as pd

RES_DIR_DIC = {
    "classic":"/home/paulimer/Documents/CoreSimul_rewrite/CoreAliSim/cherry_rw",
    "x10_interval": "/home/paulimer/Documents/CoreSimul_rewrite/CoreAliSim/cherry_rw_x10_interval",
    "/10_lowerbound": "/home/paulimer/Documents/CoreSimul_rewrite/CoreAliSim/cherry_rw_x10_lowerbound",
    "/100_lowerbound": "/home/paulimer/Documents/CoreSimul_rewrite/CoreAliSim/cherry_rw_x100_lowerbound",
    "no_lowerbound_1": "/home/paulimer/Documents/CoreSimul_rewrite/CoreAliSim/cherry_rw_no_lowerbound",
    "no_lowerbound_2": "/home/paulimer/Documents/CoreSimul_rewrite/CoreAliSim/cherry_rw_no2_lowerbound"
}

def get_results(param):
    small_step_df = pd.read_csv(os.path.join(RES_DIR_DIC[param], "res_rw_v_height_1e+08_rw_1e+03/fitted_params.csv"), header=None)
    medium_step_df = pd.read_csv(os.path.join(RES_DIR_DIC[param], "res_rw_v_height_1e+08_rw_1e+04/fitted_params.csv"), header=None)
    big_step_df = pd.read_csv(os.path.join(RES_DIR_DIC[param], "res_rw_v_height_1e+08_rw_1e+05/fitted_params.csv"), header=None)
    small_step_inf = small_step_df[2].values[0]
    medium_step_inf = medium_step_df[2].values[0]
    big_step_inf = big_step_df[2].values[0]
    return small_step_inf, medium_step_inf, big_step_inf

res_list = []
for key in RES_DIR_DIC:
    res = list(get_results(key))
    res.append(key)
    res_list.append(res)

df = pd.DataFrame(res_list)
df.rename(columns={0:"small", 1:"medium", 2:"large", 3:"param"}, inplace=True)
df.to_csv("inf_tau_by_case.csv")
