import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json

import time as cas

from MathModels import EnergyExplicit
from MathModels import Energy
from Solver import Result
from impulseSelection import ImpulseSelection


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_dir = Path(__file__).resolve().parent
path = str(base_dir / "state_dicts") + os.sep

new_folder_flag = True

new_folder = str(base_dir / "retest_regular_control")

a = 1
print(a)


if new_folder_flag:
    path = str(Path(new_folder) / "state_dicts") + os.sep
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(str(Path(new_folder) / "Graphs")):
        os.makedirs(str(Path(new_folder) / "Graphs"))
        #path = new_folder + path
    graph_path = str(Path(new_folder) / "Graphs") + os.sep
ref_flag = False


dim_x, dim_y, dim_d,dim_j, dim_h, N, itr, batch_size, MC_size, lr = 3, 1, 1, 2, 100, 50, 3, 10, 50, 0.0001
x0, T, multiplyer = 0.0, 1.0, 5

eq_type=1

dict_parameters = {'T': 1, 'x0': [0.4, 0.7, 0.0], 'N': 50, 'lam': [0.0271*365,0.896*365], 'control_parameter':None, 'jump_size': [0.2328,1.3387], 'sig': [[1.0305, 1.1593], [0.0, 0.9781]], 'xi':[0.8977*365,0.6539*365], 'd':0.7, 'r':0.4, 'impulse_cost_rate': 0.1, 'impulse_cost_fixed': 0.0, 's': 1.0, 'control_min': 0.5, 'control_max': 3.0, 'number_of_impulses': 1}


#dict_parameters = {'T': 1, 'x0': [0.4, 0.7, 0.4], 'N': 50, 'lam': [5.0,2.0], 'control_parameter':None, 'jump_size': [3.0,1.0], 'sig': [[0.2, 0.5], [0.0, 0.05]], 'xi':[0.5,0.2], 'd':0.7, 'r':0.0, 'impulse_cost_rate': 1.0, 'impulse_cost_fixed': 0.0, 's': 1.0, 'control_min': 0.0, 'control_max': 0.7, 'number_of_impulses': 3}
T, x0, N, lam, control_parameter, jump_size, sig, xi, d, r, impulse_cost_rate, impulse_cost_fixed, s, control_min, control_max, number_of_impulses = dict_parameters.values()


mathModel = EnergyExplicit(T, lam, control_parameter, jump_size, sig, N, x0, dim_x, dim_y, dim_d, dim_j, eq_type, xi, d, r, impulse_cost_rate, impulse_cost_fixed, s)


impulse = ImpulseSelection(control_min, control_max, number_of_impulses, mathModel, graph_path)
control, y0= impulse.select_impulse(dim_h, N, path, MC_size, batch_size, itr, multiplyer, lr)
print("Best Control: " + str(control) + "Y0: " +str(y0))

print(os.getcwd())

#Opt. control for evaluation, put in here the control you want to evaluate
#mathModel.control_parameter = 1.5789473684210527

opt_path = path + f"control_{mathModel.control_parameter}/"

eval_size = 100
results = Result(mathModel, dim_h)
x, jumps, poiss = results.gen_x(eval_size, N)
y, u = results.predict(N, eval_size, x, opt_path, jumps, poiss)


t = np.linspace(0, T, N)

n_of_plots = 3

fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        # Plot X process samples
for i in range(n_of_plots):
    axs[0].plot(t, x[i, 0, :], label=f"Sample {i}")
    axs[0].set_title("V(t)")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("X1")
    axs[0].grid(True)

# Plot Y approximation samples
for i in range(n_of_plots):
    axs[1].plot(t, x[i,1,:], label=f"Sample {i}")
    axs[1].set_title("D(t)")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("X2")
    axs[1].grid(True)

for i in range(n_of_plots):
    axs[2].plot(t, x[i, -1,:], label=f"Sample {i}")
    axs[2].set_title("C_R(t)")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("X3")
    axs[2].grid(True)


plt.tight_layout()
plt.savefig(graph_path + "state.png")
plt.show()



fig, axs = plt.subplots(1, 1, figsize=(10, 8))
        # Plot X process samples
for i in range(n_of_plots):
    axs.plot(t, y[i, 0, :].detach().numpy(), label=f"Sample {i}")
    axs.set_title("Y(t)")
    axs.set_xlabel("Time")
    axs.set_ylabel("X1")
    axs.grid(True)


plt.tight_layout()
plt.savefig(graph_path + "y.png")
plt.show()
