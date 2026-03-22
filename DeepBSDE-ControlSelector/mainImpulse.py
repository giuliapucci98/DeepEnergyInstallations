import torch
import numpy as np
import os
from pathlib import Path
import json
import time

import time as cas

from MathModels import EnergyExplicit
from Solver import Result
from impulseSelection import ImpulseSelection


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


plot = False
train= True

#base_dir = Path(__file__).resolve().parent
base_dir = Path.cwd()
path = str(base_dir / "state_dicts") + os.sep

new_folder_flag = True

from datetime import datetime
timestamp = datetime.now().strftime("%m%d_%H%M")
new_folder = base_dir / timestamp

if new_folder_flag:
    path = str(Path(new_folder) / "state_dicts") + os.sep
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(str(Path(new_folder) / "Graphs")):
        os.makedirs(str(Path(new_folder) / "Graphs"))
        #path = new_folder + path
    graph_path = str(Path(new_folder) / "Graphs") + os.sep
ref_flag = False

dim_y, dim_d, dim_h =  1, 3, 50
dim_x = 2*dim_d + 1
dim_j = dim_d + 1
itr, batch_size, MC_size, lr =  200, 10, 5000, 0.0001
x0, T, multiplyer = 0.0, 1.0, 20


a = 365

lam_full = torch.tensor([0.0271, 0.0538, 0.1383, 0.896], device=device) * a
jump_size_full = torch.tensor([0.2328, 0.3282, 0.5260, 1.3387], device=device)
xi_full = torch.tensor([0.8977, 0.7589, 0.8513, 0.6539], device=device) * a

sig_full = torch.tensor([
    [1.0305, 0.0,   0.0,   1.1593],
    [0.0,    0.6101,0.0,   0.9792],
    [0.0,    0.0,   1.1674,0.8247],
    [0.0,    0.0,   0.0,   0.9781]
], device=device)


# Construct reduced-dimension parameters
lam = torch.cat([lam_full[:dim_d], lam_full[-1:]])           # [λ1, λ2, λ3, λ4]
xi = torch.cat([xi_full[:dim_d], xi_full[-1:]])             # [ξ1, ξ2, ξ3, ξ4]
jump_size = torch.cat([jump_size_full[:dim_d], jump_size_full[-1:]])  # [J1, J2, J3, J4]

sig = torch.zeros((dim_d + 1, dim_d + 1), device=device)  # dim_j x dim_j
sig[:dim_d, :dim_d] = torch.diag(torch.diag(sig_full[:dim_d, :dim_d]))
sig[:dim_d, dim_d] = sig_full[:dim_d, -1]
sig[dim_d, dim_d] = sig_full[-1, -1]

eq_type = 1

# Updated parameter dictionary
dict_parameters = {
    'T': T,
    'N': 50,
    'lam': lam,
    'control_parameter': None,
    'jump_size': jump_size,
    'sig': sig,
    'xi': xi,
    'd': 0.7,
    'r': 0.4,
    'impulse_cost_rate': 0.01,
    'impulse_cost_fixed': 0.0,
    's': 1.0,
    'control_min': 0.0,
    'control_max': 1.0,
    'number_of_impulses': 20,
    'a': torch.tensor([0.1721, 0.2848, 0.2294], device=device),  # dimension 1
    'b': torch.tensor([-0.0491, -0.0405, -0.0322], device=device),
    'c': torch.tensor([-0.0804, -0.0956, -0.1226], device=device)
}

x0 = x0*torch.ones(2*dim_d + 1, device=device)


import wandb
wandb.init(
    project="energy-control",
    config=dict_parameters
)


#T, x0, N, lam, control_parameter, jump_size, sig, xi, d, r, impulse_cost_rate, impulse_cost_fixed, s, control_min, control_max, number_of_impulses = dict_parameters.values()
T, N, lam, control_parameter, jump_size, sig, xi, d, r, impulse_cost_rate, impulse_cost_fixed, s, control_min, control_max, number_of_impulses, a, b, c = dict_parameters.values()


mathModel = EnergyExplicit(T, lam, control_parameter, jump_size, sig, N, x0, dim_x, dim_y, dim_d, dim_j, eq_type, xi, d, r, a, b, c, impulse_cost_rate, impulse_cost_fixed)#, a,b,c)


impulse = ImpulseSelection(control_min, control_max, number_of_impulses, mathModel, graph_path)
start_time = time.time()
control, y0= impulse.select_impulse(dim_h, N, path, MC_size, batch_size, itr, multiplyer, lr)
total_time = time.time() - start_time
print("Best Control: " + str(control) + "Y0: " +str(y0))

total_trajectories = batch_size * itr * (N-1)

peak_memory = torch.cuda.max_memory_allocated() / 1024**3

wandb.log({
    "total_training_time_sec": total_time,
    "total_training_time_min": total_time / 60,
    "total_gradient_updates": itr * (N-1),
    "batch_size": batch_size,
    "total_simulated_trajectories": total_trajectories,
    "peak_gpu_memory_GB": peak_memory,
    "final_objective": y0
})

print(os.getcwd())

#Opt. control for evaluation, put in here the control you want to evaluate
#mathModel.control_parameter = 1.5789473684210527

#opt_path = path + f"control_{mathModel.control_parameter}/"
opt_path = path + f"control_{control}/"
mathModel.control_parameter = control

eval_size = 100
results = Result(mathModel, dim_h)
x, jumps, poiss, h = results.gen_x(eval_size, N)
y, u = results.predict(N, eval_size, x, opt_path, jumps, poiss)

t = np.linspace(0, T, N)

n_of_plots = 3

if train:
    with torch.no_grad():

        n_samples = min(5, eval_size)
        sample_indices = torch.randperm(eval_size)[:n_samples]

        x_cpu = x.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()


    for i, idx in enumerate(sample_indices):


        wandb.log({
            f"forward/x1_sample_{i}": wandb.plot.line_series(
                xs=t,
                ys=[x_cpu[idx, 0, :]],
                keys=["x1(t)"],
                title=f"Sample {idx}: x1(t) = V(t)",
                xname="Time"
            )
        })


    for i, idx in enumerate(sample_indices):

        wandb.log({
            f"forward/x2_sample_{i}": wandb.plot.line_series(
                xs=t,
                ys=[x_cpu[idx, dim_d, :]],
                keys=["D(t)"],
                title=f"Sample {idx}: D(t)",
                xname="Time"
            )
        })

    for i, idx in enumerate(sample_indices):

        wandb.log({
            f"orward/x3_sample_{i}": wandb.plot.line_series(
                xs=t,
                ys=[x_cpu[idx, -1, :]],
                keys=["C_R(t)"],
                title=f"Sample {idx}: C_R(t)",
                xname="Time"
            )
        })


    for i, idx in enumerate(sample_indices):

        ys = [x_cpu[idx, j, :] for j in range(dim_d)]
        keys = [f"V{j+1}" for j in range(dim_d)]

        wandb.log({
            f"V_components_sample_{i}": wandb.plot.line_series(
                xs=t,
                ys=ys,
                keys=keys,
                title=f"Sample {idx}: V components",
                xname="Time"
            )
        })


    for i, idx in enumerate(sample_indices):

        wandb.log({
            f"backward/Y_sample_{i}": wandb.plot.line_series(
                xs=t,
                ys=[y_cpu[idx, 0, :]],
                keys=["Y(t)"],
                title=f"Sample {idx}: Y(t)",
                xname="Time"
            )
        })

print(f'Run "{new_folder.name}" is finished.')

if plot:
    import matplotlib.pyplot as plt
    t_grid = np.linspace(0, T, N)
    fig1, axs1 = plt.subplots(2, 1, figsize=(10, 8))

    for i in range(batch_size):
        axs1[0].plot(t_grid, h[i, 0, :].numpy())
    axs1[0].set_title("h1(t)")
    axs1[0].grid(True)

    for i in range(batch_size):
        axs1[1].plot(t_grid, h[i, 1, :].numpy())
    axs1[1].set_title("h2(t)")
    axs1[1].grid(True)

    plt.tight_layout()
    plt.show()

    # x shape: [batch_size, 2*dim_d + 1, n_time]
    # layout:
    # top row: V (dim_d subplots)
    # middle row: D (1 subplot, spanning)
    # bottom row: C_R (dim_d subplots)
    fig, axs = plt.subplots(3, dim_d, figsize=(4 * dim_d, 10))

    if dim_d == 1:
        axs = axs.reshape(3, 1)

    for j in range(dim_d):
        for i in range(batch_size):
            axs[0, j].plot(t_grid, x[i, j, :].numpy())
        axs[0, j].set_title(f"V{j + 1}(t)")
        axs[0, j].grid(True)

    for i in range(batch_size):
        axs[1, 0].plot(t_grid, x[i, dim_d, :].numpy())
    axs[1, 0].set_title("D(t)")
    axs[1, 0].grid(True)
    # If dim_d > 1, hide unused middle subplots
    for j in range(1, dim_d):
        axs[1, j].axis('off')

    for j in range(dim_d):
        for i in range(batch_size):
            axs[2, j].plot(t_grid, x[i, dim_d + 1 + j, :].numpy())
        axs[2, j].set_title(f"C_R{j + 1}(t)")
        axs[2, j].grid(True)

    plt.tight_layout()
    plt.show()

    y_cpu = y.detach().cpu().numpy()
    fig2, axs2 = plt.subplots(1, 1, figsize=(10, 6))
    for i in range(batch_size):
        axs2.plot(t_grid, y_cpu[i, 0, :])
    axs2.set_title("Y(t)")
    axs2.grid(True)
    plt.tight_layout()
    plt.show()

