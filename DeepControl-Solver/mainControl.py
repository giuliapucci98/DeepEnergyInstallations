import torch
import numpy as np
import os
from pathlib import Path
import json


import wandb

import time as cas


from MathModels import EnergyExplicit
from Solver import Train


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


train= False
#train = True

#base_dir = Path(__file__).resolve().parent
base_dir = Path.cwd()
path = str(base_dir / "state_dicts") + os.sep

new_folder_flag = True

new_folder = str(base_dir / "control09_03")

if new_folder_flag:
    path = str(Path(new_folder) / "state_dicts") + os.sep
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(str(Path(new_folder) / "Graphs")):
        os.makedirs(str(Path(new_folder) / "Graphs"))
        #path = new_folder + path
    graph_path = str(Path(new_folder) / "Graphs") + os.sep
ref_flag = False

dim_y, dim_d, dim_h =  1, 3, 256
dim_x = 2*dim_d + 1
dim_j = dim_d + 1
itr, batch_size, MC_size, lr =  400, 2000, 5000, 0.001
x0, T, multiplyer = 0.0, 1.0, 20

n_runs = 20


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
    'control_parameter': 1.0,
    'jump_size': jump_size,
    'sig': sig,
    'xi': xi,
    'd': 0.7,
    'r': 0.4,
    'impulse_cost_rate': 0.01,
    'impulse_cost_fixed': 0.0,
    's': 1.0,
    'control_min': 1.0,
    'control_max': 1.0,
    'number_of_impulses': 1,
    'a': torch.tensor([0.1721, 0.2848, 0.2294], device=device),  # dimension 1
    'b': torch.tensor([-0.0491, -0.0405, -0.0322], device=device),
    'c': torch.tensor([-0.0804, -0.0956, -0.1226], device=device)
}

x0 = x0*torch.ones(2*dim_d + 1, device=device)


'''
eq_type = 1
dict_parameters = {'T': 1, 'x0': [0.4, 0.7, 0.0], 'N': 50, 
                   'lam': [5.0,5.0], 'control_parameter':1.58,
                   'jump_size': [0.5,1.0], 'sig': [[0.2, 0.2], [0.0, 0.05]],
                     'xi':[0.2,0.2], 'd':0.7, 'r':0.4, 'impulse_cost_rate': 0.1,
                       'impulse_cost_fixed': 0.0, 's': 1.0, 'control_min': 1.58, 
                       'control_max': 3.0, 'number_of_impulses': 1}

'''

'''
wandb.init(
    project="energy-control",
    config=dict_parameters
)
'''
#T, x0, N, lam, control_parameter, jump_size, sig, xi, d, r, impulse_cost_rate, impulse_cost_fixed, s, control_min, control_max, number_of_impulses = dict_parameters.values()
T, N, lam, control_parameter, jump_size, sig, xi, d, r, impulse_cost_rate, impulse_cost_fixed, s, control_min, control_max, number_of_impulses, a, b, c = dict_parameters.values()




mathModel = EnergyExplicit(T, lam, control_parameter, jump_size, sig, N, x0, dim_x, dim_y, dim_d, dim_j, eq_type, xi, d, r, a, b, c, impulse_cost_rate, impulse_cost_fixed)#, a,b,c)

#mathModel = EnergyExplicit(T, lam, control_parameter, jump_size, sig, N, x0, dim_x, dim_y, dim_d, dim_j, eq_type, xi, d, r, impulse_cost_rate, impulse_cost_fixed, s)

loss_min = 2
loss_mins = []
#while loss_min > 0.1:

peak_memories = []

if train:

    import wandb

    wandb.init(
    project="energy-control-control",
    config=dict_parameters
)
    start = cas.time()
    peak_memories = []

    for i in range(n_runs):
        torch.cuda.reset_peak_memory_stats()  # reset peak memory for this run
        train_class = Train(mathModel, dim_h)

        # Save initial model state
        torch.save(train_class.model.state_dict(), path + "state_dict_" + str(i))

        # Track start time for this run
        start_run = cas.time()

        losses, control, x = train_class.train(batch_size, itr, lr)

        # Track peak GPU memory
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_memories.append(peak_mem)

        # Log losses per iteration with elapsed time to WandB
        for step, loss in enumerate(losses):
            elapsed_time = cas.time() - start_run
            wandb.log({
                f"loss/run_{i}": loss,
                f"time/run_{i}": elapsed_time,
                "iteration": step
            })

        # Log final loss and peak memory to WandB
        wandb.log({
            f"final_loss/run_{i}": losses[-1],
            f"peak_mem/run_{i}": peak_mem
        })

        # Save losses locally
        with open(path + str(i) + "_loss.json", 'w') as f:
            json.dump(losses, f, indent=2)

        print("Peak GPU memory usage per run (GB):", peak_memories)

else:
    import matplotlib.pyplot as plt

    losses = []
    losses_last = []
    loss_min = 2
    indx_min = 0


    for i in range(n_runs):
        with open(path + str(i) + "_loss.json", 'r') as f:
            loss = json.load(f)

        losses.append(loss)
        losses_last.append(loss[-1])
        print(f"Run {i}: Final Loss = {loss[-1]:.4f}")

        if loss[-1] < loss_min:
            loss_min = loss[-1]
            indx_min = i

    print("index min:", indx_min)
    train_class = Train(mathModel, dim_h)
    train_class.model.load_state_dict(torch.load(path + "state_dict_" + str(indx_min), map_location=torch.device('cpu')))
    train_class.model.eval()
    control, x, J = train_class.model(batch_size)


    #########################
    # Plots


    plt.plot(losses[indx_min][5:])
    plt.savefig(graph_path + f"loss_{i}.png")
    plt.show()


    t = torch.linspace(0, T, N)
    x = x.detach().numpy()
    control = control.detach().numpy()


    t_grid = np.linspace(0, T, N)

    fig, axs = plt.subplots(3, dim_d, figsize=(4 * dim_d, 10))

    plot_size = 10

    if dim_d == 1:
        axs = axs.reshape(3, 1)

    for j in range(dim_d):
        for i in range(plot_size):
            axs[0, j].plot(t_grid, x[i, :, j])
        axs[0, j].set_title(f"$V_{j + 1}(t)$")
        axs[0, j].grid(True)

    for i in range(plot_size):
        axs[1, 0].plot(t_grid, x[i, :, dim_d])
    axs[1, 0].set_title("D(t)")
    axs[1, 0].grid(True)
    # If dim_d > 1, hide unused middle subplots
    for j in range(1, dim_d):
        axs[1, j].axis('off')

    for j in range(dim_d):
        for i in range(plot_size):
            axs[2, j].plot(t_grid, x[i, :, dim_d + 1 + j])
        axs[2, j].set_title(f"$C_R${j + 1}(t)")
        axs[2, j].grid(True)

    plt.tight_layout()
    plt.savefig(graph_path + f"plot{indx_min}.png")
    plt.show()

    # Create a 3 x dim_d grid of subplots
    fig, axs = plt.subplots(1, dim_d, figsize=(10, 4))

    # Plot control trajectories
    for j in range(dim_d):
        for i in range(plot_size):
            axs[j].plot(t_grid[:-1], control[i, :-1, j])  # first row, control
        axs[j].set_title(f"$Control_{j + 1}(t)$")
        axs[j].grid(True)

    plt.tight_layout()
    plt.savefig(graph_path + f"plot_control{indx_min}.png")
    plt.show()

#####################################################################


'''
samples=20
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        # Plot X process samples
for i in range(samples):
    axs[0].plot(t, x[i, :, 0], label=f"Sample {i}")
    axs[0].set_title("V(t)")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("X1")
    axs[0].grid(True)

# Plot Y approximation samples
for i in range(samples):
    axs[1].plot(t, x[i,:,1], label=f"Sample {i}")
    axs[1].set_title("D(t)")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("X2")
    axs[1].grid(True)

for i in range(samples):
    axs[2].plot(t, x[i, :, -1], label=f"Sample {i}")
    axs[2].set_title("C_R(t)")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("X3")
    axs[2].grid(True)


plt.tight_layout()
plt.savefig(graph_path + "state.png")
plt.show()



fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        # Plot X process samples
for i in range(samples):
    axs[0].plot(t[:-1], control[i, :-1, 0], label=f"Sample {i}")
    axs[0].set_title("control1(t)")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Control 1")
    axs[0].grid(True)

# Plot Y approximation samples
for i in range(samples):
    axs[1].plot(t[:-1], control[i, :-1, 1], label=f"Sample {i}")
    axs[1].set_title("control2(t)")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Control 2")
    axs[1].grid(True)


plt.tight_layout()
plt.savefig(graph_path + "control.png")
plt.show()


# Plot control's dependence on state
x1 = torch.linspace(0.3, 1.0, 100).unsqueeze(-1)
x2 = torch.linspace(0.4, 1.2, 100).unsqueeze(-1)
x3 = torch.linspace(0.4, 1.3, 100).unsqueeze(-1)
t_tensor = torch.linspace(0, T, 100).unsqueeze(-1)

x1def = torch.ones_like(x1)*0.7
x2def = torch.ones_like(x2)*1.05
x3def = torch.ones_like(x3)*0.5

tdef = torch.ones_like(x3)*0.2

train_class.model.eval()

fig, axs = plt.subplots(4, 2, figsize=(10, 8))


axs[0,0].plot(x1[:,0],(train_class.model.phi(torch.cat((x1,x2def,x3def,tdef), dim=-1))[:,0]).detach().numpy())
axs[0,0].set_title("Control 1 as function of X1")
axs[0,0].set_xlabel("X1")
axs[0,0].set_ylabel("Control 1")
axs[0,0].grid(True)

axs[0,1].plot(x1[:,0],train_class.model.phi(torch.cat((x1,x2def,x3def,tdef), dim=-1))[:,1].detach().numpy())
axs[0,1].set_title("Control 2 as function of X1")
axs[0,1].set_xlabel("X1")
axs[0,1].set_ylabel("Control 2")
axs[0,1].grid(True)

axs[1,0].plot(x2[:,0],train_class.model.phi(torch.cat((x1def,x2,x3def,tdef), dim=-1))[:,0].detach().numpy())
axs[1,0].set_title("Control 1 as function of X2")
axs[1,0].set_xlabel("X2")
axs[1,0].set_ylabel("Control 1")
axs[1,0].grid(True)

axs[1,1].plot(x2[:,0],train_class.model.phi(torch.cat((x1def,x2,x3def,tdef), dim=-1))[:,1].detach().numpy())
axs[1,1].set_title("Control 2 as function of X2")
axs[1,1].set_xlabel("X2")
axs[1,1].set_ylabel("Control 2")
axs[1,1].grid(True)

axs[2,0].plot(x3[:,0],train_class.model.phi(torch.cat((x1def,x2def,x3,tdef), dim=-1))[:,0].detach().numpy())
axs[2,0].set_title("Control 1 as function of X3")
axs[2,0].set_xlabel("X3")
axs[2,0].set_ylabel("Control 1")
axs[2,0].grid(True)

axs[2,1].plot(x3[:,0],train_class.model.phi(torch.cat((x1def,x2def,x3,tdef), dim=-1))[:,1].detach().numpy())
axs[2,1].set_title("Control 2 as function of X3")
axs[2,1].set_xlabel("X3")
axs[2,1].set_ylabel("Control 2")
axs[2,1].grid(True)

axs[3,0].plot(t_tensor[:,0],train_class.model.phi(torch.cat((x1def,x2def,x3def,t_tensor), dim=-1))[:,0].detach().numpy())
axs[3,0].set_title("Control 1 as function of time")
axs[3,0].set_xlabel("Time")
axs[3,0].set_ylabel("Control 1")
axs[3,0].grid(True)

axs[3,1].plot(t_tensor[:,0],train_class.model.phi(torch.cat((x1def,x2def,x3def,t_tensor), dim=-1))[:,1].detach().numpy())
axs[3,1].set_title("Control 2 as function of time")
axs[3,1].set_xlabel("Time")
axs[3,1].set_ylabel("Control 2")
axs[3,1].grid(True)


plt.tight_layout()
#plt.savefig(graph_path + "control_as_function.png")
plt.show()


##################
n_points = 50
# Define grids for each variable
x1_grid = torch.linspace(0.3, 1.0, n_points).unsqueeze(-1)
x2_grid = torch.linspace(0.4, 1.2, n_points).unsqueeze(-1)
x3_grid = torch.linspace(0.4, 1.3, n_points).unsqueeze(-1)
time_grid = torch.linspace(0.0, 1.0, n_points).unsqueeze(-1)  # adjust range as needed

# Default values (adjust as needed)
x1def = torch.ones_like(x1_grid) * 0.7
x2def = torch.ones_like(x2_grid) * 1.05
x3def = torch.ones_like(x3_grid) * 0.2
timedef = torch.ones_like(time_grid) * 0.2

combinations = [
    (x1_grid, x2_grid, x3def[0], timedef[0], 'X1', 'X2'),
    (x1_grid, x3_grid, x2def[0], timedef[0], 'X1', 'X3'),
    (x1_grid, time_grid, x2def[0], x3def[0], 'X1', 'Time'),
    (x2_grid, x3_grid, x1def[0], timedef[0], 'X2', 'X3'),
    (x2_grid, time_grid, x1def[0], x3def[0], 'X2', 'Time'),
    (x3_grid, time_grid, x1def[0], x2def[0], 'X3', 'Time'),
]

fig = plt.figure(figsize=(18, 12))
for i, (grid1, grid2, fix1, fix2, label1, label2) in enumerate(combinations):
    G1, G2 = torch.meshgrid(grid1.squeeze(), grid2.squeeze(), indexing='ij')
    # Prepare input tensor according to which variables are varying
    inputs = torch.cat((
        G1.reshape(-1,1) if label1 == 'X1' else (G2.reshape(-1,1) if label2 == 'X1' else x1def[0].repeat(n_points*n_points,1)),
        G1.reshape(-1,1) if label1 == 'X2' else (G2.reshape(-1,1) if label2 == 'X2' else x2def[0].repeat(n_points*n_points,1)),
        G1.reshape(-1,1) if label1 == 'X3' else (G2.reshape(-1,1) if label2 == 'X3' else x3def[0].repeat(n_points*n_points,1)),
        G1.reshape(-1,1) if label1 == 'Time' else (G2.reshape(-1,1) if label2 == 'Time' else timedef[0].repeat(n_points*n_points,1)),
    ), dim=1)
    Z = train_class.model.phi(inputs).detach().numpy()[:,0].reshape(n_points, n_points)
    ax = fig.add_subplot(2, 3, i+1, projection='3d')
    ax.plot_surface(G1.numpy(), G2.numpy(), Z, cmap='viridis')
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_zlabel('Control 1')
    ax.set_title(f'Control 1 as function of {label1} and {label2}')

plt.tight_layout()
plt.savefig(graph_path + "control_as_function.png")
plt.show()



############
#control2


n_points = 50
# Define grids for each variable
x1_grid = torch.linspace(0.3, 1.0, n_points).unsqueeze(-1)
x2_grid = torch.linspace(0.4, 1.2, n_points).unsqueeze(-1)
x3_grid = torch.linspace(0.4, 1.3, n_points).unsqueeze(-1)
time_grid = torch.linspace(0.0, 1.0, n_points).unsqueeze(-1)  # adjust range as needed

# Default values (adjust as needed)
x1def = torch.ones_like(x1_grid) * 0.7
x2def = torch.ones_like(x2_grid) * 1.05
x3def = torch.ones_like(x3_grid) * 0.2
timedef = torch.ones_like(time_grid) * 0.2

combinations = [
    (x1_grid, x2_grid, x3def[0], timedef[0], 'X1', 'X2'),
    (x1_grid, x3_grid, x2def[0], timedef[0], 'X1', 'X3'),
    (x1_grid, time_grid, x2def[0], x3def[0], 'X1', 'Time'),
    (x2_grid, x3_grid, x1def[0], timedef[0], 'X2', 'X3'),
    (x2_grid, time_grid, x1def[0], x3def[0], 'X2', 'Time'),
    (x3_grid, time_grid, x1def[0], x2def[0], 'X3', 'Time'),
]

fig = plt.figure(figsize=(18, 12))
for i, (grid1, grid2, fix1, fix2, label1, label2) in enumerate(combinations):
    G1, G2 = torch.meshgrid(grid1.squeeze(), grid2.squeeze(), indexing='ij')
    # Prepare input tensor according to which variables are varying
    inputs = torch.cat((
        G1.reshape(-1,1) if label1 == 'X1' else (G2.reshape(-1,1) if label2 == 'X1' else x1def[0].repeat(n_points*n_points,1)),
        G1.reshape(-1,1) if label1 == 'X2' else (G2.reshape(-1,1) if label2 == 'X2' else x2def[0].repeat(n_points*n_points,1)),
        G1.reshape(-1,1) if label1 == 'X3' else (G2.reshape(-1,1) if label2 == 'X3' else x3def[0].repeat(n_points*n_points,1)),
        G1.reshape(-1,1) if label1 == 'Time' else (G2.reshape(-1,1) if label2 == 'Time' else timedef[0].repeat(n_points*n_points,1)),
    ), dim=1)
    Z = train_class.model.phi(inputs).detach().numpy()[:,1].reshape(n_points, n_points)
    ax = fig.add_subplot(2, 3, i+1, projection='3d')
    ax.plot_surface(G1.numpy(), G2.numpy(), Z, cmap='viridis')
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_zlabel('Control 2')
    ax.set_title(f'Control 2 as function of {label1} and {label2}')

plt.tight_layout()
plt.savefig(graph_path + "control2_as_function.png")
plt.show()

'''