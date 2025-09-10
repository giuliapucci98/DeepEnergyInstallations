import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json

import time as cas

from MathModels import Energy
from Solver import Train



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = "state_dicts/"

new_folder_flag = True

new_folder = "control/"


if new_folder_flag:
    path = new_folder + path
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(new_folder + "/Graphs"):
        os.makedirs(new_folder + "/Graphs")
        #path = new_folder + path
    graph_path = new_folder + "/Graphs/"
ref_flag = False


dim_x, dim_y, dim_d,dim_j, dim_h, N, itr, batch_size, MC_size, lr = 3, 1, 1, 2, 256, 50, 50, 2000, 5000, 0.001
x0, T, multiplyer = 0.0, 1.0, 10

eq_type=1

dict_parameters = {'T': 1, 'x0': [0.4, 0.7, 0.0], 'N': 50, 'lam': [5.0,5.0], 'control_parameter':None, 'jump_size': [0.5,1.0], 'sig': [[0.2, 0.2], [0.0, 0.05]], 'xi':[0.2,0.2], 'd':0.7, 'r':0.4, 'impulse_cost_rate': 0.1, 'impulse_cost_fixed': 0.0, 's': 1.0, 'control_min': 0.0, 'control_max': 0.7, 'number_of_impulses': 2}
T, x0, N, lam, control_parameter, jump_size, sig, xi, d, r, impulse_cost_rate, impulse_cost_fixed, s, control_min, control_max, number_of_impulses = dict_parameters.values()


mathModel = Energy(T, lam, control_parameter, jump_size, sig, N, x0, dim_x, dim_y, dim_d, dim_j, eq_type, xi, d, r, impulse_cost_rate, impulse_cost_fixed, s)

loss_min = 2
loss_mins = []
while loss_min > 0.25:
#for i in range(20):
    train_class = Train(mathModel, dim_h)

    losses, control, x = train_class.train(batch_size, itr, lr)

    loss_min = np.min(losses)
    print(loss_min)
    loss_mins.append(loss_min)

    plt.plot(losses)
    plt.savefig(graph_path + "loss.png")
    plt.show()


t = torch.linspace(0, T, N)
x = x.detach().numpy()
control = control.detach().numpy()

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
