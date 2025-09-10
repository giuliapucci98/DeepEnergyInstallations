import torch
import torch.nn as nn

import numpy as np

# from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# We implement a solver for Forward-Backward SDE with jumps.
# The model is trained by using the Euler-Maruyama method for the diffusion part and the Monte Carlo method for the jump part.



class ModelY(nn.Module):
    # init initialize the neural network and takes two inputs the equation and the hidden layer size
    def __init__(self, equation, dim_h):
        super(ModelY, self).__init__()

        # We define the neural network with 4 hidden layers, the input layer has size dim x + 1 (time)
        # the hidden layers have dimension h and the output has simension dim y + dim y * dim d --> it will return Y and Z
        self.linear1 = nn.Linear(equation.dim_x + 1, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, dim_h)
        self.linear4 = nn.Linear(dim_h, equation.dim_y)



        self.mathModel = equation

    # this function defineds how the input is passed through the neural network
    # we need the total time steps, the current step, the input, the jump and the type of the input ( to know which neural network to use)
    def forward(self, n, x):

        # def normalize(x):
        #     xmax = x.max(dim=0).values
        #     xmin = x.min(dim=0).values
        #     return (x - xmin) / (xmax - xmin)
        #
        #def standardize(x):
        #     mean = torch.mean(x, dim=0)
        #     sd = torch.std(x, dim=0)
        #     return (x - mean) / sd

        def phi(inpt_tmp):
            inpt_tmp = torch.tanh(self.linear1(inpt_tmp))
            inpt_tmp = torch.tanh(self.linear2(inpt_tmp))
            inpt_tmp = torch.tanh(self.linear3(inpt_tmp))
            return self.linear4(inpt_tmp)  # [bs,(dy*dd)] -> [bs,dy,dd]



        delta_t = self.mathModel.dt

        x_nor = x
        #if n != 0:
             # x_nor = normalize(x)
        #    x_nor = standardize(x_nor)



        time = torch.ones_like(x_nor) * delta_t * n


        # the first feature of xnor is kept and concatenated with the time (?)
        #inpt = torch.cat((x_nor, time), dim=-1)
        inpt = torch.cat((x_nor, torch.ones((x.size()[0], 1), device=device) * delta_t * n), 1)

        yz = phi(inpt)

        y = yz[:, :self.mathModel.dim_y].clone()

        return y



class ModelU(nn.Module):
    # init initialize the neural network and takes two inputs the equation and the hidden layer size
    def __init__(self, equation, dim_h):
        super(ModelU, self).__init__()

        # We define the neural network with 4 hidden layers, the input layer has size dim x + 2 (time)
        self.linear1j = nn.Linear(equation.dim_x + equation.dim_j + 1, dim_h)
        self.linear2j = nn.Linear(dim_h, dim_h)
        self.linear3j = nn.Linear(dim_h, dim_h)
        self.linear4j = nn.Linear(dim_h, equation.dim_y)

        self.mathModel = equation

    # this function defineds how the input is passed through the neural network
    # we need the total time steps, the current step, the input, the jump and the type of the input ( to know which neural network to use)
    def forward(self, n, x, jumps):

        # def normalize(x):
        #     xmax = x.max(dim=0).values
        #     xmin = x.min(dim=0).values
        #     return (x - xmin) / (xmax - xmin)
        #
        # def standardize(x):
        #     mean = torch.mean(x, dim=0)
        #     sd = torch.std(x, dim=0)
        #     return (x - mean) / sd



        def phi_j(inpt_tmp):
            inpt_tmp = torch.tanh(self.linear1j(inpt_tmp))
            inpt_tmp = torch.tanh(self.linear2j(inpt_tmp))
            inpt_tmp = torch.tanh(self.linear3j(inpt_tmp))
            return self.linear4j(inpt_tmp)  # [bs,(dy*dd)] -> [bs,dy,dd]

        delta_t = self.mathModel.dt

        x_nor = x
        # if n != 0:
        #     # x_nor = normalize(x)
        #     x_nor = standardize(x_nor)

        time = torch.ones(x_nor.shape[:-1] + (1,), device=x.device, dtype=x.dtype) * delta_t * n
        inpt_j = torch.cat((x_nor, torch.exp(jumps), time), dim=-1)


        #inpt_j = torch.cat((x_nor, torch.exp(jumps), time), dim=-1)
        #inpt_j = torch.cat((x_nor, torch.exp(jumps),  torch.ones(x.size()[0], 1, device=device) * delta_t * n), dim=-1)


        u = phi_j(inpt_j).clone()

        return u