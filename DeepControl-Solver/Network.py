import torch
import torch.nn as nn

import numpy as np

# from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class ModelControl(nn.Module):
    # init initialize the neural network and takes two inputs the equation and the hidden layer size
    def __init__(self, equation, dim_h):
        super(ModelControl, self).__init__()

        # We define the neural network with 4 hidden layers, the input layer has size dim x + 1 (time)
        # the hidden layers have dimension h and the output has simension dim y + dim y * dim d --> it will return Y and Z
        self.linear1 = nn.Linear(equation.dim_x + 1, dim_h)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)

        self.linear2 = nn.Linear(dim_h, dim_h)
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear2.bias)
        self.linear3 = nn.Linear(dim_h, dim_h)
        nn.init.kaiming_normal_(self.linear3.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear3.bias)
        self.linear4 = nn.Linear(dim_h, equation.dim_d)
        nn.init.kaiming_normal_(self.linear4.weight, nonlinearity='sigmoid')
        nn.init.zeros_(self.linear4.bias)



        self.mathModel = equation

    def phi(self, inpt_tmp):
        inpt_tmp = torch.relu(self.linear1(inpt_tmp))
        inpt_tmp = torch.relu(self.linear2(inpt_tmp))
        inpt_tmp = torch.relu(self.linear3(inpt_tmp))
        return torch.sigmoid(self.linear4(inpt_tmp))  # [bs,(dy*dd)] -> [bs,dy,dd]

    # this function defineds how the input is passed through the neural network
    # we need the total time steps, the current step, the input, the jump and the type of the input ( to know which neural network to use)
    def forward(self, batch_size):

        # def normalize(x):
        #     xmax = x.max(dim=0).values
        #     xmin = x.min(dim=0).values
        #     return (x - xmin) / (xmax - xmin)
        #
        #def standardize(x):
        #     mean = torch.mean(x, dim=0)
        #     sd = torch.std(x, dim=0)
        #     return (x - mean) / sd




        x_tensor = self.mathModel.x_0 + torch.zeros(batch_size, self.mathModel.N, self.mathModel.dim_x, device = device)

        x = x_tensor[:, 0, :].clone()  # x is the initial value of the process, it is a vector of size (batch_size, dim_x)

        h_V = -torch.log(1 - x[:, :self.mathModel.dim_d]) / self.mathModel.s[:,0]
        h_D = x[:, self.mathModel.dim_d:self.mathModel.dim_d + 1] / self.mathModel.d[0]
        h = torch.cat((h_V, h_D), dim=-1)

        C_R = x[:, -self.mathModel.dim_d:]

        poisson = torch.zeros(batch_size, 1, device=device)  # [bs,1]


        control_tensor = torch.zeros(batch_size, self.mathModel.N, self.mathModel.dim_d, device=device)  # [bs,N,dim_d]

        f = torch.zeros(batch_size,1, device=device)

        poiss = torch.zeros(batch_size, self.mathModel.dim_j, device=device)  # [bs,dim_j]

        for n in range(self.mathModel.N-1):
            delta_t = self.mathModel.dt



            time = torch.ones(batch_size, 1, device=device) * delta_t * n


            # the first feature of xnor is kept and concatenated with the time (?)
            #inpt = torch.cat((x_nor, time), dim=-1)
            inpt = torch.cat((x, time), dim=-1)

            control = self.phi(inpt)
            control_tensor[:, n, :] = control

            jumps = self.mathModel.jumps(batch_size)
            #poiss += (jumps_i != 0).float()

            f += self.mathModel.f(delta_t * n, x)*delta_t

            h = self.mathModel.step_forward(n, h, jumps)

            C_R += self.mathModel.dCR(n, jumps, x, control)

            vd = self.mathModel.h_to_vd(n, h)

            x = torch.cat((vd, C_R), dim=-1)

            x_tensor[:, n + 1, :] = x

        J = f + self.mathModel.g(x, poiss)




        return control_tensor, x_tensor, J.unsqueeze(-1)


