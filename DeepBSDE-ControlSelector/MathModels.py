import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# We implement a solver for Forward-Backward SDE with jumps.
# The model is trained by using the Euler-Maruyama method for the diffusion part and the Monte Carlo method for the jump part.


# TODO: So far  for one dimensional. Extende to multi-dimensional
#TODO: Add impulse fixed cost in main
class Energy():
    def __init__(self, T, lam, control_parameter, jump_size, sig, N, x0, dim_x, dim_y, dim_d, dim_j, eq_type, xi, d, r, impulse_cost_rate, impulse_cost_fixed=0.0, s=1.0):
        self.T = T
        self.lam = torch.tensor(lam)  # jump intensity
        self.control_parameter = control_parameter
        self.jump_size = torch.tensor(jump_size)
        self.sig = torch.tensor(sig)
        self.s = s #so far scalar, but can be extended to vector if d>1.
        self.N = N  # number of time steps
        self.dt = T / N
        self.x_0 = torch.tensor(x0)
        self.times = torch.linspace(0., T, N)
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_d = dim_d
        self.dim_j = dim_j
        self.eq_type = eq_type
        self.xi = torch.tensor(xi)
        self.d = d
        self.r = r
        self.impulse_cost_rate = impulse_cost_rate
        self.impulse_cost_fixed = impulse_cost_fixed

    # Analytic approach

    def drift(self, t, x):
        x1, x2, x3 = torch.split(x, 1, dim=-1)
        drift1 = (1 - x1) * (self.xi[0] * torch.log(1 - x1) + self.lam[0] * self.sig[0,0] * self.s / (self.sig[0,0] * self.s + self.jump_size[0])  + self.lam[1] * self.sig[0,1] * self.s / (self.sig[0,1] * self.s + self.jump_size[1]))
        drift2 = (self.lam[1]*self.sig[1,1]/self.jump_size[1] - self.xi[1] *x2) * self.d
        return torch.cat((drift1, drift2, torch.zeros_like(x3)),dim=-1)

    def compensator(self, t, x):
        x1, x2, x3 = torch.split(x, 1, dim=-1)
        compensator1 = (1 - x1) * (self.lam[0] * self.sig[0,0] / (self.sig[0,0] + self.jump_size[0]) + self.lam[1] * self.sig[0,1] / (self.sig[0,1] + self.jump_size[1]))
        compensator2 = self.sig[1,1] * self.d * self.lam[1] / self.jump_size[1]
        return torch.cat((compensator1, compensator2*torch.ones_like(x2), torch.zeros_like(x3)),dim=-1)

    def gamma(self, t, x, jump):
        '''
        makes time, x in dim_x and jump in dim_j
        maps to dim_x
        result shape should be (batch_size, dim_x)
        '''
        x1, x2, x3 = torch.split(x, 1, dim=-1)
        gamma1 = (1-x1) * torch.sum((1 - torch.exp(-self.sig[0] * jump)), dim=-1, keepdim=True)
        gamma2 = self.d * self.sig[1,1]*jump[:,1:]
        #gamma3 = torch.maximum(self.control_parameter - x1, torch.zeros_like(x1)) * gamma1*2
        gamma3 = torch.maximum(x2 - self.control_parameter, torch.zeros_like(x2)) * gamma2
        return torch.cat((gamma1, gamma2, gamma3), dim=-1)


    def jumps(self, batch_size):
        # returns (bs,dim_j) tensor of z*dN
        rates = self.dt * self.lam.unsqueeze(0).expand(batch_size, -1)

        # rates = torch.ones(batch_size, self.dim_j) * self.dt * self.lam
        dN = torch.poisson(rates) * self.eq_type
        Exp = torch.distributions.Exponential(self.jump_size).sample((batch_size,))

        return Exp * dN




    def step_forward(self, i, x, jumps):
        x_next = x + (self.drift(self.dt * i, x) - self.compensator(self.dt * i, x))*self.dt + self.gamma(self.dt * i, x, jumps)
        return x_next

    def f(self, t, x, y, u, Gamma):
        x1, x2, x3 = torch.split(x, 1, dim=-1)
        diff = x2 - x1*x3
        return torch.maximum(diff,torch.zeros_like(diff))*np.exp(-self.r * t)

    def g(self, x, poisson):
        #return torch.zeros(x.shape[0], self.dim_y, device=x.device)
        return self.impulse_cost_rate * (x[:,-1:] - self.x_0[-1]) + self.impulse_cost_fixed * torch.sum(poisson, dim=-1, keepdim=True)

    # Impulse cost function

    def impulse_cost(self, t, x,  x_next):
        jump = x_next[:, 1:] - x[:,1:] # check whether x_2 jumps
        cost = self.impulse_cost_rate * jump
        return cost
