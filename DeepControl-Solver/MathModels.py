import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# We implement a solver for Forward-Backward SDE with jumps.
# The model is trained by using the Euler-Maruyama method for the diffusion part and the Monte Carlo method for the jump part.



class EnergyExplicit():
    def __init__(self, T, lam, control_parameter, jump_size, sig, N, x0, dim_x, dim_y, dim_d, dim_j, eq_type, xi, d, r,
                 a, b, c, impulse_cost_rate, impulse_cost_fixed=0.0, s=1.0):
        self.T = T
        self.lam = torch.tensor(lam, device=device)  # jump intensity
        self.control_parameter = control_parameter
        self.jump_size = torch.tensor(jump_size, device=device)
        self.sig = torch.tensor(sig, device=device)
        self.N = N
        self.dt = T / N
        self.x_0 = torch.tensor(x0, device=device)
        self.times = torch.linspace(0., T, N)
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_d = dim_d
        self.dim_j = dim_j
        if self.dim_d != self.dim_j-1:
            raise ValueError("dim_d should be equal to dim_j-1 for the explicit model.")
        self.eq_type = eq_type
        self.xi = torch.tensor(xi, device=device)
        self.r = r
        self.impulse_cost_rate = impulse_cost_rate
        self.impulse_cost_fixed = impulse_cost_fixed
        # quick computation of seasonality
        t_grid = torch.linspace(0, T, N, device=device)
        a =  torch.repeat_interleave(a.unsqueeze(-1), N, dim=-1)
        b = torch.repeat_interleave(b.unsqueeze(-1), N, dim=-1)
        c = torch.repeat_interleave(c.unsqueeze(-1), N, dim=-1)
        self.s = a + b * torch.sin(2 * torch.pi * t_grid) + c * torch.cos(2 * np.pi * t_grid)
        self.d = d * torch.ones(N, device=device)

    def jumps_slow(self, batch_size):
        # returns (bs,dim_j) tensor of z*dN
        rates = self.dt * self.lam.unsqueeze(0).expand(batch_size, -1)

        # rates = torch.ones(batch_size, self.dim_j) * self.dt * self.lam
        dN = torch.poisson(rates) * self.eq_type

        jump = torch.zeros_like(dN, device=device)
        jump_v = torch.zeros(batch_size, self.dim_d, device=device)

        # TODO: Columb loop can be vectorized
        for row in range(batch_size):
            for col in range(self.dim_j):
                if dN[row, col] > 0:
                    Exp = torch.distributions.Exponential(self.jump_size[col]).sample((int(dN[row, col]),)).to(device)
                    Uni = torch.distributions.Uniform(0, self.dt).sample((int(dN[row, col]),)).to(device)
                    jump[row, col] = torch.sum(torch.exp(-self.xi[col] * (self.dt - Uni)) * Exp, dim=0)
                    if col != self.dim_j - 1:
                        jump_v[row, col] += torch.sum(1 - torch.exp(-self.sig[col, col]) * Exp)
                    else:
                        for l in range(self.dim_j - 1):
                            jump_v[row, l] += torch.sum(1 - torch.exp(- self.sig[l, col]) * Exp)

        jump_H = jump @ self.sig.T

        return jump_H, jump_v

    def jumps(self, batch_size):
        # returns (bs,dim_j) tensor of z*dN
        rates = self.dt * self.lam.unsqueeze(0).expand(batch_size, -1)

        # rates = torch.ones(batch_size, self.dim_j) * self.dt * self.lam
        dN = torch.poisson(rates) * self.eq_type

        M = int(dN.max())

        if M == 0:
            return torch.zeros((batch_size, self.dim_j), device=device), torch.zeros((batch_size, self.dim_d),
                                                                                     device=device)

        Exp = torch.distributions.Exponential(self.jump_size).sample((batch_size, M)).to(device)
        Exp = Exp.transpose(1, 2)  # (bs, dim_j, M)
        Uni = torch.distributions.Uniform(0, self.dt).sample((batch_size, self.dim_j, M)).to(device)  # (bs, dim_j, M)

        mask = torch.arange(M, device=device).expand(batch_size, self.dim_j, M) < dN.unsqueeze(-1)

        Exp = Exp * mask
        Uni = Uni * mask

        xi = self.xi.unsqueeze(0).unsqueeze(-1).expand(batch_size, self.dim_j, M)  # (bs, dim_j, M)

        jump = torch.exp(-xi * (self.dt - Uni)) * Exp

        jump = torch.sum(jump, dim=-1)
        jump_H = jump @ self.sig.T  # (bs, dim_j) aka (bs,dim_d + 1)

        sig_diag = torch.diagonal(self.sig)[:-1]  # dim_d
        sig_diag = sig_diag.unsqueeze(0).unsqueeze(-1).expand(batch_size, self.dim_d, M)  # (bs, dim_d, M)

        sig_last_col = self.sig[:-1, -1]  # dim_d
        sig_last_col = sig_last_col.unsqueeze(0).unsqueeze(-1).expand(batch_size, self.dim_d, M)  # (bs, dim_d, M)

        jump_diag = 1 - torch.exp(-sig_diag * Exp[:, :-1, :])  # (bs, dim_d, M)
        jump_last_col = 1 - torch.exp(-sig_last_col * Exp[:, -1:, :])  # (bs, dim_d, M)

        jump_v = torch.sum(jump_diag + jump_last_col, dim=-1)  # (bs, dim_d)

        return jump_H, jump_v

    def step_forward(self, i, h, jumps):
        jumps_H, jumps_V = jumps
        h_next = h * torch.exp(- self.xi * self.dt) + jumps_H
        return h_next

    def dCR(self, i, jumps, x, control):
        V = x[:, :self.dim_d]  # V is the first d columns of x
        delta_C_R = (1 - V) * jumps[1] * torch.maximum(V-control, torch.zeros_like(V))
        return 2*delta_C_R

    def h_to_vd(self, i, h):
        h_V = h[:, :self.dim_d]  # first d columns used to compute V
        h_D = h[:, self.dim_d:]  # last column used to compute D
        V = 1 - torch.exp(- self.s[:,i] * h_V)
        D = self.d[i] * h_D
        return torch.cat((V, D), dim=-1)

    def f(self, t, x, y=None, u=None, Gamma=None):
        d = self.dim_d
        x1 = x[:, :d]
        x2 = x[:, d:d + 1]
        x3 = x[:, d + 1:]
        diff = x2 - torch.sum(x1 * x3, dim=1, keepdim=True)
        return torch.maximum(diff, torch.zeros_like(diff)) * np.exp(-self.r * t)

    def g(self, x, poisson):
        d = self.dim_d
        # return torch.zeros(x.shape[0], self.dim_y, device=x.device)
        return self.impulse_cost_rate * torch.sum((x[:, d + 1:] - self.x_0[d + 1:]), dim=1,
                                                  keepdim=True)  # + self.impulse_cost_fixed * torch.sum(poisson, dim=-1, keepdim=True)

    # Impulse cost function

    def impulse_cost(self, t, x, x_next):
        jump = x_next[:, 1:] - x[:, 1:]  # check whether x_2 jumps
        cost = self.impulse_cost_rate * jump
        return cost
