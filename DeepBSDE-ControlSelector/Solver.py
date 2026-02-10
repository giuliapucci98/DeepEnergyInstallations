import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt


# from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from Network import ModelY
from Network import ModelU


# We implement a solver for Forward-Backward SDE with jumps.
# The model is trained by using the Euler-Maruyama method for the diffusion part and the Monte Carlo method for the jump part.


# with this class we solve the BSDE with jumps
class BSDEsolver():
    # inputes: equation, hidden layer size, model, learning rate, coefficient which we'll use to scale the learning rate
    def __init__(self, mathModel, dim_h, lr, coeff):
        # then we store the model, the equation, the hidden layer size and we set the Adam optimizer
        self.model_y = ModelY(mathModel, dim_h).to(device)
        self.model_u = ModelU(mathModel, dim_h).to(device)
        self.mathModel = mathModel
        self.optimizer = torch.optim.Adam(list(self.model_y.parameters()) + list(self.model_u.parameters()),
                                          lr * coeff)
        self.dim_h = dim_h

    # compute the oss fucntion for training the BSDE solver
    def loss(self, x, n, y_prev, y, u, compensator, Gamma, x_next, pretrain=False):
        if pretrain:
           dist = (y - y_prev).norm(2,dim=1)
        else:
            dt = self.mathModel.dt
            # compute the discrete time approximation of the bsde
            estimate = (y - self.mathModel.f(dt * n, x, y, u, Gamma) * dt  + (u - compensator) * self.mathModel.eq_type)
            # then the loss is the norm difference between y_prev and the estimated value
            dist = (y_prev - estimate).norm(2, dim=1)
        return torch.mean(dist)

    # here we simulate the forward dynamics of a jump-diffusion.
    # batchsize: number of samples to simulate in parallel
    # N: number of time steps
    # n: current time step
    def gen_forward(self, batch_size, N, n):
        dt = self.mathModel.dt
        # initialize the state x at initial time
        x = self.mathModel.x_0 + torch.zeros(batch_size, self.mathModel.dim_x, device=device,
                                             requires_grad=True).reshape(
            -1, self.mathModel.dim_x)  # [bs,dim_x]
        
        h_V = -torch.log(1 - x[:, :self.mathModel.dim_d]) / self.mathModel.s
        h_D = x[:, self.mathModel.dim_d:self.mathModel.dim_d + 1] / self.mathModel.d
        h = torch.cat((h_V, h_D), dim=-1)

        poisson = torch.zeros(batch_size, 1, device=device)  # [bs,1]


        # tensor where each element represents the Poisson jump rate for the corresponding sample and dimension (in our case the jump has dimension 1 so this is a vector)
        # self.l is the jump intensity
        if n == 0:
            # eq_type = 1 means that we are considering the jump process
            # torch.poisson(rates) generates Poisson-distribution random numbers with the given rates:
            # it indicates the number of jumps that occur in a time interval of length dt
            jumps = self.mathModel.jumps(batch_size)
            jumps_H, jumps_V = jumps
            h_next = self.mathModel.step_forward(n, h,jumps)
            x_next = self.mathModel.h_to_vdc(h_next, jumps)
            poisson += (torch.sum(jumps_H, dim=-1, keepdim=True) != 0.0).float()  # we store the jumps in the poisson variable
        else:
            for i in range(n):
                jumps = self.mathModel.jumps(batch_size)
                jumps_H, jumps_V = jumps
                h = self.mathModel.step_forward(i, h, jumps)
                poisson += (torch.sum(jumps_H, dim=-1, keepdim=True) != 0.0).float()
            jumps = self.mathModel.jumps(batch_size)
            jumps_H, jumps_V = jumps
            h_next = self.mathModel.step_forward(n, h, jumps)
            x_next = self.mathModel.h_to_vdc(h_next, jumps)
            poisson += (torch.sum(jumps_H, dim=-1, keepdim=True) != 0.0).float()

        return x, x_next, jumps_H, poisson

    # output: x: state at the current time step (n)
    # x_next: state at the next time step (n+1)
    # dN: jump at the current time step (n)

    # train at step n
    def train(self, batch_size, N, n, itr, path, multiplyer, MC_size, pretrain=False):
        loss_n = []
        if n != N - 2:
            # if we are not at the last step we load the pre-trained model and optimizer, corresponding to the step n+1
            model_y_prev = ModelY(self.mathModel, self.dim_h).to(device)
            model_y_prev.load_state_dict(torch.load(path + "Y_state_dict_" + str(n + 1)), strict=False)
            # we set the model in evaluation mode
            model_y_prev.eval()

        if n >= N - 2:  # for the last two time steps we increase the iterations
            itr_actual = multiplyer * itr
        else:
            itr_actual = itr

        for i in range(itr_actual):
            if i % 200 == 0:
                print("itr=" + str(i))
            flag = True
            flag_n = 0
            # TODO: remove while if no nans
            while flag:  # we ensure that no nan appears
                flag_n += 1
                x,  x_next, jumps, poisson = self.gen_forward(batch_size, N, n)
                flag = torch.isnan(x_next).any()
                if flag_n == 50:
                    raise ValueError('Keeps getting nan in forward!')

            # copmute solution components, y is the BSDE solution, z the diffusion coeff, u the jump component



            if n == N - 2:
                # set the terminal condition
                # y_prev = torch.zeros(batch_size, self.mathModel.dim_y)
                #y_prev = self.mathModel.g(x)
                if pretrain:
                    y = self.model_y(n, x_next)
                    itr_actual = itr_actual*2
                else:
                    y = self.model_y(n, x)
                y_prev = self.mathModel.g(x_next, poisson)
            else:
                # load previous model to get yprev which we use in the loss
                y = self.model_y(n, x)
                y_prev = model_y_prev(n + 1, x_next)

            # we compute the jump component

            u = self.model_u(n, x, jumps)

            # Compensator MC estimation
            #xc = x.clone()  # --> [bs, dimx]
            #xc = torch.unsqueeze(xc, -1)  # add an extra dimension from (bs, dimx) to [bs, dimx, 1]
            #xc = torch.repeat_interleave(xc, MC_size,  dim=1)  # repeat each sample MC_size times along the axis 1 , new shape [bs, dimx* MC_size, 1] [bs*mc,dim_x]

            xc = x.clone().unsqueeze(1)  # [bs, 1, dimx]
            xc = xc.repeat(1, MC_size, 1)  # [bs, MC_size, dimx]

            #xc= torch.ones(batch_size, MC_size, MC_size)
            MC_jumps,_ = self.mathModel.jumps(MC_size)
            MC_jumps = torch.unsqueeze(MC_jumps, 0)
            MC_jumps = torch.repeat_interleave(MC_jumps, batch_size,
                                               dim=0)  # TODO: here we have same jumps for all x in batch (same as Xavier). Maybe beter to generate batch_size x MC_size jumps
            # we run the model to compute the jump component over mc samples  --> but we need this just for the compensator

            u_mc = self.model_u(n, xc, MC_jumps)

            # compensator = E[u]
            compensator = torch.mean(u_mc, dim=1)  # Includes rate and dt

            Gamma = compensator / self.mathModel.dt  # Compensator in the driver f (as in Castro)
            loss = self.loss(x, n, y_prev, y, u, compensator, Gamma, x_next, pretrain=pretrain)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            loss_n.append(float(loss))

            # if i%(itr_actual-1) == 0:
            # print("time_"+str(n)+ "iter_"+str(i))
            # for par_group in self.optimizer.param_groups:
            # print(par_group["lr"])

        return loss_n


class BSDEiter():
    def __init__(self, mathModel, dim_h, lr):
        self.mathModel = mathModel
        self.dim_h = dim_h
        self.lr = lr

    def train_whole(self, batch_size, N, path, itr, multiplyer, MC_size):
        loss_data = []

        # reversed time loop
        for n in range(N - 2, -1, -1):
            # fix the learning rate
            coeff = 1
            if n == N - 2:
                coeff = 1
            print("time " + str(n))
            # initialize the model and the solver
            bsde_solver = BSDEsolver(self.mathModel, self.dim_h, self.lr, coeff)


            if n != N - 2:
                # load previous time step's model and the parameters
                bsde_solver.model_y.load_state_dict(torch.load(path + "Y_state_dict_" + str(n + 1)), strict=False)
                bsde_solver.model_u.load_state_dict(torch.load(path + "U_state_dict_" + str(n + 1)), strict=False)
                bsde_solver.optimizer.load_state_dict(torch.load(path + "optimizer_state_dict_opt_" + str(n + 1)))

            if n == N-2:
                print("Pre-training")
                loss_n = bsde_solver.train(batch_size, N, n, itr, path, multiplyer, MC_size, pretrain=True)
                plt.plot(loss_n)
                plt.title("Loss function at time N-1, pre-training")
                plt.show()

                print(loss_n[-20:])
                print("Actual-training")
            # train the model at time step n
            loss_n = bsde_solver.train(batch_size, N, n, itr, path, multiplyer, MC_size)
            loss_data.append(loss_n)
            torch.save(bsde_solver.model_y.state_dict(), path + "Y_state_dict_" + str(n)) #we save all the learnable parameters of the model — i.e., weights and biases.
            torch.save(bsde_solver.model_u.state_dict(), path + "U_state_dict_" + str(n))
            torch.save(bsde_solver.optimizer.state_dict(), path + "optimizer_state_dict_opt_" + str(n))

        return loss_data


class Result():
    def __init__(self, mathModel, dim_h):
        self.model_y = ModelY(mathModel, dim_h).to(device)
        self.model_u = ModelU(mathModel, dim_h).to(device)
        self.mathModel = mathModel


    def gen_x(self, batch_size, N):
        delta_t = self.mathModel.dt
        flag= True
        while flag:
            x = torch.zeros(batch_size, self.mathModel.dim_x, N, device=device) #[bs,dx,N]                                                                         N)  # [bs,dx,N]
            h = torch.zeros(batch_size, self.mathModel.dim_d + 1, N, device=device)  # [bs,dim_h,N]

            x_0_tensor = self.mathModel.x_0 + torch.zeros(batch_size, self.mathModel.dim_x, device=device)  # [bs,dx]
            h_V = -torch.log(1 - x_0_tensor[:, :self.mathModel.dim_d]) / self.mathModel.s
            h_D = x_0_tensor[:, self.mathModel.dim_d:self.mathModel.dim_d + 1] / self.mathModel.d
            h[:, :, 0] = torch.cat((h_V, h_D), dim=-1)
            x[:, :, 0] = x_0_tensor

            poisson = torch.zeros(batch_size, 1, device=device)  # [bs,dd,N]


            #x = self.mathModel.x_0*torch.ones(self.mathModel.dim_x) + torch.zeros(batch_size, N * self.mathModel.dim_x, device=device).reshape(-1, self.mathModel.dim_x,    N)  # [bs,dx,N]

            jumps = torch.zeros(batch_size, self.mathModel.dim_j, N, device=device)
            for i in range(N - 1):
                jumps_i = self.mathModel.jumps(batch_size)
                h[:, :, i + 1] = self.mathModel.step_forward(i, h[:, :, i], jumps_i)
                x[:, :, i + 1] = self.mathModel.h_to_vdc(h[:, :, i + 1], jumps_i)
                jumps[:, :, i] = jumps_i[0]

                poisson += (torch.sum(jumps_i[0], dim=-1, keepdim=True) != 0.0).float()# we store the jumps in the poisson variable

            if torch.isnan(x).any() or torch.isinf(x).any():
                print("NaN or Inf detected in generated x, retrying...")
            else:
                flag = False

        return x, jumps, poisson

    def predict(self, N, batch_size, x, path, jumps, poisson):
        ys = torch.zeros(batch_size, self.mathModel.dim_y, N)
        us = torch.zeros(batch_size, self.mathModel.dim_j, N)

        for n in range(N - 1):
            self.model_y.eval()
            self.model_u.eval()
            self.model_y.load_state_dict(torch.load(path + "Y_state_dict_" + str(n)), strict=False)
            self.model_u.load_state_dict(torch.load(path + "U_state_dict_" + str(n)), strict=False)
            y = self.model_y(n, x[:, :, n])
            u = self.model_u(n, x[:, :, n], jumps[:, :, n])

            ys[:, :, n] = y
            us[:, :, n] = u

        ys[:, :, N - 1] = self.mathModel.g(x[:, :, N - 1], poisson)

        return ys, us

    def L2(self, true, est, N):
        dt = self.mathModel.T / N
        diff = torch.mean(torch.sum(torch.linalg.norm((true - est) ** 2, dim=1) * dt, dim=-1), dim=0)
        l2_true = torch.mean(torch.sum(torch.linalg.norm((true) ** 2, dim=1) * dt, dim=-1), dim=0)
        return float(torch.sqrt(diff / l2_true))

    def compensator(self, N, batch_size, x, path, MC_size):
        compensators = torch.zeros(batch_size, self.mathModel.dim_j, N - 1)
        for n in range(N - 2):
            if n == N - 2:
                print(N - 2)
            self.model_u.load_state_dict(torch.load(path + "U_state_dict_" + str(n)), strict=False)
            self.model_u.eval()
            x_n = x[:, :, n]

            #xc = x_n.clone()  # --> [bs, dimx]
            #xc = torch.unsqueeze(xc, -1)  # add an extra dimension from (bs, dimx) to [bs, dimx, 1]
            #xc = torch.repeat_interleave(xc, MC_size, dim=1)  # repeat each sample MC_size times along the axis 1 , new shape [bs, dimx* MC_size, 1]

            xc = x_n.clone().unsqueeze(1)  # [bs, 1, dimx]
            xc = xc.repeat(1, MC_size, 1)  # [bs, MC_size, dimx]

            MC_jumps,_ = self.mathModel.jumps(MC_size)
            MC_jumps = torch.unsqueeze(MC_jumps, 0)
            MC_jumps = torch.repeat_interleave(MC_jumps, batch_size,
                                               dim=0)  # TODO: here we have same jumps for all x in batch (same as Xavier). Maybe beter to generate batch_size x MC_size jumps
            # we run the model to compute the jump component over mc samples  --> but we need this just for the compensator

            u_mc = self.model_u(n, xc, MC_jumps)

            # compensator = E[u]
            compensator = torch.mean(u_mc, dim=1)  # Includes rate and dt
            compensators[:, :, n] = compensator

        return compensators


    def monte_carlo(self, sample_size=10000):
        x,_,_ = self.gen_x(sample_size, self.mathModel.N)[0]
        return torch.mean(torch.sum(x[:,:,-1], dim=-1), dim=0)
