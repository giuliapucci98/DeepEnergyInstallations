import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from Network import ModelControlConstant



# We implement a solver for Forward-Backward SDE with jumps.
# The model is trained by using the Euler-Maruyama method for the diffusion part and the Monte Carlo method for the jump part.


# with this class we solve the BSDE with jumps

class Train():
    def __init__(self, mathModel, dim_h):
        self.mathModel = mathModel
        self.dim_h = dim_h
        #self.model = ModelControl(mathModel, self.dim_h)
        self.model = ModelControlConstant(mathModel)
        self.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


    def train(self, batch_size, itr, lr):
        losses = []

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for i in range(itr):
            control, x, J = self.model(batch_size)
            if i % 50 == 0:
                dims_str = " | ".join([f"dim{j+1}: {control[0, 0, j].item():.4f}" for j in range(self.mathModel.dim_d)])
                print(f"itr={i} | {dims_str} | J: {J[0].item():.4f}")
            if torch.isnan(x).any():
                print("NaN detected in x at iteration", i)
                break
            # compute the loss function
            loss = torch.mean(J,dim=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))

        return losses, control, x












