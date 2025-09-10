import os

import torch
import numpy as np
import json
import matplotlib.pyplot as plt


from Solver import Result
from Solver import BSDEiter

class ImpulseSelection:
    def __init__(self, control_min, control_max, number_of_impulses, mathModel, graph_path):
        self.control_min = control_min
        self.control_max = control_max
        self.number_of_impulses = number_of_impulses
        self.controls = np.linspace(control_min, control_max, number_of_impulses)
        self.mathModel = mathModel
        self.graph_path = graph_path

        self.Y0s = []

    def train_single_bsde(self, dim_h, lr, batch_size, N, path, itr, multiplyer, MC_size):
        bsde_itr = BSDEiter(self.mathModel, dim_h, lr)
        loss = bsde_itr.train_whole(batch_size, N, path, itr, multiplyer, MC_size)
        with open(path + "loss.json", 'w') as f:
            # indent=2 is not needed but makes the file human-readable
            # if the data is nested
            json.dump(loss, f, indent=2)
        return loss

    def evaluate_single_bsde(self, dim_h, N, path, MC_size):
        with open(path + "loss.json", 'r') as f:
            loss = json.load(f)
        eval_size = 100
        results = Result(self.mathModel, dim_h)
        x, jumps, poisson = results.gen_x(eval_size, N)
        y, u = results.predict(N, eval_size, x, path, jumps, poisson)
        compensator = results.compensator(N,eval_size,x,path,MC_size)
        return y

    def select_impulse(self, dim_h, N, path, MC_size, batch_size, itr, multiplyer, lr):
        for i in range(self.controls.shape[0]):

            control_path = path + f"control_{self.controls[i]}/"
            os.makedirs(control_path, exist_ok=True)

            # Update the control parameter in the math model
            self.mathModel.control_parameter = self.controls[i]

            # Train the BSDE for the current control
            loss = self.train_single_bsde(dim_h, lr, batch_size, N, control_path, itr, multiplyer, MC_size)

            # Plot losses
            self.plot_losses(loss, self.controls[i])




            # Evaluate the BSDE for the current control
            y=self.evaluate_single_bsde(dim_h, N, control_path, MC_size)

            # Store the results
            self.Y0s.append(float(y[0,0,0]))


        npY0s = np.array(self.Y0s)
        best_control_index = np.argmin(npY0s)

        #plot control and y0
        plt.scatter(self.controls, npY0s)
        plt.title("Y0 with respect to control parameter")
        plt.savefig(self.graph_path + "Y0_vs_control.png")
        plt.show()

        return (self.controls[best_control_index], npY0s[best_control_index])

    def plot_losses(self, loss, control):

        '''
        plt.plot(loss[0][:])
        plt.title("Loss function at time N-1, control parameter: " + str(control))
        #plt.savefig(graph_path + "loss1.png")
        plt.show()

        last_few = int(len(loss[0])*0.1)
        plt.plot(loss[0][-last_few:])
        plt.title("Loss function at time N-1, last 1000 iterations, control parameter: " + str(control))
        #plt.savefig(graph_path + "loss2.png")
        plt.show()


        plt.plot(loss[1][:])
        plt.title("Loss function at time N-2, control parameter: " + str(control))
        #plt.savefig(graph_path + "loss3.png")
        plt.show()

        plt.plot(loss[2][:])
        plt.title("Loss function at time N-3, control parameter: " + str(control))
        #plt.savefig(graph_path + "loss4.png")
        plt.show()

        '''

        loss_last_epoch = []
        for loss_n in loss:
            last_few = int(len(loss_n) * 0.1)
            loss_n = np.mean(np.array(loss_n[-last_few:]))
            loss_last_epoch.append(loss_n)

        loss_last_epoch = list(reversed(loss_last_epoch))
        plt.title("Averaged reversed Loss function for each time step, control parameter: " + str(control))
        plt.plot(loss_last_epoch)
        #plt.savefig(graph_path + "loss_all.png")
        plt.show()