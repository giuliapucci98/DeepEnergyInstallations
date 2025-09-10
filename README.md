# DeepEnergyInstallations

This repository contains complementary approaches for **stochastic control of renewable capacity installations under uncertainty**.

# DeepBSDE-ControlSelector 
  A threshold-based policy approach, reformulated via a nonlinear PIDE–BSDE and solved with a neural backward BSDE method adapted to jump processes.  

This module implements the threshold-based BSDE control approach.  
It is structured into the following files:

- **mainImpulse.py** – [selects the optimal threshold-based control strategy, evaluates it on sample trajectories, and generates plots of the results]
- **MathModels.py** – [model]  
- **Networks.py** – [neural network architectures]  
- **Solver.py** – [training / BSDE solver routine]  
- **ImpulseSelection.py** – [class that trains and evaluates BSDEs for a range of threshold controls, and selects the optimal impulse strategy.]  

Run via:
```bash
cd DeepBSDE-ControlSelector
python mainImpulse.py 
```

# DeepControl-Solver

This module implements the direct neural control approach, where the feedback control law is fully parameterized as a neural network and optimized through stochastic gradient descent.

It is structured into the following files:

- **mainControl.py** – [trains the neural feedback control, evaluates its performance, and generates plots of state trajectories and learned control laws]
- **MathModels.py** – [defines the model]
- **Networks.py** – [neural network architectures]  
- **Solver.py** – [class that contains the training routines for the neural control]

Run the solver via:
```bash
cd DeepControl-Solver
python mainControl.py
```




