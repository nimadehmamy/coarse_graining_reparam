#!/usr/bin/env python3

# This script runs a set of experiments to compare the performance of the gradient descent minimizer, the CG minimizer and the GNN minimizer
# on a simple energy function. The energy function is a combination of a quadratic potential along the bonds and a Lennard-Jones potential
# for the loop formed by the bonds. The energy function is defined in the energy module of the coarsegrainer package. The energy function is
# defined as a combination of two energy functions, one for the quadratic potential and one for the Lennard-Jones potential. The energy function
# is defined as a sum of indexed energy functions, which allows for efficient computation of all-to-all energies. The energy function is then
# used to create a coarse grainer object, which is used to extract the coarse graining modes. The coarse graining modes are then used to create
# a GNN reparametrization, which is used to create a GNN minimizer. The GNN minimizer is then used to run a set of experiments with different
# values of the hyperparameters. The results of the experiments are then saved in a csv file using the experimentlogger module of the coarsegrainer
# package. The experiments are run multiple times to get an average of the results. The results of the experiments are then saved in a csv file.

# get current working directory
import os, sys
pwd = os.getcwd()
# append ../ to the sys path to access the coarsegraining package
sys.path.append(pwd + '/../src/')

import time

import numpy as np

import torch

### import the coarse-graining module
import coarsegrainer as cg
from coarsegrainer.minimizer import EnergyMinimizerPytorch, CGMinimizerPytorch, GNNMinimizerPytorch
from coarsegrainer.energy import Energy, LJ_potential, quadratic_potential
from coarsegrainer.GNN import GCN_CG, ResGCN_CG, GNNRes, GNN, GNNReparam

from experimentlogger import ExperimentLogger

V = lambda x: x.detach().cpu().numpy()

# # Quadratic Bonds + LJ loop
# Define an energy function which uses a quadratic potential for bonds and LJ for forming a loop. 
# It consists of:
# 1. a strong backbone line graph LJ, where every node is attracted to the next
# 2. a weaker loop LJ, where every l-th pair of nodes are attracted 

num_nodes, dims = 400, 3
n,d = num_nodes, dims
loop = 10
a = 1e-1 # strength of the loop
# init_sigma = 3.0 # initial standard deviation of the coordinates
# an initial std of 
init_sigma = n**(1/3)/2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

A = cg.utils.line_graph_A(n, k=1).to(device)
vdw = cg.utils.line_graph_A(n, k=loop).to(device)

A_loop = A+a*vdw

x = init_sigma*torch.randn(n, d).to(device) 


energy_params = dict(radius = 1, thres_A = 1e-4, lj_pow = 6, repel_pow = 1, repel_mag = 2.5e-3, 
                device = 'cuda')

energy_bond_lj = Energy(A_list=[A, .1*vdw], energy_func_list=[quadratic_potential, LJ_potential],
                        log_name='Energy_Bond_LJ', **energy_params)
energy_bond_lj(x).item()
## Extract CG modes using multiple samples
# energy_bond_lj.num_neg_pairs = n**2//2
# energy_bond_lj.get_indices()
energy_bond_lj.indices_neg[0].shape
t0 = time.time()

# even very few samples yield good quality cg_modes in this case
k = 10
# produce k samples with different std for x
x_samples = init_sigma*torch.randn(k, n, d, device =device)*torch.linspace(3e-0, 8e-0, k)[:, None, None].to(device)

cg_bond_lj = cg.CG.CoarseGrainer(energy_bond_lj, num_cg_modes=n)
cg_bond_lj.get_cg_modes(x_samples)

cg_time = time.time() - t0

exp_logger = ExperimentLogger(save_prefix='../results/CG_Bond_LJ_experiments') 


## Bonds + LJ
# quadratic potential along bonds and LJ for weak interactions. 
# define the initial position
initial_pos = x = init_sigma*1*torch.randn(n, d).to(device) / 2

init_x_std = x.std().item()

# we run a set of experiments with different values of the hyperparameters
# each run will include three types of experiments:
# - a run with a GNN minimizer
# - a run with a CG minimizer
# - a run with a gradient descent minimizer
# 
# the runs will be saved in a csv file with the results of the experiments
# using experimentlogger
# 
# the hyperparameters are:
# - the learning rate
# - the patience
# - the minimum delta for the early stopping criteria
# - the number of CG modes to use
# - the width of the GNN layers

# we will run the experiments with the following values of the hyperparameters
# learning rates
lrs = [5e-2, 2e-2, 1e-2]
# patience
patiences = [20]
# minimum delta
min_deltas = [1e-6, 1e-7]
# number of CG modes
num_cg_modes = n//np.array([3, 4, 5, 10])
# width of the GNN layers
gnn_widths = [8, 32]

# we run run each experiment 5 times to get an average of the results
num_runs = 5

CLAMP = 1e-1

# GNN becomes unstable with large lr. use lr/10 for GNN
# LR_GNN = 2e-3

EPOCHS = 20 
STEPS = 3000

# The experiment loops 
for run in range(num_runs):
    for lr in lrs:
        for patience in patiences:
            for min_delta in min_deltas:
                # create the gradient descent minimizer
                gd_minimizer = EnergyMinimizerPytorch(energy_bond_lj, initial_pos, optimizer_type='Adam', lr=lr, 
                    clamp_grads=CLAMP, log_step=20, log_pos_step=0, 
                    log_dir='../results/logs', log_name='Bond_LJ', patience=patience, min_delta=min_delta)
                # run the gradient descent minimizer
                exp_logger.run_experiment(gd_minimizer, epochs=EPOCHS, steps=STEPS, x0_std=init_x_std, num_nodes=num_nodes)
                
                for num_cg in num_cg_modes:
                    # create the CG minimizer
                    cg_minimizer = CGMinimizerPytorch(energy_bond_lj, initial_pos, cg_bond_lj.cg_modes[:,:num_cg], 
                        optimizer_type='Adam', lr=lr, lr_cg=lr,
                        clamp_grads=CLAMP, log_step=20, log_pos_step=0, 
                        log_dir='../results/logs', log_name=f'CG_Bond_LJ{num_cg/n:.2f}', 
                        patience=patience, min_delta=min_delta,
                        cg_patience=patience, cg_min_delta=min_delta*1e1)
                    # run the CG minimizer
                    exp_logger.run_experiment_cg(cg_minimizer, cg_time, epochs=EPOCHS, steps=STEPS, x0_std=init_x_std, num_nodes=num_nodes)
                    
                    for gnn_width in gnn_widths:
                        # create the name of the run
                        run_name = f'lr_{lr}_pat_{patience}_min_d_{min_delta}_cg_modes_{num_cg}_gnn_w_{gnn_width}_run_{run}'
                        print(run_name)
                        ### GNN
                        # create the GNN reparametrization
                        h = gnn_width
                        gnn_reparam = GNNReparam([h, h//2, d], cg_bond_lj, num_cg=num_cg, 
                            bias=True, activation=torch.nn.Tanh()).to(device)
                        gnn_reparam.rescale_output(init_x_std)
                        # create the GNN minimizer
                        gnn_minimizer = GNNMinimizerPytorch(energy_bond_lj, initial_pos, gnn_reparam, 
                            optimizer_type='Adam', lr=lr, lr_gnn=lr/10,
                            clamp_grads=CLAMP, log_step=20, log_pos_step=0, 
                            log_dir='../results/logs', log_name=f'GNN_Bond_LJ{num_cg/n:.2f}',
                            patience=patience, min_delta=min_delta, 
                            gnn_patience=patience, gnn_min_delta=min_delta*1e1)
                        # run the GNN minimizer
                        exp_logger.run_experiment_gnn(gnn_minimizer, cg_time, epochs=EPOCHS, steps=STEPS, x0_std=init_x_std, num_nodes=num_nodes)
                        # clear the cache
                        torch.cuda.empty_cache()
                        print('End of run\n========================\n')
                        time.sleep(2)