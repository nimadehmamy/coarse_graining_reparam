
# Pure pytorch implementation of the CG minimizer and EnergyMinimizer
# We will use the same energy function as before
# We will use the same optimization loop as before
import os
import time

import matplotlib.pyplot as plt


import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# import early stopping from the parent directory
# import sys
# sys.path.append('..')

from ..earlystopping import EarlyStopping
        
# We should move the early stopping functionality into its own class

# Can we define the energy minimizer a bit more generally?
# We can define the energy minimizer as a subclass of the PyTorch Module
# We want the train_step method to be subclass independent
# the main difference between the subclasses will be the forward method
# and the parameters that are optimized
# we only need to change the clamp_grads and the optimizer
# we can use the same training loop for all subclasses
# we can also use the same logging and early stopping mechanism
# we can also use the same plotting method
# for clamping, we can use the clamp_grads parameter
# for the optimizer, we can use the optimizer_type parameter
# we can also use the same logging and early stopping mechanism


# Pure pytorch implementation of the EnergyMinimizer


class EnergyMinimizerPytorch(torch.nn.Module):
    def __init__(self, energy_func, initial_pos, optimizer_type=None, lr=0.1, clamp_grads=1., 
                log_step=10, log_pos_step=0, log_dir='../results/logs', patience=5, min_delta=0.0, log_name=None, earlystopping=None):
        """This is a class to minimize the energy of a configuration using PyTorch.
        It uses the energy function to compute the energy of the configuration.

        Args:
            energy_func (_type_): _description_
            initial_pos (_type_): _description_
            optimizer_type (_type_, optional): _description_. Defaults to None.
            lr (float, optional): _description_. Defaults to 0.1.
            clamp_grads (_type_, optional): _description_. Defaults to 1..
            log_step (int, optional): _description_. Defaults to 10.
            log_pos_step (int, optional): _description_. Defaults to 0.
            log_dir (str, optional): _description_. Defaults to '../results/logs'.
            patience (int, optional): _description_. Defaults to 5.
            min_delta (float, optional): _description_. Defaults to 0.0.
        """
        super().__init__()
        self.energy_func = energy_func
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.clamp_grads = clamp_grads
        self.initialize_pos_params(initial_pos)
        
        self.optimizer = self.get_optimizer(self.parameters(), lr=self.lr)
        
        self.history = dict(time=[], energy=[], x=[])
        self._last_time = 0.
        self._log_step = log_step
        self._log_pos_step = log_pos_step
        self._time_since_train_start = 0.
        self._last_time = 0.
        self.best_energy = float('inf')
        if earlystopping is not None:
            self.early_stop = earlystopping
        else:
            self.early_stop = EarlyStopping(patience=patience, min_delta=min_delta)
        self.get_log_dir(log_dir, initial_pos, log_name)
        self.writer = SummaryWriter(log_dir=self.log_path)
        # save hyperparameters
        self.writer.add_hparams({'lr': self.lr, 'clamp_grads': self.clamp_grads, 'log_step': self._log_step,
                'log_pos_step': self._log_pos_step, 'patience': self.early_stop.patience, 'min_delta': self.early_stop.min_delta}, {})
        
    # define patience as a property which also updates the early stopping object
    @property
    def patience(self):
        return self.early_stop.patience
    @patience.setter
    def patience(self, patience):
        self.early_stop.patience = patience
        
    # define min_delta as a property which also updates the early stopping object
    @property
    def min_delta(self):
        return self.early_stop.min_delta
    @min_delta.setter
    def min_delta(self, min_delta):
        self.early_stop.min_delta = min_delta
        
    def get_log_dir(self, log_dir,x, log_name = None):
        if log_name is not None:
            self.log_name = log_name
        else:
            try: 
                self.log_name = self.energy_func.__qualname__.split('.')[0].lstrip('get_') 
            except:
                try:
                    self.log_name = self.energy_func.__class__.__name__
                except:
                    self.log_name = 'energy'
        print('Log name:',self.log_name)
        # use the shape of the initial_pos in the name of the log file
        self.log_name += f'_n{x.shape[0]}_d{x.shape[1]}'
        self.log_path = os.path.join(log_dir, self.log_name)
        print('Logging to:', self.log_path)
        
    def check_early_stopping(self, energy):
        self.early_stop.check_early_stopping(energy)
        self.early_stopping_triggered = self.early_stop.early_stopping_triggered
        
    def get_optimizer(self, params, **optim_kwargs):
        if self.optimizer_type is not None:
            print(f'Using {self.optimizer_type} optimizer')
            optim = getattr(torch.optim, self.optimizer_type)
        else:
            print('Using Adam optimizer')        
            optim = torch.optim.Adam
        return optim(params, **optim_kwargs)
            
    def initialize_pos_params(self, initial_pos):
        # initialize the position parameters
        # ensure they are registered as nn.parameters
        self.x = torch.nn.Parameter(initial_pos.clone().detach())   
            
    def get_x(self):
        return self.x
    
    def forward(self):
        return self.energy_func(self.get_x())

    def training_step(self,):# batch_idx):
        opt = self.optimizer
        opt.zero_grad()
        # Compute energy
        energy = self.forward()
        energy.backward()
        # clamp gradients to avoid infinities
        # this needs to be done for all parameters of the current optimizer
        for param in opt.param_groups[0]['params']:
            # print(param.shape)
            param.grad = torch.clamp(param.grad, -self.clamp_grads, self.clamp_grads)
        opt.step()
        return energy.item()
            
    def log(self, batch_idx, energy, elapsed_time):
        self.writer.add_scalar('Energy', energy, batch_idx)
        self.writer.add_scalar('Time', elapsed_time, batch_idx)
        # also append to history
        self.history['time'].append(elapsed_time)
        self.history['energy'].append(energy)
        if self._log_pos_step > 0 and batch_idx % self._log_pos_step == 0:
            self.history['x'].append(self.x.detach().clone().numpy())
            
    # train the model
    def train(self, nsteps):
        # to get an accurate measure of the time it takes to compute the energy
        self.update_start_time()
        # self._time_since_train_start = time.time()
        start_time = time.time()
        for step in range(nsteps):
            energy = self.training_step()
            # logging costs time, so use it every k steps
            if step % self._log_step == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                # total_time = end_time - self._time_since_train_start    
                self.log(step, energy, elapsed_time)
                start_time = end_time
                # Early stopping        
                self.check_early_stopping(energy)
            if self.early_stopping_triggered:
                print("Early stopping at step", step)
                break
        # because we may call train multiple times,
        # the writer will be closed only when the object is deleted
        # self.writer.close()
        return self.history
    
    def update_start_time(self):
        if self._time_since_train_start == 0:
            self._time_since_train_start = time.time()
        else:
            # we have trained before, so we need to update the time
            # use the last time logged to update the time since training start
            self._time_since_train_start = time.time() - self.history['time'][-1] 
        # we will use self._last_time to update the time since training start
        self._last_time = time.time()
    
    def plot_history(self, start=0, end=None):
        if end is None:
            end = len(self.history['time'])
        plt.plot(np.cumsum(self.history['time'])[start:end], self.history['energy'][start:end])
        plt.xlabel('time (s)')
        plt.ylabel('energy')
        plt.xscale('log')
        plt.yscale('symlog')
        # plt.show()
        
