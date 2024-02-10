
# Pure pytorch implementation of the CG minimizer and EnergyMinimizer
# We will use the same energy function as before
# We will use the same optimization loop as before
import os
import time

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

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
                log_step=10, log_pos_step=0, log_dir='../results/logs', patience=5, min_delta=0.0, log_name=None):
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
        self.patience = patience
        self.patience_counter = 0
        self.min_delta = min_delta
        self.early_stopping_triggered = False
        self.get_log_dir(log_dir, initial_pos, log_name)
        self.writer = SummaryWriter(log_dir=self.log_path)
        # save hyperparameters
        self.writer.add_hparams({'lr': self.lr, 'clamp_grads': self.clamp_grads, 'log_step': self._log_step,
                'log_pos_step': self._log_pos_step, 'patience': self.patience, 'min_delta': self.min_delta}, {})
        
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
            
    def check_early_stopping(self, energy):
        if energy < self.best_energy - self.min_delta:
            self.best_energy = energy
            self.patience_counter = 0
            self.early_stopping_triggered = False
        else:
            self.patience_counter += 1
            self.early_stopping_triggered = False
        if self.patience_counter >= self.patience:
            self.early_stopping_triggered = True
            # print("Early stopping")
        
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
        
# we can define the CG minimizer as a subclass of the EnergyMinimizer class
# The CG minimizer will take a set of cg_modes as inputs and use them to compute the energy
# the coefficients of the cg_modes will be the optimization variables self.z
# we will use the same training loop, logging, early stopping mechanism and plotting method
# we only need to change the forward method and the parameters that are optimized and the optimizer
# we will also have a method to initialize the cg parameters
# we will have a coarse grained training step and a fine grained training step
# we will also have a method to switch between the two stages
# the switch will be triggered by the early stopping mechanism
# when switching, we will change the optimizer to the fine grained optimizer
# we will also reset the early stopping mechanism
# the get_x method will return the reparameterized x when coarse grained and the original x when fine grained

class CGMinimizerPytorch(EnergyMinimizerPytorch):
    def __init__(self, energy_func, initial_pos, cg_modes, optimizer_type=None, lr=0.1, lr_cg=0.1, clamp_grads=1., 
                log_step=10, log_pos_step=0, log_dir='../results/logs', 
                patience=5, min_delta=0.0, cg_patience=5, cg_min_delta=0.0, log_name=None):
        super().__init__(energy_func, initial_pos, optimizer_type, lr, clamp_grads, 
                log_step, log_pos_step, log_dir, patience, min_delta, log_name)
        self.cg_modes = cg_modes
        self.lr_cg = lr_cg
        self.initialize_cg_params(initial_pos)
        # self.get_cg_optimizer()
        self.cg_optimizer = self.get_optimizer([self.z], lr=self.lr_cg)
        # store the original optimizer as the fg_optimizer
        self.fg_optimizer = self.optimizer
        # choose the cg optimizer as the optimizer
        self.optimizer = self.cg_optimizer
        self.init_CG_hyperparameters(cg_patience, cg_min_delta, patience, min_delta)
        self.fine_grained = False
        # add the cg_patience and cg_min_delta to the hyperparameters
        self.writer.add_hparams({'cg_num': self.cg_modes.shape[1],
            'lr': self.lr, 'lr_cg': self.lr_cg, 'clamp_grads': self.clamp_grads, 'log_step': self._log_step,
            'log_pos_step': self._log_pos_step, 'patience': self.fg_patience, 'min_delta': self.fg_min_delta,
            'cg_patience': cg_patience, 'cg_min_delta': cg_min_delta}, {})
        
    def init_CG_hyperparameters(self, cg_patience, cg_min_delta, fg_patience, fg_min_delta):
        self.cg_patience = cg_patience
        self.cg_min_delta = cg_min_delta
        # during cg, self.patience and self.min_delta should be the cg patience and min_delta
        # later during fg, we will reset them to the original values
        # first, keep the originals as fg_patience and fg_min_delta
        self.fg_patience = fg_patience
        self.fg_min_delta = fg_min_delta
        # we are starting with cg, so set the patience and min_delta to the cg values
        self.patience = cg_patience
        self.min_delta = cg_min_delta 
        
    def initialize_cg_params(self, initial_pos):
        # initialize the CG parameters
        # ensure they are registered as nn.parameters
        init_z = self.cg_modes.T @ initial_pos.clone().detach()
        # due to projection onto cg_modes, x has a different scale
        # we need to scale it back to the original scale
        # we can use the std of the initial x to scale it back
        original_sigma_x = initial_pos.std()
        # get std of cg_modes @ z
        projected_sigma_x = (self.cg_modes @ init_z).std()
        self.scaling_factor = original_sigma_x/projected_sigma_x
        self.z = torch.nn.Parameter((self.scaling_factor*init_z).requires_grad_(True))
        
    def get_x(self):
        # get the position variables
        # in the CG stage, x = cg_modes @ z
        # in the fine-grained stage, x = x
        if self.fine_grained:
            return self.x
        else:
            return self.cg_modes @ self.z
            
    # toggle fine_grained to switch between the two stages
    # when turning fine_grained to True, we need to update the value of x
    # and reset the state of the fine-grained optimizer
    def start_fine_graining(self):
        print('Starting fine-graining')
        self.x.data = self.get_x().data
        # now that the CG stage has finished, log current `x_cg = self.get_x()` in history as "x_cg"
        self.history['x_cg'] = self.x.detach().cpu().clone().numpy()
        
        # reset the state of the fine-grained optimizer
        self.optimizer = getattr(torch.optim, self.optimizer_type)([self.x], lr = self.lr)
        self.fg_optimizer = self.optimizer
        # self.optimizer = self.fg_optimizer
        self.cg_steps = len(self.history['time'])
        # change the patience and min_delta to the fine-grained values
        self.patience = self.fg_patience
        self.min_delta = self.fg_min_delta
        # unset the early stopping flag
        self.early_stopping_triggered = False
        self.fine_grained = True
        # we are starting the fine-grained stage, so reset the patience counter
        self.patience_counter = 0
        
    # We also want a full training loop that can switch between the two stages
    # we will use the same training loop as before
    # the only difference is that we will switch between the two stages
    # we will use the early stopping mechanism to switch between the two stages
    # def train_full(self, nsteps):
    #     # first, we will train the CG stage
    #     # since both stages use the same training loop, we can use the same method
    #     h = self.train(nsteps)
    #     # now check if early stopping was triggered
    #     # then, if we were not already in the fine-grained stage, we switch to the fine-grained stage
    #     if self.early_stopping_triggered and not self.fine_grained:
    #         self.start_fine_graining()
    #         # now we train the fine-grained stage
    #         h = self.train(nsteps)
            
    #     return h
            
    
    # instead of introducing train_full, we could override the train method
    # and call the super().train method for individual stages
    # this way, we can use the same training loop for both stages
    # we can also use the same early stopping mechanism for both stages
    def train(self, nsteps):
        # first, we will train the CG stage
        # since both stages use the same training loop, we can use the same method
        # h = self.train(nsteps)
        # use the parent train method 
        h = super().train(nsteps)
        # now check if early stopping was triggered
        # then, if we were not already in the fine-grained stage, we switch to the fine-grained stage
        if self.early_stopping_triggered and not self.fine_grained:
            self.start_fine_graining()
            # now we train the fine-grained stage
            # h = self.train(nsteps)
            h = super().train(nsteps)
            
        return h
    
    # for backward compatibility, we can also define a train_full method
    # that calls the train method
    def train_full(self, nsteps):
        return self.train(nsteps)
            
            
# we will use the GNNReaparam to reparameterize the x variables
# for the optimization, we will use the same two stage optimization
# in the first stage the optimization variables will be the GNN parameters 
# and in the second stage the optimization variables will be the x variables
# we will use the same EnergyMinimizerPytorch class to perform the optimization
# the only difference is that we will use the GNN to compute the x variables
# and we will use the x variables to compute the energy
# The GNN minimizer will also use the GNN parameters as the optimization variables
# make this a subclass of the EnergyMinimizerPytorch1 class
# we will have a method to switch between the two stages
# the switch will be triggered by the early stopping mechanism
# when switching, we will change the optimizer to the fine grained optimizer
# we will also reset the early stopping mechanism
# the get_x method will return the GNN reparameterized x when coarse grained 
# and the original x when fine grained

class GNNMinimizerPytorch(EnergyMinimizerPytorch):
    def __init__(self, energy_func, initial_pos, gnn_reparam, optimizer_type=None, lr=0.1, lr_gnn=0.1, clamp_grads=1., 
                log_step=10, log_pos_step=0, log_dir='../results/logs', 
                patience=5, min_delta=0.0, gnn_patience=5, gnn_min_delta=0.0, log_name=None):
        super().__init__(energy_func, initial_pos, optimizer_type, lr, clamp_grads, 
                log_step, log_pos_step, log_dir, patience, min_delta, log_name)
        self.gnn = gnn_reparam
        self.lr_gnn = lr_gnn
        self.optimizer_gnn = self.get_optimizer(self.gnn.parameters(), lr=self.lr_gnn)
        self.fg_optimizer = self.optimizer
        self.optimizer = self.optimizer_gnn
        self.init_GNN_hyperparameters(gnn_patience, gnn_min_delta, patience, min_delta)
        self.fine_grained = False
        # add the gnn_patience and gnn_min_delta to the hyperparameters
        self.writer.add_hparams({'gnn_hidden_dims': torch.tensor(self.gnn.hidden_dims),
            'lr': self.lr, 'lr_gnn': self.lr_gnn, 'clamp_grads': self.clamp_grads, 'log_step': self._log_step,
            'log_pos_step': self._log_pos_step, 'patience': self.fg_patience, 'min_delta': self.fg_min_delta,
            'gnn_patience': gnn_patience, 'gnn_min_delta': gnn_min_delta}, {})
        
    def init_GNN_hyperparameters(self, gnn_patience, gnn_min_delta, fg_patience, fg_min_delta):
        self.gnn_patience = gnn_patience
        self.gnn_min_delta = gnn_min_delta
        # during gnn, self.patience and self.min_delta should be the gnn patience and min_delta
        # later during fg, we will reset them to the original values
        # first, keep the originals as fg_patience and fg_min_delta
        self.fg_patience = fg_patience
        self.fg_min_delta = fg_min_delta
        # we are starting with gnn, so set the patience and min_delta to the gnn values
        self.patience = gnn_patience
        self.min_delta = gnn_min_delta
        
    def get_x(self):
        # get the position variables
        # in the GNN stage, x = gnn(latent_embedding)
        # in the fine-grained stage, x = x
        if self.fine_grained:
            return self.x
        else:
            return self.gnn()
        
    # toggle fine_grained to switch between the two stages
    # when turning fine_grained to True, we need to update the value of x
    # and reset the state of the fine-grained optimizer
    def start_fine_graining(self):
        self.x.data = self.get_x().data
        # now that the CG stage has finished, log current `x_cg = self.get_x()` in history as "x_cg"
        self.history['x_cg'] = self.x.detach().cpu().clone().numpy()
        
        # reset the state of the fine-grained optimizer
        self.optimizer = getattr(torch.optim, self.optimizer_type)([self.x], lr = self.lr)
        self.fg_optimizer = self.optimizer
        self.gnn_steps = len(self.history['time'])
        # change the patience and min_delta to the fine-grained values
        self.patience = self.fg_patience
        self.min_delta = self.fg_min_delta
        # unset the early stopping flag
        self.early_stopping_triggered = False
        self.fine_grained = True
        # we are starting the fine-grained stage, so reset the patience counter
        self.patience_counter = 0
    
    # We also want a full training loop that can switch between the two stages
    # we will use the same training loop as before
    # instead of introducing new method, we override the train method
    # and call the super().train method for individual stages
    # this way, we can use the same training loop for both stages
    # we can also use the same early stopping mechanism for both stages
    def train(self, nsteps):
        # first, we will train the CG stage
        # since both stages use the same training loop, we can use the same method
        # h = self.train(nsteps)
        # use the parent train method 
        h = super().train(nsteps)
        # now check if early stopping was triggered
        # then, if we were not already in the fine-grained stage, we switch to the fine-grained stage
        if self.early_stopping_triggered and not self.fine_grained:
            self.start_fine_graining()
            # now we train the fine-grained stage
            # h = self.train(nsteps)
            h = super().train(nsteps)
            
        return h
    
    
    # for backward compatibility, we can also define a train_full method
    # that calls the train method
    def train_full(self, nsteps):
        return self.train(nsteps)