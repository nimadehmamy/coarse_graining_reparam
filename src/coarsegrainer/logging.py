### Todo: 
# 1. Move the logging code here
# 2. Merge the logging in experimentlogger.py file and Minimizer class

import os
import time
from torch.utils.tensorboard import SummaryWriter


# logger class 
# we want to take out the logging parts from the minimizer class and experiment logger and put them in a separate class
# we can then use the logger class in the minimizer and experiment logger
# the class should be able to log to tensorboard and to a csv file
# we can also use the class to log the hyperparameters of the model

class EnergyLogger:
    def __init__(self, x_shape, log_dir='../results/logs', log_name=None, log_step=10, log_pos_step=0,):
        self.history = dict(time=[], energy=[], x=[])
        self._last_time = 0.
        self._log_step = log_step
        self._log_pos_step = log_pos_step
        self._time_since_train_start = 0.
        self._last_time = 0.
        self.get_log_dir(log_dir, x_shape, log_name)
        self.writer = SummaryWriter(log_dir=self.log_path)
        # save hyperparameters
        self.writer.add_hparams({'log_step': self._log_step, 'log_pos_step': self._log_pos_step,}, {})
            
    def get_log_dir(self, log_dir,x_shape, log_name = None):
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
        self.log_name += f'_n{x_shape[0]}_d{x_shape[1]}'
        self.log_path = os.path.join(log_dir, self.log_name)
        print('Logging to:', self.log_path)
        
    def start_log_epoch(self):
        # to get an accurate measure of the time it takes to compute the energy
        self.update_start_time()
        # self._time_since_train_start = time.time()
        self._start_time = time.time()
        
    def log(self, step, energy, x=None):
        # logging costs time, so use it every k steps
        if step % self._log_step == 0:
            end_time = time.time()
            elapsed_time = end_time - self._start_time
            # total_time = end_time - self._time_since_train_start    
            self.log_step(step, energy, elapsed_time, x)
            self._start_time = end_time
        
    def log_step(self, batch_idx, energy, elapsed_time, x=None):
        self.writer.add_scalar('Energy', energy, batch_idx)
        self.writer.add_scalar('Time', elapsed_time, batch_idx)
        # also append to history
        self.history['time'].append(elapsed_time)
        self.history['energy'].append(energy)
        if self._log_pos_step > 0 and batch_idx % self._log_pos_step == 0:
            if x is not None:
                self.history['x'].append(x.detach().clone().numpy())
    
    def update_start_time(self):
        if self._time_since_train_start == 0:
            self._time_since_train_start = time.time()
        else:
            # we have trained before, so we need to update the time
            # use the last time logged to update the time since training start
            self._time_since_train_start = time.time() - self.history['time'][-1] 
        # we will use self._last_time to update the time since training start
        self._last_time = time.time()
    