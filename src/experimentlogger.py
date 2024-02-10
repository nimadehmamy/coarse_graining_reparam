# To collect and plot results, use a list of dictionaries
import datetime

import numpy as np
import pandas as pd
import seaborn as sns


class ExperimentLogger:
    def __init__(self, save_prefix = '../results/CG_experiment_', previous_results_csv = None):
        self.results = []
        self.save_prefix = save_prefix
        self.current_experiment = None
        # load previous results as a pd dataframe, if path is given
        # assume the path is a csv file
        if previous_results_csv is not None:
            self.results = pd.read_csv(previous_results_csv).to_dict('records')

    def start_experiment(self, energy_function, # num_nodes, 
                        model_name):
        self.current_experiment = {
            'energy_function': energy_function,
            # 'num_nodes': num_nodes,
            'model_name': model_name,
            'energy': None,
            'time': None
        }
        

    def log_result(self, energy, time, **kws):
        if self.current_experiment is not None:
            self.current_experiment['energy'] = energy
            self.current_experiment['time'] = time
            # allow for additional keyword arguments to be logged
            self.current_experiment.update(kws)

    def end_experiment(self):
        if self.current_experiment is not None:
            self.results.append(self.current_experiment)
            self.current_experiment = None
    
    
    def run_experiment(self, model, epochs=10, steps=5000, log_name=None, **extra_log_kws):
        self._run_stage(model, epochs, steps, log_name)
        # log the result
        self._log_stage(model, **extra_log_kws)
        # end the experiment
        self.end_experiment()
        # save csv after each experiment
        self.to_csv() 
        
    def run_experiment_cg(self, model, cg_mode_time, epochs=10, steps=5000, log_name=None, **extra_log_kws):
        self._run_stage(model, epochs, steps, log_name)
        # log the result
        self._log_stage_cg(model, cg_mode_time, **extra_log_kws)
        # end the experiment
        self.end_experiment()
        # save csv after each experiment
        self.to_csv()
        
    def run_experiment_gnn(self, model, cg_mode_time, epochs=10, steps=5000, log_name=None, **extra_log_kws):
        self._run_stage(model, epochs, steps, log_name)
        # log the result
        self._log_stage_gnn(model, cg_mode_time, **extra_log_kws)
        # end the experiment
        self.end_experiment()
        # save csv after each experiment
        self.to_csv()
                
    def _run_stage(self, model, epochs=10, steps=5000, log_name=None):
        log_name = log_name or model.log_name
        # start the experiment
        self.start_experiment(model.energy_func.log_name, log_name)
        print(f'Running experiment {log_name}')
        for i in range(epochs):
            # train the model
            # em.early_stopping_triggered = False
            model.energy_func.update_neg_pairs()
            h = model.train(steps)
            print(len(h['energy']), f"{h['energy'][-1]:.3g}, {np.sum(h['time']):.2f}")
            # log interim result
            self.log_result(model.history['energy'][-1], np.sum(model.history['time']),)
            if model.early_stopping_triggered:
                break
            
    def _log_stage(self, model, **extra_log_kws):
        # log the result
        self.log_result(model.history['energy'][-1], np.sum(model.history['time']),
                    # log hyperparameters such as patience and min_delta
                    lr=model.lr, clamp_grads=model.clamp_grads,
                    patience=model.patience, min_delta=model.min_delta, 
                    **extra_log_kws)    
        
    def _log_stage_cg(self, model, cg_mode_time, **extra_log_kws):
        self.log_result(model.history['energy'][-1], np.sum(model.history['time'])+cg_mode_time,
                # log hyperparameters such as patience and min_delta
                lr=model.lr, lr_cg=model.lr_cg, clamp_grads=model.clamp_grads,
                patience=model.fg_patience, min_delta=model.fg_min_delta, 
                cg_patience=model.cg_patience, cg_min_delta=model.cg_min_delta,
                cg_steps=model.cg_steps, 
                cg_time=np.sum(model.history['time'][:model.cg_steps])+cg_mode_time, 
                cg_energy=model.history['energy'][model.cg_steps-1],
                **extra_log_kws)
        
    def _log_stage_gnn(self, model, cg_mode_time, **extra_log_kws):
        self.log_result(model.history['energy'][-1], np.sum(model.history['time'])+cg_mode_time,
                # log hyperparameters such as patience and min_delta
                num_cg_modes=model.gnn.num_cg,
                lr=model.lr, lr_cg=model.lr_gnn, clamp_grads=model.clamp_grads,
                patience=model.fg_patience, min_delta=model.fg_min_delta, 
                cg_patience=model.gnn_patience, cg_min_delta=model.gnn_min_delta,
                cg_steps=model.gnn_steps, 
                cg_time=np.sum(model.history['time'][:model.gnn_steps])+cg_mode_time, 
                cg_energy=model.history['energy'][model.gnn_steps-1],
                hidden_dims=model.gnn.hidden_dims,
                **extra_log_kws)
            
    def to_dataframe(self):
        return pd.DataFrame(self.results)
    
    def to_csv(self):
        self.df = self.to_dataframe()
        self.df.to_csv(self.save_prefix + datetime.datetime.now().strftime("%Y-%m-%d-%H") + '.csv', index=False)
        
    def plot_results(self, x='time', y='energy', hue='model_name', style='energy_function'):
        # plot as scatter plot
        sns.relplot(data = self.df, x=x, y=y, hue=hue, style=style, kind='scatter')
        # sns.relplot(data = self.df, x=x, y=y, hue=hue, style=style, kind='line')
        
    # when object itself is printed, print the dataframe
    def __repr__(self):
        # only make the dataframe if it doesn't exist
        if not hasattr(self, 'df'):
            self.df = self.to_dataframe()
        # We wnat ensure that the dataframe displayed in the fancy format in the notebook
        # so we use the _repr_html_ method
        return self.df._repr_html_()
        # return self.df
        