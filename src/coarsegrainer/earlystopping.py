### Todo:
# 1. Define early stopping class
# 2. modify the minimizer class to use the early stopping class


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.reset()
        
    def reset(self, energy=float('inf')):
        self.best_energy = energy
        self.patience_counter = 0
        self.early_stopping_triggered = False
        
    def check_early_stopping(self, energy):
        if energy < self.best_energy - self.min_delta:
            self.reset(energy)
        else:
            self.patience_counter += 1
            self.early_stopping_triggered = False
        if self.patience_counter >= self.patience:
            self.early_stopping_triggered = True
            # print("Early stopping")
            