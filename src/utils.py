import torch
import os

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def log_training(epoch_phase, train_loss, val_loss, val_acc, lr):
    os.makedirs("logs", exist_ok=True)
    # Name updated to V2
    log_path = "logs/v2_training_log.txt"
    
    cabecera = False
    if not os.path.exists(log_path):
        cabecera = True
            
    with open(log_path, "a") as f:
        if cabecera:
             f.write("Fase-Ep | Train Loss | Val Loss | Val Acc | LR\n")
             f.write("-" * 65 + "\n")
        f.write(f"{epoch_phase:<8} | {train_loss:.4f}     | {val_loss:.4f}   | {val_acc:.4f}  | {lr:.2e}\n")