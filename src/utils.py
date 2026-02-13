import torch
import os

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_acc = 0.0  # Initialize at 0
        self.early_stop = False

    def __call__(self, val_acc):
        # If there is a real improvement (however small)
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
            print(f"[EarlyStopping] Improved accuracy! Patience reset.")
        else:
            self.counter += 1
            print(f"[EarlyStopping] Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def log_training(epoch_str, train_loss, val_loss, val_acc, lr):
    log_file = "entrenamiento_log.txt"
    modo = 'a' if os.path.exists(log_file) else 'w'
    with open(log_file, modo) as f:
        if modo == 'w':
            # Adjust the header to fit the phase name
            f.write("Fase-Ep | Train Loss | Val Loss | Val Acc | Learning Rate\n")
            f.write("-" * 65 + "\n")
        
        # Remove ':02d' and use a fixed space to look tidy
        f.write(f"{epoch_str:<7} | {train_loss:.4f} | {val_loss:.4f} | {val_acc:.4f} | {lr:.6e}\n")