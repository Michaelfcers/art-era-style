import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from data_setup import preparar_dataloaders
from utils import EarlyStopping, log_training
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import os

# ==========================================
# CONFIGURATION V2
# ==========================================
PHASE = 'V2'
IMG_SIZE = 256
BATCH_SIZE = 8       
ACCUM_STEPS = 4      
EPOCHS = 40          
LR = 5e-5

# Adjust paths according to your environment
RUTA_CSV = 'classes_clean.csv'
RUTA_IMGS = r'C:\Users\micha\OneDrive\Documentos\Proyectos\art-style\dataset_opt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================

def apply_mixup(inputs, labels, alpha=0.4):
    """Mixes two images by blending them."""
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(inputs.size(0)).to(DEVICE)
    mixed_x = lam * inputs + (1 - lam) * inputs[index, :]
    y_a, y_b = labels, labels[index]
    return mixed_x, y_a, y_b, lam

def apply_cutmix(inputs, labels, alpha=1.0):
    """Cuts a patch from one image and pastes it onto another."""
    indices = torch.randperm(inputs.size(0)).to(DEVICE)
    shuffled_inputs = inputs[indices]
    y_a = labels
    y_b = labels[indices]

    lam = np.random.beta(alpha, alpha)
    
    image_h, image_w = inputs.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    inputs[:, :, y0:y1, x0:x1] = shuffled_inputs[:, :, y0:y1, x0:x1]
    
    # Recalculate lambda based on the exact cropped area
    lam = 1 - ((x1 - x0) * (y1 - y0) / (image_w * image_h))
    return inputs, y_a, y_b, lam

def main():
    os.makedirs('checkpoints/v2', exist_ok=True)
    
    # Data preparation
    train_loader, val_loader, clases = preparar_dataloaders(RUTA_CSV, RUTA_IMGS, IMG_SIZE, BATCH_SIZE)

    print(f"Starting Training V2 (Mixup + CutMix)...")
    
    model = models.convnext_tiny(weights='IMAGENET1K_V1')
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(clases))
    model = model.to(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': LR / 10},
        {'params': model.classifier.parameters(), 'lr': LR}
    ], weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader)//ACCUM_STEPS, epochs=EPOCHS
    )
    
    # Label smoothing helps with generalization in art
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    scaler = GradScaler('cuda')
    early_stopping = EarlyStopping(patience=10) # Patience adjusted for V2
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS} [V2]")
        
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # === Augmentation Logic (Mixup/CutMix) ===
            # Total probability of applying some mix: 60%
            # Within that: 50% Mixup, 50% CutMix
            apply_aug = np.random.rand() < 0.6
            
            if apply_aug:
                if np.random.rand() > 0.5:
                    # Mixup
                    inputs, labels_a, labels_b, lam = apply_mixup(inputs, labels, alpha=0.4)
                else:
                    # CutMix
                    inputs, labels_a, labels_b, lam = apply_cutmix(inputs, labels, alpha=1.0)
                
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                # Normal
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

            # === Gradient Accumulation ===
            loss = loss / ACCUM_STEPS
            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item() * ACCUM_STEPS
            pbar.set_postfix(loss=f"{loss.item()*ACCUM_STEPS:.4f}")

        # --- VALIDATION ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        
        # We use criterion WITHOUT smoothing for pure validation if we want strict metric,
        # but for loss consistency we use the same criterion.
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) # Loss with smoothing
                    v_loss += loss.item()
                    
                    # Real Accuracy
                    _, preds = torch.max(outputs, 1)
                    v_correct += (preds == labels).sum().item()
                    v_total += labels.size(0)

        val_acc = v_correct / v_total
        val_loss = v_loss / len(val_loader)
        
        log_training(f"V2-{epoch+1}", running_loss/len(train_loader), val_loss, val_acc, optimizer.param_groups[1]['lr'])

        # Checkpoint saving
        # We only save the best one to save space, or the last one
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'checkpoints/v2/best_v2_model.pth')
            print(f"New record: {val_acc:.4f} - Saved.")
        
        # Optional: Save last state always in case of power failure
        torch.save(model.state_dict(), f'checkpoints/v2/last_v2_model.pth')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping reached.")
            break

if __name__ == '__main__':
    main()