import torch
import torch.nn as nn
from torchvision import models
# Import existing data logic
from data_setup import preparar_dataloaders
from utils import EarlyStopping, log_training
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import os
import numpy as np

# ==========================================
# CONFIGURATION V3 - HIGH RES FINE-TUNING
# ==========================================
PHASE = 'V3-HD'
IMG_SIZE = 384       # <--- INCREASE RESOLUTION (Key for textures)
BATCH_SIZE = 4       # <--- LOWER BATCH (To fit in 4GB VRAM)
ACCUM_STEPS = 8      # <--- INCREASE ACCUMULATION (Effective batch = 32)
EPOCHS = 10          # Few epochs, it's just fine-tuning
LR = 1e-5            # VERY LOW learning rate (to not break what was learned)

# PATH TO CHECKPOINT V2 (Your best current model)
# PLACEHOLDER: Insert your checkpoint path here or download it from:
# https://huggingface.co/michaelrodcs/art-style-convnext
LOAD_CHECKPOINT = 'checkpoints/v2/best_v2_model.pth' 

# IMPORTANT! Change this to the folder with ORIGINAL or 400px images
RUTA_IMGS = r'C:\Users\micha\OneDrive\Documentos\Proyectos\art-style\dataset400' 
RUTA_CSV = 'classes_clean.csv'
DEVICE = torch.device("cuda")
# ==========================================

def main():
    os.makedirs('checkpoints/v3_hd', exist_ok=True)
    
    # Prepare dataloaders with NEW resolution
    print(f"--- Loading data at {IMG_SIZE}x{IMG_SIZE} ---")
    train_loader, val_loader, clases = preparar_dataloaders(RUTA_CSV, RUTA_IMGS, IMG_SIZE, BATCH_SIZE)

    print(f"Starting Fine-Tuning HD from {LOAD_CHECKPOINT}...")
    
    # 1. Create Architecture
    model = models.convnext_tiny()
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(clases))
    
    # 2. Load the trained brain (V2)
    if os.path.exists(LOAD_CHECKPOINT):
        state_dict = torch.load(LOAD_CHECKPOINT, map_location=DEVICE)
        # Input layers might give error if size changed, but ConvNeXt is flexible.
        # ConvNeXt uses Global Average Pooling, so it accepts any input size.
        model.load_state_dict(state_dict)
        print("V2 Brain loaded. Now putting on reading glasses.")
    else:
        print("ERROR: V2 checkpoint not found. Aborting.")
        return

    model = model.to(DEVICE)

    # 3. Slow Optimizer (Fine-tuning)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    
    # Soft Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader)//ACCUM_STEPS, epochs=EPOCHS,
        pct_start=0.1 # Fast warmup
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler('cuda')
    early_stopping = EarlyStopping(patience=4) # Less patience, if it doesn't improve, cut
    best_acc = 0.0

    # --- SIMPLIFIED TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS} [HD]")
        
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # In Fine-Tuning HD, sometimes it's better NOT to use excessive Mixup/Cutmix
            # so the model focuses on real textures.
            # We will use standard training here for maximum sharpness.
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

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
        v_correct, v_total = 0, 0
        v_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    v_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    v_correct += (preds == labels).sum().item()
                    v_total += labels.size(0)

        val_acc = v_correct / v_total
        val_loss_avg = v_loss / len(val_loader)
        
        log_training(f"V3-HD-{epoch+1}", running_loss/len(train_loader), val_loss_avg, val_acc, optimizer.param_groups[0]['lr'])

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'checkpoints/v3_hd/best_hd_model.pth')
            print(f"HD Record: {val_acc:.4f}")

        early_stopping(val_loss_avg)
        if early_stopping.early_stop:
            print("Early stopping in HD.")
            break

if __name__ == '__main__':
    main()