import torch
import torch.nn as nn
from torchvision import models
from data_setup import preparar_dataloaders
from utils import EarlyStopping, log_training
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import os

# ==========================================
# ⚙️ PHASE CONFIGURATION
# ==========================================
PHASE = 'C'          # 'A', 'B' or 'C'
IMG_SIZE = 256      # A: 128, B: 224, C: 256
BATCH_SIZE = 8    # A: 32, B: 16, C: 12 or 16 (according to VRAM)
EPOCHS = 15         # How many epochs you want for this phase
LR = 1e-5           # Phase A: 4e-4, Phase B: 1e-4, Phase C: 1e-5 (Fine-tuning)

# 🔗 PATH TO THE BEST CHECKPOINT OF THE PREVIOUS PHASE
LOAD_CHECKPOINT = 'best_phase_B_cont.pth' 

# 📂 PATHS
RUTA_CSV = 'classes_clean.csv'
RUTA_IMGS = r'C:\Users\micha\OneDrive\Documentos\Proyectos\art-style\dataset_opt'
DEVICE = torch.device("cuda")
# ==========================================

def main():
    os.makedirs('checkpoints', exist_ok=True)
    train_loader, val_loader, clases = preparar_dataloaders(RUTA_CSV, RUTA_IMGS, IMG_SIZE, BATCH_SIZE)

    print(f"Starting PHASE {PHASE} at {IMG_SIZE}px...")
    model = models.convnext_tiny(weights='IMAGENET1K_V1')
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(clases))
    model = model.to(DEVICE)

    # --- LOAD PREVIOUS PROGRESS ---
    if LOAD_CHECKPOINT:
        print(f"Loading weights from {LOAD_CHECKPOINT}...")
        model.load_state_dict(torch.load(LOAD_CHECKPOINT))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, 
                                                    steps_per_epoch=len(train_loader), epochs=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler('cuda')
    early_stopping = EarlyStopping(patience=7)
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"PHASE {PHASE} | Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- VALIDATION ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
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
        val_loss = v_loss / len(val_loader)
        
        # Save to Logs including the Phase
        log_training(f"{PHASE}-{epoch+1}", running_loss/len(train_loader), val_loss, val_acc, optimizer.param_groups[0]['lr'])

        # --- SAVING CHECKPOINTS ---
        # 1. Save all checkpoints of the phase
        cp_name = f'checkpoints/phase_{PHASE}_ep{epoch+1}.pth'
        torch.save(model.state_dict(), cp_name)

        # 2. Save the "Selected Best" of this phase
        if val_acc > best_acc:
            best_acc = val_acc
            best_name = f'best_phase_{PHASE}.pth'
            torch.save(model.state_dict(), best_name)
            print(f"New record Phase {PHASE}: {val_acc:.4f} -> Saved as {best_name}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Phase {PHASE} finished by Early Stopping.")
            break

if __name__ == '__main__':
    main()