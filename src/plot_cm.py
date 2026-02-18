import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from torchvision import models
from tqdm import tqdm
import os

# Import existing data logic for consistency
from data_setup import preparar_dataloaders

# ================= CONFIGURATION =================
# Ensure this path points to your BEST model (e.g., best_v2_model.pth)
# PLACEHOLDER: Insert your checkpoint path here or download it from:
# https://huggingface.co/michaelrodcs/art-style-convnext
CHECKPOINT_PATH = 'checkpoints/v3_hd/best_hd_model.pth' 
RUTA_CSV = 'classes_clean.csv'
RUTA_IMGS = r'C:\Users\micha\OneDrive\Documentos\Proyectos\art-style\dataset_opt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================

def plot_confusion_matrix():
    print("--- Loading Original Dataloaders ---")
    # Using batch_size=32 for faster inference
    _, val_loader, clases = preparar_dataloaders(RUTA_CSV, RUTA_IMGS, img_size=256, batch_size=32)
    
    print(f"Classes detected ({len(clases)}): {clases}")
    
    # 1. Reconstruct the model EXACTLY as trained
    print("--- Loading Model ---")
    model = models.convnext_tiny()
    # Adjust final layer to exact number of classes from training
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(clases))
    
    # Load weights
    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    else:
        print(f"Error: File {CHECKPOINT_PATH} not found.")
        return

    model.to(DEVICE)
    model.eval()

    # 2. Get predictions
    all_preds = []
    all_labels = []

    print("--- Generating Predictions ---")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 3. Generate Matrix
    print("--- Drawing Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    # Normalize to see percentages (better with imbalanced classes)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Pretty names for plot (remove underscores)
    clases_bonitas = [c.replace('_', ' ') for c in clases]

    plt.figure(figsize=(20, 18)) # Large size to fit 22 classes
    sns.heatmap(cm_perc, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=clases_bonitas, yticklabels=clases_bonitas,
                cbar_kws={'label': 'Confidence Scale (0-1)'})
    
    plt.title(f'Normalized Confusion Matrix - {len(clases)} Art Styles', fontsize=20, pad=20)
    plt.ylabel('Ground Truth Style', fontsize=15)
    plt.xlabel('AI Prediction', fontsize=15)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    output_file = 'matriz_confusion_final.png'
    plt.savefig(output_file, dpi=300)
    print(f"Ready! Image saved as: {output_file}")

if __name__ == '__main__':
    plot_confusion_matrix()