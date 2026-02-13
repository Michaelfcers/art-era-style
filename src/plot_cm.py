import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# --- EXACT LIST OF THE 21 FOLDERS THAT WERE TRAINED ---
# Note: We use the exact names of your folders with underscores
CARPETAS_ENTRENADAS = [
    'Abstract_Expressionism', 'Art_Nouveau_Modern', 'Baroque', 
    'Color_Field_Painting', 'Cubism', 'Early_Renaissance', 
    'Expressionism', 'Fauvism', 'High_Renaissance', 'Impressionism', 
    'Mannerism_Late_Renaissance', 'Minimalism', 'Naive_Art_Primitivism', 
    'Northern_Renaissance', 'Pop_Art', 'Post_Impressionism', 'Realism', 
    'Rococo', 'Romanticism', 'Symbolism', 'Ukiyo_e'
]

def plot_cm():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. LOAD MODEL (Forces 21 outputs)
    model = models.convnext_tiny()
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 21)
    model.load_state_dict(torch.load('best_phase_C.pth', map_location=DEVICE))
    model.to(DEVICE).eval()

    # 2. LOAD DATASET
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_ds = datasets.ImageFolder('dataset_opt', transform=transform)
    
    # Create a filter: we only want images whose folders are in CARPETAS_ENTRENADAS
    indices_validos = [i for i, (path, label) in enumerate(full_ds.imgs) 
                      if full_ds.classes[label] in CARPETAS_ENTRENADAS]
    
    # Create a subset of the dataset with only the 21 classes
    subset_ds = torch.utils.data.Subset(full_ds, indices_validos)
    loader = DataLoader(subset_ds, batch_size=32, shuffle=False)

    # Mapping so that the alphabetical order of the folders matches the diagonal
    # PyTorch assigns indices 0-20 following the alphabetical order of the folders
    clases_alfabeticas = sorted(CARPETAS_ENTRENADAS)

    all_preds, all_labels = [], []
    print(f"Analyzing {len(subset_ds)} images of the 21 official styles...")

    with torch.no_grad():
        for inputs, labels in loader:
            # Get the real folder name
            for i in range(len(labels)):
                nombre_carpeta = full_ds.classes[labels[i]]
                # Search what index (0-20) that name has in the alphabetical list
                all_labels.append(clases_alfabeticas.index(nombre_carpeta))
            
            outputs = model(inputs.to(DEVICE))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    # 3. GENERATE MATRIX
    cm = confusion_matrix(all_labels, all_preds)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 4. DRAW (Replace underscores with spaces to look nice in the graph)
    clases_bonitas = [c.replace('_', ' ') for c in clases_alfabeticas]

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_perc, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=clases_bonitas, yticklabels=clases_bonitas)
    
    plt.title('Confusion Matrix - ArtEra AI\n(Aligned Diagonal)', fontsize=16, pad=20)
    plt.ylabel('Real Style', fontsize=12)
    plt.xlabel('AI Prediction', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('confusion_matrix_clean.png', dpi=300)
    print("Matrix generated! The diagonal should be perfect now.")

if __name__ == "__main__":
    plot_cm()