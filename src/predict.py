import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- CONFIGURATION ---
CLASES = [
    'Abstract Expressionism', 'Art Nouveau Modern', 'Baroque', 
    'Color Field Painting', 'Cubism', 'Early Renaissance', 
    'Expressionism', 'Fauvism', 'High Renaissance', 'Impressionism', 
    'Mannerism Late Renaissance', 'Minimalism', 'Naive Art Primitivism', 
    'Northern Renaissance', 'Pop Art', 'Post Impressionism', 'Realism', 
    'Rococo', 'Romanticism', 'Symbolism', 'Ukiyo e'
]

# Simplified list of the 21 classes
NAMES = [
    "Abstract_Expressionism", "Art_Nouveau_Modern", "Baroque", 
    "Color_Field_Painting", "Cubism", "Early_Renaissance", 
    "Expressionism", "Fauvism", "High_Renaissance", "Impressionism", 
    "Mannerism_Late_Renaissance", "Minimalism", "Naive_Art_Primitivism", 
    "Northern_Renaissance", "Pop_Art", "Post_Impressionism", "Realism", 
    "Rococo", "Romanticism", "Symbolism", "Ukiyo-e"
]

from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
REPO_ID = "michaelrodcs/art-style-convnext"
FILENAME = "art-style-convnext.pth"          

MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cargar_modelo():
    print(f"Loading model on {DEVICE}...")
    model = models.convnext_tiny()
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predecir(model, image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        # Clean the path (remove quotes if the file was dragged)
        image_path = image_path.strip().replace('"', '').replace("'", "")
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Obtener los 3 mejores resultados
        top_probs, top_indices = torch.topk(probs, 3)
        
        print("\nAnalysis results:")
        print("-" * 30)
        for i in range(3):
            estilo = CLASES[top_indices[i]]
            confianza = top_probs[i].item() * 100
            print(f"{i+1}. {estilo:<20} | {confianza:>6.2f}%")
        print("-" * 30)

    except Exception as e:
        print(f"Error processing image: {e}")

def main():
    model = cargar_modelo()
    print("\nReady! To exit type 'exit'.")
    
    while True:
        print("\nDrag an image here and press Enter:")
        path = input("> ")
        
        if path.lower() == 'exit':
            break
        
        if os.path.exists(path.strip().replace('"', '').replace("'", "")):
            predecir(model, path)
        else:
            print("The file does not exist. Try again.")

if __name__ == "__main__":
    main()