import os
import pandas as pd
import ast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ArtDataset(Dataset):
    def __init__(self, dataframe, root_dir, map_estilos, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.map_estilos = map_estilos
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, os.path.normpath(row['filename']))
        label = self.map_estilos[row['estilo']]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224)) # Fallback
        if self.transform:
            image = self.transform(image)
        return image, label

def preparar_dataloaders(ruta_csv, ruta_imgs, img_size=224, batch_size=32):
    df = pd.read_csv(ruta_csv)
    df = df[df['subset'] != 'uncertain artist']
    df['estilo'] = df['genre'].apply(lambda x: ast.literal_eval(x)[0])
    
    # Initial cleanup: styles with more than 500 images
    conteo = df['estilo'].value_counts()
    estilos_validos = conteo[conteo > 500].index
    df = df[df['estilo'].isin(estilos_validos)].copy()
    
    lista_clases = sorted(df['estilo'].unique().tolist())
    map_estilos = {estilo: i for i, estilo in enumerate(lista_clases)}
    
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    ds_train = ArtDataset(train_df, ruta_imgs, map_estilos, train_transforms)
    ds_val = ArtDataset(val_df, ruta_imgs, map_estilos, val_transforms)
    
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return loader_train, loader_val, lista_clases