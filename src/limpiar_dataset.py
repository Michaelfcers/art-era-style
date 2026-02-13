import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
CSV_ORIGINAL = 'classes.csv'
RUTA_IMGS = r'C:\Users\micha\OneDrive\Documentos\Proyectos\art-style\dataset_opt'
MIN_IMGS_POR_ESTILO = 500

def limpiar():
    print("--- Starting Deep Dataset Cleanup ---")
    df = pd.read_csv(CSV_ORIGINAL)
    total_inicial = len(df)

    # 1. Remove rows without physical image or corrupt
    print("Verifying file integrity...")
    indices_validos = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(RUTA_IMGS, os.path.normpath(row['filename']))
        if os.path.exists(img_path):
            try:
                # We only verify that it is a valid image without loading it all into memory
                with Image.open(img_path) as img:
                    img.verify() 
                indices_validos.append(idx)
            except:
                continue
    
    df = df.loc[indices_validos].copy()
    print(f"Files verified. Deleted: {total_inicial - len(df)}")

    # 2. Remove duplicates by PHASH
    antes_dup = len(df)
    df = df.drop_duplicates(subset=['phash'])
    print(f"Duplicates removed (PHASH): {antes_dup - len(df)}")

    # 3. Filter by ambiguity (only 1 genre per frame)
    antes_amb = len(df)
    df = df[df['genre_count'] == 1]
    print(f"Ambiguous frames removed: {antes_amb - len(df)}")

    # 4. Clean style names and filter by volume
    import ast
    df['estilo'] = df['genre'].apply(lambda x: ast.literal_eval(x)[0])
    
    conteo = df['estilo'].value_counts()
    estilos_finales = conteo[conteo >= MIN_IMGS_POR_ESTILO].index
    df = df[df['estilo'].isin(estilos_finales)]
    
    print(f"\nFinal Summary:")
    print(f"- Remaining styles: {len(estilos_finales)}")
    print(f"- Total images: {len(df)}")
    print(f"- Total data loss from cleanup: {100 - (len(df)/total_inicial*100):.2f}%")

    # Save Clean CSV
    df.to_csv('classes_clean.csv', index=False)
    print("\nFile 'classes_clean.csv' saved successfully.")

if __name__ == '__main__':
    limpiar()