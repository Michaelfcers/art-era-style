import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import shutil

# ================= CONFIGURATION =================
# Folder with ORIGINAL images (Giant ones)
INPUT_FOLDER = r'C:\Users\micha\Downloads\archive' 
# Folder where new images will be saved (Ready for training)
OUTPUT_FOLDER = r'C:\Users\micha\OneDrive\Documentos\Proyectos\art-style\dataset400'

TARGET_SIZE = 400  # A bit huge than 384 to give margin for RandomCrop
QUALITY = 90       # JPG Quality
# =================================================

def procesar_imagen(args):
    ruta_origen, ruta_destino = args
    
    # If it already exists, skip (useful if it stops and you resume)
    if os.path.exists(ruta_destino):
        return
    
    try:
        with Image.open(ruta_origen) as img:
            # Convert to RGB (in case of transparent PNGs or CMYK)
            img = img.convert('RGB')
            
            # Calculate new height maintaining aspect ratio
            ratio = float(TARGET_SIZE) / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            
            # Resize (LANCZOS is best quality for downscaling)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save
            img.save(ruta_destino, "JPEG", quality=QUALITY, optimize=True)
    except Exception as e:
        print(f"Error in {os.path.basename(ruta_origen)}: {e}")

def main():
    if not os.path.exists(INPUT_FOLDER):
        print("Error: Input folder not found.")
        return

    tareas = []
    print(f"Scanning files in {INPUT_FOLDER}...")

    # Traverse folders recursively
    for root, dirs, files in os.walk(INPUT_FOLDER):
        # Create mirror structure in destination
        relative_path = os.path.relpath(root, INPUT_FOLDER)
        dest_dir = os.path.join(OUTPUT_FOLDER, relative_path)
        os.makedirs(dest_dir, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                origen = os.path.join(root, file)
                destino = os.path.join(dest_dir, os.path.splitext(file)[0] + '.jpg')
                tareas.append((origen, destino))

    print(f"Processing {len(tareas)} images at {TARGET_SIZE}px...")
    
    # Use multiple cores to fly
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(procesar_imagen, tareas), total=len(tareas), unit="img"))

    print("\nReady! HD Dataset created.")
    print(f"New path for your training script: {OUTPUT_FOLDER}")

if __name__ == '__main__':
    # Needed in Windows for multiprocessing
    main()