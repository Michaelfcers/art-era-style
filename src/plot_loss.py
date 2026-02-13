import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_loss():
    # "Beauty" style configuration
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Read the log (adjust the separator if necessary)
    df = pd.read_csv('entrenamiento_log.txt', sep='|', skipinitialspace=True, skiprows=[1])
    df.columns = [c.strip() for c in df.columns]

    plt.plot(df['Fase-Ep'], df['Train Loss'], label='Train', color='#1f77b4', linewidth=2, marker='o', markersize=4)
    plt.plot(df['Fase-Ep'], df['Val Loss'], label='Validation', color='#ff7f0e', linewidth=2, linestyle='--', marker='s', markersize=4)

    plt.title('Loss History - ArtEra AI', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend(frameon=True, fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('loss_history.png', dpi=300)
    print("Image 'loss_history.png' generated.")

if __name__ == "__main__":
    plot_loss()