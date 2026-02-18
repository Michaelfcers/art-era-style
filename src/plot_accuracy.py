import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_accuracy():
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    df = pd.read_csv('logs/v2_training_log.txt', sep='|', skipinitialspace=True, skiprows=[1])
    df.columns = [c.strip() for c in df.columns]

    # Convert to percentage if it is not
    acc_values = df['Val Acc'] * 100 if df['Val Acc'].max() <= 1.0 else df['Val Acc']

    plt.fill_between(df['Fase-Ep'], acc_values, color="#2ca02c", alpha=0.1)
    plt.plot(df['Fase-Ep'], acc_values, color='#2ca02c', linewidth=3, marker='^', markersize=6)

    plt.title('Accuracy Evolution', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(min(acc_values)-5, max(acc_values)+5)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('accuracy_evolution.png', dpi=300)
    print("Image 'accuracy_evolution.png' generated.")

if __name__ == "__main__":
    plot_accuracy()