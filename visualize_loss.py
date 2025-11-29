import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_loss(history_file):
    try:
        df = pd.read_csv(history_file)
        print(f"Loaded history from {history_file}")
    except FileNotFoundError:
        print(f"❌ Error: History file '{history_file}' not found.")
        print("Did you run the updated train.py?")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Lines
    ax.plot(df['epoch'], df['train_loss'], label='Training Loss', color='blue', linewidth=2)
    ax.plot(df['epoch'], df['test_loss'], label='Testing Loss', color='orange', linewidth=2, linestyle='--')

    # Formatting
    ax.set_title("Training vs. Testing Loss", fontsize=16, weight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Total Loss Value", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Save
    model_dir = os.path.dirname(history_file)
    output_file = os.path.join(model_dir, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"✅ Loss graph saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    # Change this path to the experiment you want to graph
    # Example: "experiments/perceptual_baseline/loss_history.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to loss_history.csv")
    args = parser.parse_args()
    
    plot_loss(args.file)