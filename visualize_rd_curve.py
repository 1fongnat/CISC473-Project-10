import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def get_metrics_from_folder(folder_name):
    """
    Reads test_metrics.csv from the experiment folder and returns average
    [BPP, PSNR, MS-SSIM, LPIPS].
    """
    csv_path = os.path.join("experiments", folder_name, "test_metrics.csv")
    
    if not os.path.exists(csv_path):
        print(f"⚠️ Warning: Could not find '{csv_path}'. Skipping this point.")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Calculate averages
        avg_bpp = df['bpp'].mean()
        avg_psnr = df['psnr'].replace([np.inf, -np.inf], np.nan).mean()
        avg_msssim = df['ms_ssim'].mean()
        
        # Check if LPIPS column exists (it might not for old baseline runs)
        if 'lpips' in df.columns:
            avg_lpips = df['lpips'].mean()
        else:
            avg_lpips = float('nan') # Metric not available
            
        print(f"✅ Loaded {folder_name}: BPP={avg_bpp:.3f}, PSNR={avg_psnr:.2f}")
        return [avg_bpp, avg_psnr, avg_msssim, avg_lpips]
        
    except Exception as e:
        print(f"❌ Error reading {csv_path}: {e}")
        return None

def plot_rd_curves(baseline_folders, perceptual_folders):
    """
    Gather data and plot RD curves.
    """
    
    # --- 1. Gather Data ---
    baseline_data = []
    for folder in baseline_folders:
        metrics = get_metrics_from_folder(folder)
        if metrics: baseline_data.append(metrics)

    perceptual_data = []
    for folder in perceptual_folders:
        metrics = get_metrics_from_folder(folder)
        if metrics: perceptual_data.append(metrics)

    if not baseline_data and not perceptual_data:
        print("❌ No data found. Check your folder names.")
        return

    # Sort by BPP (Index 0) to ensure lines connect correctly from left to right
    baseline_data.sort(key=lambda x: x[0])
    perceptual_data.sort(key=lambda x: x[0])

    # --- 2. Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Metrics config: (Name, Index in list, Lower is Better?)
    metrics_config = [
        ("PSNR", 1, False),
        ("MS-SSIM", 2, False),
        ("LPIPS", 3, True)
    ]

    # Extract BPPs for x-axis
    b_bpp = [x[0] for x in baseline_data]
    p_bpp = [x[0] for x in perceptual_data]

    # --- 3. Generate 3 Plots ---
    for metric_name, idx, lower_is_better in metrics_config:
        fig, ax = plt.subplots(figsize=(9, 6))
        
        # Extract Metric Values
        b_vals = [x[idx] for x in baseline_data]
        p_vals = [x[idx] for x in perceptual_data]

        # Check if we have valid data for this metric (e.g., skip LPIPS if NaN)
        if all(np.isnan(v) for v in b_vals) and all(np.isnan(v) for v in p_vals):
            print(f"Skipping {metric_name} plot (no data).")
            continue

        # Plot Curves
        if b_vals and not all(np.isnan(b_vals)):
            ax.plot(b_bpp, b_vals, marker='o', linewidth=2, label='Baseline (MSE)', color='skyblue')
        
        if p_vals and not all(np.isnan(p_vals)):
            ax.plot(p_bpp, p_vals, marker='s', linewidth=2, label='Perceptual (LPIPS)', color='orange')

        # Formatting
        ax.set_title(f'Rate-Distortion: {metric_name} vs. Bitrate', fontsize=14, weight='bold')
        ax.set_xlabel('Bitrate (BPP)', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        # Invert axis for LPIPS (since lower is better)
        if lower_is_better:
            ax.invert_yaxis()
            ax.set_title(f'{metric_name} vs. Bitrate (Lower is Better)', fontsize=14, weight='bold')

        # Save
        filename = f'rd_curve_{metric_name.lower()}.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"📈 Saved {filename}")
        plt.close(fig)

if __name__ == '__main__':
    # =========================================================
    #  CONFIGURATION: EDIT YOUR FOLDER NAMES HERE
    # =========================================================
    
    # List the exact folder names inside 'experiments/' for your 3 Baseline points
    # Example: ['baseline_low', 'baseline_med', 'baseline_high']
    baseline_experiment_names = [
        'baseline_low', 
        'baseline_med', 
        'baseline_high'
    ]

    # List the exact folder names inside 'experiments/' for your 3 Perceptual points
    perceptual_experiment_names = [
        'perceptual_low', 
        'perceptual_med', 
        'perceptual_high'
    ]

    # =========================================================
    
    plot_rd_curves(baseline_experiment_names, perceptual_experiment_names)