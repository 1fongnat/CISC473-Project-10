import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from torchvision.utils import save_image
import numpy as np
import tqdm as tqdm_module
import math
from pytorch_msssim import ms_ssim
from torchvision import transforms
import pandas as pd

from src.datasets import CLICDataset
from src.model import ImageCompressionModel
# --- NEW IMPORT ---
from src.model_flextok import FlexTokImageCompressionModel
# ------------------
from src.utils import load_checkpoint, compute_padding
try:
    from src.perceptual_loss import PerceptualLoss
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

def compute_metrics(org, rec, max_val=1.0):
    rec = rec.clamp(0, 1)
    mse = torch.mean((org - rec) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
        psnr = psnr.item()
    ms_ssim_val = ms_ssim(org, rec, data_range=max_val).item()
    return psnr, ms_ssim_val

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Test dataset path")
    parser.add_argument("-c", "--ckpt_path", type=str, required=True, help="Path to checkpoint")
    
    # --- NEW ARGUMENT ---
    parser.add_argument("--use_flextok", action="store_true", help="Enable FlexTok architecture")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if LPIPS_AVAILABLE:
        lpips_scorer = PerceptualLoss(use_gpu=(device=="cuda")).to(device)
    
    # Full resolution testing
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = CLICDataset(os.path.join(args.dataset, "test"), transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    try:
        # --- INITIALIZE CORRECT MODEL ---
        if args.use_flextok:
            print(">>> Initializing FlexTok Model...")
            model = FlexTokImageCompressionModel().to(device)
        else:
            print(">>> Initializing Baseline/Perceptual Model...")
            model = ImageCompressionModel().to(device)
        
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        if hasattr(model, 'hyperprior') and hasattr(model.hyperprior, 'gaussian_conditional'):
             model.hyperprior.gaussian_conditional.update_scale_table([0.11, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0], force=True)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"Successfully loaded model from {args.ckpt_path}")
    except Exception as e:
         print(f"❌ Error loading model: {e}")
         return

    experiment_dir = os.path.dirname(os.path.dirname(args.ckpt_path))
    reconstruction_dir = os.path.join(experiment_dir, "reconstructions")
    os.makedirs(reconstruction_dir, exist_ok=True)

    results = [] 

    print("Starting evaluation...")
    with torch.no_grad():
        for i, d in enumerate(tqdm_module.tqdm(test_dataloader, desc="Testing")):
            if d is None: continue
            d = d.to(device)
            h, w = d.size(2), d.size(3)

            padding, unpadding = compute_padding(h, w, 64)
            d_padded = F.pad(d, padding, mode="replicate")

            try:
                out = model(d_padded)
                out["x_hat"] = F.pad(out["x_hat"], unpadding)
            except Exception as e:
                print(f"\n⚠️ Error during inference: {e}. Skipping.")
                continue

            try:
                if i < len(test_dataset.image_files):
                    image_name = os.path.basename(test_dataset.image_files[i])
                    save_image(out["x_hat"], os.path.join(reconstruction_dir, image_name))

                    psnr, mssim_val = compute_metrics(d, out["x_hat"])
                    
                    if LPIPS_AVAILABLE:
                        lpips_val = lpips_scorer(d, out["x_hat"]).mean().item()
                    else:
                        lpips_val = float('nan')

                    num_pixels = d.size(0) * h * w
                    if "likelihoods" in out and out["likelihoods"]:
                         bpp = sum((torch.log(lk.clamp(min=1e-9)).sum() / (-math.log(2) * num_pixels))
                                   for lk in out["likelihoods"].values() if isinstance(lk, torch.Tensor)).item()
                    else:
                        bpp = float('nan')

                    results.append({
                        "image": image_name, "psnr": psnr, "ms_ssim": mssim_val, "lpips": lpips_val, "bpp": bpp
                    })
            except Exception as e:
                 print(f"\n⚠️ Error processing metrics: {e}. Skipping.")

    if not results:
        print("❌ No images processed.")
        return

    df = pd.DataFrame(results)
    df.dropna(subset=['bpp', 'image'], inplace=True)
    metrics_csv_path = os.path.join(experiment_dir, "test_metrics.csv")
    df.to_csv(metrics_csv_path, index=False)
    
    print("\n--- Evaluation Results ---")
    if not df.empty:
        print(f"Average PSNR: {df['psnr'].mean():.4f}")
        print(f"Average MS-SSIM: {df['ms_ssim'].mean():.4f}")
        if LPIPS_AVAILABLE: print(f"Average LPIPS: {df['lpips'].mean():.4f}")
        print(f"Average BPP: {df['bpp'].mean():.4f}")

if __name__ == "__main__":
    test()