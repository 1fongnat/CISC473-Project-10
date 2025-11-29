import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import random
import numpy as np
import pandas as pd  # Added for saving history

from src.datasets import CLICDataset
from src.model import ImageCompressionModel
from src.loss import RateDistortionLoss
from src.utils import logger_setup, save_checkpoint

# Try importing FlexTok model if needed
try:
    from src.model_flextok import FlexTokImageCompressionModel
except ImportError:
    FlexTokImageCompressionModel = None

def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger):
    """Trains the model for one epoch and returns the average training loss."""
    model.train()
    device = next(model.parameters()).device
    epoch_loss = 0.0

    # Use leave=False to keep the progress bar cleaner
    pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch}", leave=False)
    
    for i, d in enumerate(pbar):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)
        
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        
        loss_val = out_criterion["loss"].item()
        epoch_loss += loss_val
        
        # Update progress bar with current loss
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return epoch_loss / len(train_dataloader)

def test_epoch(model, criterion, test_dataloader):
    """Runs validation on the test set to calculate Average Test Loss."""
    model.eval()
    device = next(model.parameters()).device
    epoch_loss = 0.0
    
    with torch.no_grad():
        for d in tqdm(test_dataloader, desc="Validating", leave=False):
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            epoch_loss += out_criterion["loss"].item()
            
    return epoch_loss / len(test_dataloader)

def main():
    parser = argparse.ArgumentParser(description="Training script with Loss History.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset path (root folder).")
    parser.add_argument("-m", "--model_type", type=str, default="baseline", help="Model name.")
    parser.add_argument("-e", "--epochs", default=100, type=int, help="Number of training epochs.")
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--aux_learning_rate", default=1e-3, type=float, help="Auxiliary loss learning rate.")
    parser.add_argument("--lmbda", type=float, default=0.01, help="Rate-distortion parameter.")
    parser.add_argument("--lmbda_lpips", type=float, default=10.0, help="Perceptual loss weight.")
    parser.add_argument("--clip_max_norm", default=1.0, type=float, help="Gradient clipping.")
    parser.add_argument("--save", action="store_true", help="Save checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_flextok", action="store_true", help="Enable FlexTok architecture")
    
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    experiment_dir = os.path.join("experiments", args.model_type)
    os.makedirs(experiment_dir, exist_ok=True)
    
    logger = logger_setup(logpath=os.path.join(experiment_dir, "logs.txt"), filepath=os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # --- DataLoaders ---
    # Training Data
    train_dataset = CLICDataset(os.path.join(args.dataset, "train"), patch_size=256)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, pin_memory=(device == "cuda"))
    
    # Validation Data (Using CenterCrop 256 to ensure it fits in memory during training)
    from torchvision import transforms
    val_transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
    test_dataset = CLICDataset(os.path.join(args.dataset, "test"), transform=val_transform)
    # Using same batch size as training for validation speed
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    # --- Model Selection ---
    if args.use_flextok:
        if FlexTokImageCompressionModel is None:
             raise ImportError("FlexTokImageCompressionModel not found. Check src/model_flextok.py")
        print(">>> Initializing FlexTok Model...")
        model = FlexTokImageCompressionModel()
    else:
        print(">>> Initializing Baseline/Perceptual Model...")
        model = ImageCompressionModel()
        
    model = model.to(device)
    
    optimizer, aux_optimizer = model.configure_optimizers(args)
    criterion = RateDistortionLoss(lmbda=args.lmbda, lmbda_lpips=args.lmbda_lpips)
    
    # --- Training Loop with History ---
    history = []

    for epoch in range(args.epochs):
        # 1. Train
        train_loss = train_one_epoch(
            model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm, logger
        )
        
        # 2. Validate (Test Loss)
        test_loss = test_epoch(model, criterion, test_dataloader)
        
        # 3. Log and Save
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Test Loss={test_loss:.4f}")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss
        })
        
        # Save CSV continuously so you don't lose data if it crashes
        pd.DataFrame(history).to_csv(os.path.join(experiment_dir, "loss_history.csv"), index=False)

        if args.save:
            save_checkpoint(
                {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
                is_best=False,
                out_dir=os.path.join(experiment_dir, "checkpoints")
            )

if __name__ == "__main__":
    main()