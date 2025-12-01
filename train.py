"""
FILE: train.py
DESCRIPTION:
This is the main training script for the Deep Learning Image Compression project.
It handles the complete training pipeline, including:
1. Data Loading: Loading and preprocessing training and validation images.
2. Model Initialization: Setting up the Baseline, Perceptual, or FlexTok models.
3. Training Loop: Iterating through epochs to optimize model weights.
4. Validation: Calculating test loss during training to monitor generalization.
5. Logging & Checkpointing: Saving loss history and model weights for future use.

KEY FEATURES for REPORT:
- Supports multiple model architectures via the --use_flextok flag.
- Implements a custom Rate-Distortion Loss function.
- Uses separate optimizers for the main network and the entropy bottleneck (auxiliary).
"""

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

# Try importing FlexTok model if needed (Modular architecture support)
try:
    from src.model_flextok import FlexTokImageCompressionModel
except ImportError:
    FlexTokImageCompressionModel = None

def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, logger):
    """
    Executes one full pass (epoch) over the training data.
    Calculates gradients and updates model weights to minimize the Rate-Distortion loss.
    """
    model.train()
    device = next(model.parameters()).device
    epoch_loss = 0.0

    # Use leave=False to keep the progress bar cleaner
    pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch}", leave=False)
    
    for i, d in enumerate(pbar):
        d = d.to(device) # Move batch to GPU/CPU

        # Zero out gradients before backpropagation
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # Forward pass: Get reconstructed image and likelihoods
        out_net = model(d)
        
        # Calculate Rate-Distortion Loss (MSE + BPP + LPIPS)
        out_criterion = criterion(out_net, d)
        
        # Backward pass (Compute gradients)
        out_criterion["loss"].backward()
        
        # Gradient Clipping prevents exploding gradients, stabilizing training
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step() # Update main model weights

        # Update Entropy Bottleneck (Auxiliary Loss)
        # This is crucial for learning accurate probability estimates for BPP calculation
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        
        loss_val = out_criterion["loss"].item()
        epoch_loss += loss_val
        
        # Update progress bar with current loss
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    return epoch_loss / len(train_dataloader)

def test_epoch(model, criterion, test_dataloader):
    """
    Runs a validation pass on the test set without updating weights.
    Used to calculate Average Test Loss and detect overfitting.
    """
    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    device = next(model.parameters()).device
    epoch_loss = 0.0
    
    with torch.no_grad(): # Disable gradient calculation to save memory
        for d in tqdm(test_dataloader, desc="Validating", leave=False):
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            epoch_loss += out_criterion["loss"].item()
            
    return epoch_loss / len(test_dataloader)

def main():
    parser = argparse.ArgumentParser(description="Training script with Loss History.")
    # ... (Arguments setup code) ...
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
    parser.add_argument("--use_saliency", action="store_true", help="Enable Saliency/ROI Weighted Compression")
    args = parser.parse_args()

    # Reproducibility: Set seed for consistent results across runs
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    experiment_dir = os.path.join("experiments", args.model_type)
    os.makedirs(experiment_dir, exist_ok=True)
    
    logger = logger_setup(logpath=os.path.join(experiment_dir, "logs.txt"), filepath=os.path.abspath(__file__))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # --- DataLoaders ---
    # Training Data: Uses random crops (patch_size=256) to augment data
    train_dataset = CLICDataset(os.path.join(args.dataset, "train"), patch_size=256)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, pin_memory=(device == "cuda"))
    
    # Validation Data: Uses CenterCrop to ensure consistent validation metrics
    from torchvision import transforms
    val_transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
    test_dataset = CLICDataset(os.path.join(args.dataset, "test"), transform=val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    # --- Model Selection Logic ---
    # dynamically selects between the standard Baseline/Perceptual model 
    # and the advanced FlexTok model based on arguments.
    if args.use_flextok:
        if FlexTokImageCompressionModel is None:
             raise ImportError("FlexTokImageCompressionModel not found. Check src/model_flextok.py")
        print(">>> Initializing FlexTok Model...")
        model = FlexTokImageCompressionModel()
    else:
        print(">>> Initializing Baseline/Perceptual Model...")
        model = ImageCompressionModel()
        
    model = model.to(device)
    
    # Setup Optimizers: One for weights, one for quantization parameters
    optimizer, aux_optimizer = model.configure_optimizers(args)
    
    # Loss Function: Combines Rate (BPP) + Distortion (MSE) + Perception (LPIPS)
    criterion = RateDistortionLoss(
        lmbda=args.lmbda, 
        lmbda_lpips=args.lmbda_lpips, 
        use_saliency=args.use_saliency
    )
    
    # --- Training Loop with History Tracking ---
    history = []

    for epoch in range(args.epochs):
        # 1. Train
        train_loss = train_one_epoch(
            model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm, logger
        )
        
        # 2. Validate (Test Loss)
        test_loss = test_epoch(model, criterion, test_dataloader)
        
        # 3. Log and Save Results
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Test Loss={test_loss:.4f}")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss
        })
        
        # Save CSV continuously to allow for loss curve plotting later
        pd.DataFrame(history).to_csv(os.path.join(experiment_dir, "loss_history.csv"), index=False)

        # Checkpoint Saving
        if args.save:
            save_checkpoint(
                {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
                is_best=False,
                out_dir=os.path.join(experiment_dir, "checkpoints")
            )

if __name__ == "__main__":
    main()