import logging
import os
import shutil
import torch
import torch.nn.functional as F

def logger_setup(logpath, filepath):
    """Configures a logger to save to a file and print to the console."""
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Console handler
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File handler
        file_handler = logging.FileHandler(logpath, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logging setup from {filepath}")
    return logger

def save_checkpoint(state, is_best, out_dir, filename='checkpoint.pth.tar'):
    """Saves a model checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(out_dir, filename), os.path.join(out_dir, 'model_best.pth.tar'))

def load_checkpoint(filepath, model, optimizer=None, aux_optimizer=None):
    """Loads a model checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")
        
    checkpoint = torch.load(filepath, map_location="cpu")
    
    # Adjust state_dict keys if saved with DataParallel
    state_dict = checkpoint["state_dict"]
    if all(key.startswith('module.') for key in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if aux_optimizer and "aux_optimizer" in checkpoint:
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        
    return checkpoint
import torch.nn.functional as F

def compute_padding(h, w, min_div=64):
    """
    Computes padding tuple for an image to make its dimensions divisible by min_div.
    Returns (padding, unpadding) tuples.
    """
    h_padded = (h + min_div - 1) // min_div * min_div
    w_padded = (w + min_div - 1) // min_div * min_div
    
    pad_top = (h_padded - h) // 2
    pad_bottom = h_padded - h - pad_top
    pad_left = (w_padded - w) // 2
    pad_right = w_padded - w - pad_left
    
    # PyTorch's F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    
    # Unpadding for cropping the output
    unpadding = (-pad_left, -pad_right, -pad_top, -pad_bottom)
    
    return padding, unpadding