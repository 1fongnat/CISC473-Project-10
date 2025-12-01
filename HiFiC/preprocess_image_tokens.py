from src.FlexTok.flextok.flextok_wrapper import FlexTokFromHub
from src.FlexTok.flextok.utils.demo import img_from_file, imgs_from_urls
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torch
import glob
import os
device = "cuda"

IMG_DIR = Path(__file__).parent / "data" / "clic_all" / "train"

# Load FlexTok model
model = FlexTokFromHub.from_pretrained('EPFL-VILAB/flextok_d18_d28_dfn').eval().to(device)

# Get all images
files = glob.glob("*.png", root_dir=IMG_DIR)

# Store tokens in dictionary
tokens_dict = dict()

# Load example images of shape (B, 3, 256, 256), normalized to [-1,1]
for f in files:
    img = img_from_file(os.path.join(IMG_DIR, f))
    img = img.to(device)
    # tokens_list is a list of [1, 256] discrete token sequences
    tokens_dict[f] = model.tokenize(img)
    # print(f, tokens_dict[f].shape())

torch.save(tokens_dict, os.path.join(IMG_DIR, "tokens_dict.pt"))



#############
# Demo Code #
#############

# # Load example images of shape (B, 3, 256, 256), normalized to [-1,1]
# imgs = imgs_from_urls(urls=['https://storage.googleapis.com/flextok_site/nb_demo_images/0.png'])
# imgs = imgs.to(device)

# # tokens_list is a list of [1, 256] discrete token sequences
# tokens_list = model.tokenize(imgs)

# # tokens_list is a list of [1, l] discrete token sequences, with l <= 256
# # reconst is a [B, 3, 256, 256] tensor, normalized to [-1,1]
# reconst = model.detokenize(
#     tokens_list,
#     timesteps=20, # Number of denoising steps
#     guidance_scale=7.5, # Classifier-free guidance scale
#     perform_norm_guidance=True, # See https://arxiv.org/abs/2410.02416
# )

# plt.imshow(reconst.permute(0, 2, 3, 1)[0].cpu().numpy() / 2 + 0.5)
# plt.show()

