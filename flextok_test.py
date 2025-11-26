from flextok.flextok_wrapper import FlexTokFromHub
from flextok.utils.demo import imgs_from_urls
import matplotlib.pyplot as plt

# os.environ["TORCHINDUCTOR"] = "0"
# os.environ["TORCH_COMPILE_DISABLE"] = "1"

device = "cuda"

# Load FlexTok model
model = FlexTokFromHub.from_pretrained('EPFL-VILAB/flextok_d18_d28_dfn').eval().to(device)

# Load example images of shape (B, 3, 256, 256), normalized to [-1,1]
imgs = imgs_from_urls(urls=['https://storage.googleapis.com/flextok_site/nb_demo_images/0.png'])
imgs = imgs.to(device)

# tokens_list is a list of [1, 256] discrete token sequences
tokens_list = model.tokenize(imgs)

# tokens_list is a list of [1, l] discrete token sequences, with l <= 256
# reconst is a [B, 3, 256, 256] tensor, normalized to [-1,1]
reconst = model.detokenize(
    tokens_list,
    timesteps=20, # Number of denoising steps
    guidance_scale=7.5, # Classifier-free guidance scale
    perform_norm_guidance=True, # See https://arxiv.org/abs/2410.02416
)

plt.imshow(reconst.permute(0, 2, 3, 1)[0].cpu().numpy() / 2 + 0.5)
plt.show()

