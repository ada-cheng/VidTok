import torch
from scripts.inference_evaluate import load_model_from_config
from lightning.pytorch import seed_everything
from PIL import Image
import torchvision.transforms as T


# Load the image
img = Image.open("assets/gemini.png").convert("RGB")  # Ensure 3 channels

# Define transforms: resize, convert to tensor, normalize to [-1, 1]
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),                      # Converts to shape [3, 256, 256], values [0, 1]
    T.Normalize(mean=[0.5, 0.5, 0.5],  # Now values in [-1, 1]
                std=[0.5, 0.5, 0.5])
])

# Apply transform
tensor = transform(img)  # shape: [3, 256, 256]

# Add dimensions to get shape (1, 3, 1, 256, 256)
tensor = tensor.unsqueeze(0).unsqueeze(2)

cfg_path = "configs/vidtok_v1_1/vidtok_fsq_causal_488_32768_v1_1.yaml"
ckpt_path = "checkpoints/vidtok_kl_causal_488_16chn_v1_1.ckpt"

# load pre-trained model
model = load_model_from_config(cfg_path, ckpt_path)
model.to('cuda').eval()

num_frames = 1
x_input = tensor.to('cuda')  # [B,C,T,H,W], range -1~1

with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
    _, x_recon3, _ = model(x_input)

model.enable_tiling()
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
    _, x_recon, _ = model(x_input)

model.disable_tiling()
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
    _, x_recon2, _ = model(x_input)
    
print(f'x recon2 - x recon: {(x_recon - x_recon2).abs().max().item()}')  # should be very small
print(f'x recon - x recon3: {(x_recon - x_recon3).abs().max().item()}')  # should be very small


output_img1 = (x_recon.squeeze(0).squeeze(1)).permute(1, 2, 0)
output_img1 = ((output_img1 + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()  # Convert to [0,
# 255] range and to numpy array
output_img1 = Image.fromarray(output_img1)  # Convert to PIL Image
output_img1.save("assets/gemini_output1.png")
