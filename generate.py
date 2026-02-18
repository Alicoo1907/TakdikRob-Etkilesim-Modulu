import torch
import numpy as np
import os
from new_net_G import ActFormer_Generator
from gp_sampling import sample_gp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_dim = 64
num_classes = 15
num_joints = 7
T = 60

current_path = os.getcwd()
model_path = os.path.join(current_path, "netG_epoch1915.pt")

netG = ActFormer_Generator(
    Z=noise_dim,
    T=T,
    C=3,
    V=num_joints,
    spectral_norm=True,
    learnable_pos_embed=True,
    out_normalize=None,
    num_class=num_classes,
    embed_dim_ratio=64,
    depth=12,
    num_heads=14
).to(device)

netG.load_state_dict(torch.load(model_path, map_location=device))
netG.eval()


def generate_motion_by_label(label: int):
    print("hareket üretiliyor")
    """Label verildiğinde hareket üretir ve numpy array döner"""
    z = sample_gp(1, T, noise_dim, device)
    label_tensor = torch.tensor([label], dtype=torch.long, device=device)

    with torch.no_grad():
        fake_seq = netG(z, label_tensor)

    fake_np = fake_seq.squeeze().cpu().numpy()   # (3,7,T)
    fake_np = fake_np.transpose(1,0,2)           # (7,3,T)
    return fake_np
