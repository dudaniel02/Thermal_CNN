
import argparse
import os
import torch
from torch.utils.data import DataLoader
from data_prep import ThermalDataset
from model import UNet
import matplotlib.pyplot as plt
import numpy as np

def overlay_contours(img, real_mask, pred_mask, out_path):
    plt.figure(figsize=(6,6))
    # show grayscale thermal image
    plt.imshow(img, cmap='gray')
    # find contours by detecting mask boundaries
    real_edges = np.logical_xor(real_mask, 
                    np.pad(real_mask, ((1,0),(1,0)), mode='constant')[:-1,1:] |
                    np.pad(real_mask, ((1,0),(0,1)), mode='constant')[:-1,:-1] |
                    np.pad(real_mask, ((0,1),(1,0)), mode='constant')[1:,1:] |
                    np.pad(real_mask, ((0,1),(0,1)), mode='constant')[1:,:-1])
    pred_edges = np.logical_xor(pred_mask, 
                    np.pad(pred_mask, ((1,0),(1,0)), mode='constant')[:-1,1:] |
                    np.pad(pred_mask, ((1,0),(0,1)), mode='constant')[:-1,:-1] |
                    np.pad(pred_mask, ((0,1),(1,0)), mode='constant')[1:,1:] |
                    np.pad(pred_mask, ((0,1),(0,1)), mode='constant')[1:,:-1])
    # overlay edges
    plt.contour(real_edges, levels=[0.5], colors='red', linewidths=1.5)
    plt.contour(pred_edges, levels=[0.5], colors='blue', linewidths=1.5)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dirs', nargs='+', required=True,
                   help='folders with .TIFF thermal files')
    p.add_argument('--mask_dir', required=True,
                   help='folder of real binary mask TIFFs')
    p.add_argument('--ckpt', required=True,
                   help='path to model checkpoint .pth')
    p.add_argument('--output_dir', default='overlays_cmp',
                   help='where to save overlay images')
    p.add_argument('--threshold', type=float, default=50.0,
                   help='temp threshold for fallback')
    p.add_argument('--num_samples', type=int, default=10,
                   help='how many samples to process')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    ds = ThermalDataset(args.data_dirs,
                        threshold=args.threshold,
                        mask_dir=args.mask_dir)
    print(f"Dataset size: {len(ds)} samples")
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = UNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt = {k.replace("_orig_mod.", ""):v for k,v in ckpt.items()}
    model.load_state_dict(ckpt)
    model.eval()

    for idx, sample in enumerate(loader):
        if idx >= args.num_samples:
            break
        img       = sample['image'].numpy().squeeze()
        real_mask = sample['mask'].numpy().squeeze()

        with torch.no_grad():
            inp       = sample['image'].unsqueeze(1).to(device)
            pred_logits = model(inp)
            pred_mask = (torch.sigmoid(pred_logits)>0.5).cpu().numpy().squeeze().astype(np.uint8)

        out_path = os.path.join(args.output_dir, f'overlay_cmp_{idx}.png')
        overlay_contours(img, real_mask, pred_mask, out_path)

if __name__ == '__main__':
    main()
