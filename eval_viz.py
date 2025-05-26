import argparse
import os
import torch
from torch.utils.data import DataLoader
from data_prep import ThermalDataset
from model import UNet
import matplotlib.pyplot as plt

def overlay_and_save(img, mask, out_path):
    plt.figure()
    plt.imshow(img, cmap='hot')
    plt.imshow(mask, alpha=0.4, cmap='Reds')
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dirs', nargs='+', required=True,
                   help='FLAME3 root(s)')
    p.add_argument('--ckpt',       required=True)
    p.add_argument('--output_dir', default='overlays')
    p.add_argument('--threshold',  type=float, default=50.0)
    p.add_argument('--num_samples',type=int, default=10)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = ThermalDataset(args.data_dirs,
                             threshold=args.threshold)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet().to(device)
    model.load_state_dict(torch.load(args.ckpt,
                                     map_location=device))
    model.eval()

    for idx, sample in enumerate(loader):
        if idx >= args.num_samples:
            break
        img = sample['image'].numpy().squeeze()
        with torch.no_grad():
            inp  = sample['image'].unsqueeze(1).to(device)
            pred = model(inp)
            mask = (torch.sigmoid(pred) > 0.5).cpu().numpy().squeeze()

        out_path = os.path.join(args.output_dir,
                                f'overlay_{idx}.png')
        overlay_and_save(img, mask, out_path)

if __name__ == '__main__':
    main()
