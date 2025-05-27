import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import jaccard_score
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from data_prep import ThermalDataset
from model import UNet, combo_loss, dice_loss

def parse_args():
    p = argparse.ArgumentParser("Train a U-Net for fire segmentation")
    p.add_argument('--data_dirs', nargs='+', required=True,
                   help='folders containing your .TIFF thermal files')
    p.add_argument('--mask_dir', default=None,
                   help='optional folder of corrected mask TIFFs')
    p.add_argument('--runs_dir', default='runs',
                   help='TensorBoard log directory')
    p.add_argument('--ckpt_dir', default='checkpoints',
                   help='where to save model checkpoints')
    p.add_argument('--epochs', type=int, default=50,
                   help='number of training epochs')
    p.add_argument('--batch_size', type=int, default=8,
                   help='batch size')
    p.add_argument('--lr', type=float, default=1e-3,
                   help='learning rate')
    p.add_argument('--threshold', type=float, default=50.0,
                   help='thermal threshold for mask fallback')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.runs_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # DEBUG: show what data_dirs were passed and contents
    print("DATA_DIRS:", args.data_dirs)
    for d in args.data_dirs:
        import glob
        files = glob.glob(os.path.join(d, "*.TIFF"))
        print(f"  {d} has {len(files)} TIFFs")

    dataset = ThermalDataset(
        input_dirs=args.data_dirs,
        threshold=args.threshold,
        mask_dir=args.mask_dir
    )
    total = len(dataset)
    print(f"ThermalDataset found {total} samples")
    if total == 0:
        raise RuntimeError("No samples found. Check your --data_dirs paths.")

    val_size = int(0.2 * total)
    train_size = total - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    loader_args = {
        'batch_size': args.batch_size,
        'num_workers': 32,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 2
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = UNet().to(device)
    model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    writer = SummaryWriter(args.runs_dir)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{args.epochs} [train]",
                    unit="batch")
        for batch in pbar:
            imgs  = batch['image'].unsqueeze(1).to(device, non_blocking=True)
            masks = batch['mask'].unsqueeze(1).float().to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type=device):
                preds = model(imgs)
                loss  = combo_loss(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss/(pbar.n+1):.4f}")

        avg_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Validate
        model.eval()
        print(f"\nStarting validation for epoch {epoch}...")
        ious, dices = [], []
        vbar = tqdm(val_loader,
                    desc=f"Epoch {epoch}/{args.epochs} [val]  ",
                    unit="batch")
        with torch.no_grad():
            for batch in vbar:
                imgs  = batch['image'].unsqueeze(1).to(device)
                masks = batch['mask'].unsqueeze(1).float().to(device)
                preds = model(imgs)
                pred_bin = (torch.sigmoid(preds) > 0.5).float()

                flat_p = pred_bin.view(-1).cpu().numpy()
                flat_t = masks.view(-1).cpu().numpy()
                ious.append(jaccard_score(flat_t, flat_p))
                dices.append(1 - dice_loss(preds, masks).item())

                vbar.set_postfix(iou=f"{np.mean(ious):.4f}",
                                 dice=f"{np.mean(dices):.4f}")

        mean_iou  = np.mean(ious)
        mean_dice = np.mean(dices)
        writer.add_scalar('IoU/val', mean_iou, epoch)
        writer.add_scalar('Dice/val', mean_dice, epoch)

        # Checkpoint
        ckpt = os.path.join(args.ckpt_dir, f'epoch{epoch}.pth')
        torch.save(model.state_dict(), ckpt)
        print(f"Epoch {epoch:02d} done — loss {avg_loss:.4f}, IoU {mean_iou:.4f}, Dice {mean_dice:.4f}\n")

    writer.close()

if __name__ == '__main__':
    main()
