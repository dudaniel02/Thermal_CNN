import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import jaccard_score
from data_prep import ThermalDataset
from model import UNet, combo_loss, dice_loss

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dirs', nargs='+', required=True,
                   help='FLAME3 root(s), e.g. data/OutputFolder')
    p.add_argument('--mask_dir', default=None,
                   help='optional corrected masks root')
    p.add_argument('--runs_dir',   default='runs')
    p.add_argument('--ckpt_dir',   default='checkpoints')
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--threshold',  type=float, default=50.0)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.runs_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    writer = SummaryWriter(args.runs_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = ThermalDataset(
        args.data_dirs,
        threshold=args.threshold,
        mask_dir=args.mask_dir
    )
    val_size   = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=16)
    val_loader   = DataLoader(val_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=8)

    model     = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr)

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            imgs  = batch['image'].unsqueeze(1).to(device)
            masks = batch['mask'].unsqueeze(1).float().to(device)
            preds = model(imgs)
            loss  = combo_loss(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        ious, dices = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs  = batch['image'].unsqueeze(1).to(device)
                masks = batch['mask'].unsqueeze(1).float().to(device)
                preds = model(imgs)
                pred_bin = (torch.sigmoid(preds) > 0.5).float()
                flat_p = pred_bin.view(-1).cpu().numpy()
                flat_t = masks.view(-1).cpu().numpy()
                ious.append(jaccard_score(flat_t, flat_p))
                dices.append(1 - dice_loss(preds, masks).item())

        writer.add_scalar('Loss/train',  avg_loss,  epoch)
        writer.add_scalar('IoU/val',     np.mean(ious),  epoch)
        writer.add_scalar('Dice/val',    np.mean(dices), epoch)

        ckpt = os.path.join(args.ckpt_dir, f'epoch{epoch}.pth')
        torch.save(model.state_dict(), ckpt)
        print(f'Epoch {epoch}: loss {avg_loss:.4f}, '
              f'IoU {np.mean(ious):.4f}, Dice {np.mean(dices):.4f}')

if __name__ == '__main__':
    main()
