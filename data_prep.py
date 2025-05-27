import argparse
import glob
import os
import rasterio
import numpy as np
from multiprocessing import Pool
from functools import partial
from torch.utils.data import Dataset
from PIL import Image

def process_tiff(path, base_dir, output_dir, threshold):
    # read thermal TIFF
    with rasterio.open(path) as src:
        img = src.read(1)
    # create binary mask 0/255
    mask = (img >= threshold).astype('uint8') * 255
    # preserve folder structure
    rel = os.path.relpath(path, base_dir)
    out_path = os.path.join(output_dir, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # copy georef and write mask
    with rasterio.open(path) as src:
        meta = src.meta.copy()
    meta.update(dtype='uint8', count=1)
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(mask, 1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dirs', nargs='+', required=True,
                   help='e.g. data/OutputFolder')
    p.add_argument('--output_dir', required=True,
                   help='where to write your mask TIFFs')
    p.add_argument('--threshold', type=float, default=50.0)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    
    # collect all thermal TIFFs - FIXED PATTERN
    tasks = []
    for base in args.input_dirs:
        pattern = os.path.join(base, '*.TIFF')
        for path in glob.glob(pattern):
            tasks.append((path, base))
    
    print(f"Found {len(tasks)} TIFF files to process")
    
    os.makedirs(args.output_dir, exist_ok=True)
    fn = partial(process_tiff,
                 output_dir=args.output_dir,
                 threshold=args.threshold)
    
    with Pool(args.workers) as pool:
        pool.starmap(fn, tasks)

class ThermalDataset(Dataset):
    """
    Loads each .TIFF directly under the given input_dirs,
    generates (or loads) a mask, and resizes both to 512×640.
    """
    def __init__(self, input_dirs, threshold=50.0, mask_dir=None, transform=None):
        self.threshold = threshold
        self.transform = transform
        self.mask_dir = mask_dir
        self.samples = []
        for base in input_dirs:
            pattern = os.path.join(base, '*.TIFF')
            for tiff in glob.glob(pattern):
                self.samples.append((tiff, base))

        # target size
        self.target_w = 640
        self.target_h = 512

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, base = self.samples[idx]
        # load thermal
        with rasterio.open(path) as src:
            img = src.read(1).astype(np.float32)

        # load or generate mask
        if self.mask_dir:
            rel = os.path.relpath(path, base)
            mask_path = os.path.join(self.mask_dir, rel)
            if os.path.exists(mask_path):
                with rasterio.open(mask_path) as m:
                    mask = m.read(1).astype(np.uint8)
            else:
                mask = (img >= self.threshold).astype(np.uint8)
        else:
            mask = (img >= self.threshold).astype(np.uint8)

        # resize both via PIL
        pil_img  = Image.fromarray(img)
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
        img_r = np.array(pil_img.resize((self.target_w, self.target_h), Image.BILINEAR), dtype=np.float32)
        mask_r = np.array(pil_mask.resize((self.target_w, self.target_h), Image.NEAREST), dtype=np.uint8)
        mask_r = (mask_r > 0).astype(np.uint8)

        sample = {'image': img_r, 'mask': mask_r}
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == '__main__':
    main()
