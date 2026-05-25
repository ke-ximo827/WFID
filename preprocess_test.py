import numpy as np
from PIL import Image
import argparse 
import os
from tqdm import tqdm


def process_thermal(thermal_dir, dest_dir, target_size=128):
    """
    Preprocessing consistent with training:
    - Resize to target_size x target_size
    - No blending with sample.png (training doesn't do this)
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    thermal_images = os.listdir(thermal_dir)
    
    print(f"Processing {len(thermal_images)} thermal images...")
    print("Method: Resize only (consistent with training)")
    
    for img_name in tqdm(thermal_images):
        pil_image_th = Image.open(os.path.join(thermal_dir, img_name))
        pil_image_th = pil_image_th.resize((target_size, target_size), Image.BILINEAR)
        pil_image_th.save(os.path.join(dest_dir, img_name))
    
    print(f"✓ Done. Output: {dest_dir}")


def create_argparser():
    defaults = dict(
        thermal_dir='./data/test/TH/',
        dest_dir='./data/test_process/TH/',
        # thermal_dir='./data/130_16/',
        # dest_dir='./data/test_process/130_16/',
        target_size=128,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--thermal_dir", default=defaults['thermal_dir'], type=str)
    parser.add_argument("--dest_dir", default=defaults['dest_dir'], type=str)
    parser.add_argument("--target_size", default=defaults['target_size'], type=int)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    process_thermal(args.thermal_dir, args.dest_dir, args.target_size)