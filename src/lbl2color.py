import numpy as np
from PIL import Image
from pathlib import Path

lbl_dir = Path('../output/cityscapes_preds')
color_dir = Path('../output/cityscapes_color')
color_dir.mkdir()

lbl_paths = sorted(lbl_dir.iterdir())
for lbl_path in lbl_paths:
    np.array(Image.open(lbl_path))