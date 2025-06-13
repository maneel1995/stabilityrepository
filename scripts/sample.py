#### This script saves images from a parquet file to a local directory.

import pandas as pd
from PIL import Image
import io
import os

df = pd.read_parquet("data/output/")
os.makedirs("saved_images", exist_ok=True)
for idx,row in df.iterrows():
    image_bytes = row['image']['bytes']
    image = Image.open(io.BytesIO(image_bytes))
    image.save(f"saved_images/image_{idx}.png")
    # print(f"Saved image {idx} to data/images/{idx}.png")