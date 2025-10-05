import webdataset as wds
import tarfile
from pathlib import Path
import glob
from tqdm import tqdm
import os
import numpy as np
import random



random.seed(42)

TAR_DIR = Path("/p/scratch/mfmpm/tim/data/TCGA/cache")
patient_tars = glob.glob(os.path.join(TAR_DIR, "*.tar"))
OUTPUT_DIR = Path("/p/scratch/mfmpm/montillet/data/shuffled_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)
ID_MAX = 15000

tar_id = len(glob.glob(os.path.join(OUTPUT_DIR, "shard-*.tar")))
print(f"tar_id {tar_id}")
# max number of images
max_imgs = 0

# Step 1: Create iterators per patient tar
def iter_images_from_tar(tar_path):
    global max_imgs
    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(('.png', '.jpg', '.jpeg'))]
        if len(members) > max_imgs:
            max_imgs = len(members)
        random.shuffle(members)
        if len(members) > tar_id:
            members = members[tar_id:]
            for member in members:
                f = tar.extractfile(member)
                yield f.read()

print("Creating the iterators")
print(f"Maximal number of images : {max_imgs}")

patient_iters = [iter_images_from_tar(p) for p in tqdm(patient_tars)]
num_patients = len(patient_iters)
assert num_patients > 0
print(f"Number of patients {num_patients}")
exhausted = np.zeros(num_patients, dtype=bool)

# Step 2: Round-robin streaming
# This will create one output tar per round

while tar_id < ID_MAX:
    sample_id = 0
    batch_images = []
    for pid, it in enumerate(patient_iters):
        if not exhausted[pid]:
            try:
                image_bytes = next(it)
                batch_images.append(image_bytes)
            except StopIteration:
                exhausted[pid] = True
    
    if not batch_images:
        print("All patient images used.")
        break
    
    out_path = OUTPUT_DIR / f"shard-{tar_id:05d}.tar"
    with wds.TarWriter(str(out_path)) as sink:
        for image_bytes in batch_images:
            key = f"{sample_id:05d}"  
            sink.write({"__key__": key, "jpg": image_bytes})
            sample_id += 1
    
    print(f"Wrote shard {tar_id}/{max_imgs}, {sum(exhausted)}/{num_patients} archives exhausted.")
    tar_id += 1
