import os
import shutil
import random

SOURCE_DIR = "dataset"  # Name of your unzipped dataset folder
TARGET_DIR = "Split_smol"
TRAIN_RATIO = 0.8

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    
    # ❗ Ignore Split_smol if already exists inside dataset
    if class_name.lower() == "split_smol":
        continue
    
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    train_count = int(len(images) * TRAIN_RATIO)
    train_images = images[:train_count]
    val_images = images[train_count:]

    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(TARGET_DIR, "train", class_name, img)
        create_dir(os.path.dirname(dst))
        try:
            shutil.copy(src, dst)
        except Exception as e:
            print(f"⚠️ Failed to copy {src}: {e}")

    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(TARGET_DIR, "validation", class_name, img)
        create_dir(os.path.dirname(dst))
        try:
            shutil.copy(src, dst)
        except Exception as e:
            print(f"⚠️ Failed to copy {src}: {e}")

print("✅ Dataset split into 'train' and 'validation'.")
