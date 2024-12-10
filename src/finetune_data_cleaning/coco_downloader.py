import os
import shutil
import fiftyone as fo

# Configuration
DATA_DIR = '/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_random'
SAVE_DIR = f'{DATA_DIR}/real'
CATEGORY_NAME = 'person'
LIMIT = 7500  # Number of images to download


def download_images(dataset, save_dir):
    """Download images from FiftyOne dataset."""
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    for sample in dataset:
        source_path = sample.filepath
        target_path = os.path.join(save_dir, os.path.basename(source_path))

        if not os.path.exists(target_path):
            try:
                shutil.move(source_path, target_path)
                print(f"Downloaded: {os.path.basename(source_path)}")
                count += 1
                if count % 100 == 0:
                    print(f"Downloaded {count} images")

                if LIMIT and count >= LIMIT:
                    break
            except Exception as e:
                print(f"Error moving {source_path} to {target_path}: {e}")
        else:
            print(f"Skipped (already exists): {os.path.basename(source_path)}")


def main():
    # Load COCO dataset filtered by "person"
    print("Downloading COCO dataset using FiftyOne...")
    dataset = fo.zoo.load_zoo_dataset(
        "coco-2017",
        split="train",               # Change to "validation" or "test" if needed
        label_types=["detections"],  # Only object detection annotations
        classes=[CATEGORY_NAME],     # Filter only for "person"
        max_samples=LIMIT,           # Limit the number of samples
        dataset_name="coco_person",
    )

    # Download the images
    download_images(dataset, SAVE_DIR)

    print("Download complete!")


if __name__ == "__main__":
    main()
