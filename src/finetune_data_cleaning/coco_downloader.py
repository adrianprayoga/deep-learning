import os
import shutil
import fiftyone as fo

DATA_DIR = '/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_random'
SAVE_DIR = f'{DATA_DIR}/real'
CATEGORY_NAME = 'person'
LIMIT = 7500  


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

    dataset = fo.zoo.load_zoo_dataset(
        "coco-2017",
        split="train",               
        label_types=["detections"],  
        classes=[CATEGORY_NAME],     
        max_samples=LIMIT,           
        dataset_name="coco_person",
    )

    download_images(dataset, SAVE_DIR)

    print("Donew!")


if __name__ == "__main__":
    main()
