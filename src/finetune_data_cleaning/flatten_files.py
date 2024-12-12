import os
import shutil

def flatten_directory(base_path, target_folder, subfolder=None):
    source_path = os.path.join(base_path, subfolder) if subfolder else base_path
    target_path = os.path.join(base_path, target_folder)

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    file_counter = 0
    dirs_to_remove = []

   ####################GET FILES################################
    for root, dir, files in os.walk(source_path):
        for file in files:
            file_counter += 1
            file_path = os.path.join(root, file)
            file_name, file_ext = os.path.splitext(file)
            
            unique_name = f"{file_name}_{file_counter:06d}{file_ext}"
            dest_path = os.path.join(target_path, unique_name)

            try:
                shutil.move(file_path, dest_path)
                print(f"MOved {file_path} to {dest_path}")
            except Exception as e:
                print(f"coudln't move {file_path}: {e}")

        if root != source_path:
            dirs_to_remove.append(root)

    ######################REMOVE DIRS#######################
    for dir_path in dirs_to_remove:
        try:
            os.rmdir(dir_path)
            print(f"Removed empty dir {dir_path}")
        except OSError as e:
            print(f"couldnt remove dir {dir_path}: {e}")

    print(f" TOTAL FILES {file_counter}")

if __name__ == "__main__":
    # def flatten_directory(base_path, target_folder, subfolder=None):
    base_fp = '/home/ginger/code/gderiddershanghai/deep-learning/data/hyperreenact/'
    flatten_directory(base_fp, "real", "cropped_images")
