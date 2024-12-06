import os
import shutil

def flatten_directory(base_path, subfolder=None):
    """
    Flattens a directory structure by moving all files to the base path
    and renaming them to ensure unique names.
    Usefull for heygen, COllabDif, SiT datasets
    """
    target_path = os.path.join(base_path, subfolder) if subfolder else base_path
    i = 0
    dirs_to_remove = []
    
    for root, dirs, files in os.walk(target_path):
        for file in files:
            i+=1
            file_path = os.path.join(root, file)
            file_name, file_ext = os.path.splitext(file)
            unique_name = f"{file_name}_{i}{file_ext}"
            dest_path = os.path.join(base_path, unique_name)
            shutil.move(file_path, dest_path)
        for dir_name in dirs:
            dirs_to_remove.append(os.path.join(root, dir_name))
    
    # Check if it works first!!!!
    for dir_path in dirs_to_remove:
        if os.path.isdir(dir_path):
            os.rmdir(dir_path)


if __name__ == "__main__":
    fake_base_fp = 'data/CollabDiff/fake/'
    real_base_fp = 'data/CollabDiff/real/'

    flatten_directory(fake_base_fp, None)
    print("Directories flattened successfully.")
