import os
from PIL import Image, UnidentifiedImageError

def convert_and_rename_to_jpg(folder_path):
    """
    Convert all images in the folder to JPG format if they are not already JPG files.
    Rename the files to sequential numbers (1.jpg, 2.jpg, etc.).
    Removes any files that cannot be opened as images.

    Args:
        folder_path (str): Path to the folder containing images.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return

    # Get a sorted list of all files in the directory
    files = sorted(os.listdir(folder_path))
    counter = 1  # Initialize the counter for naming

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        # print(file_path)
        # break

        if not os.path.isfile(file_path):
            print('incorrect fp: ', file_path)
            continue

        try:
            # Open the image
            with Image.open(file_path) as img:
                # Save the image as a new sequential JPG file
                new_file_name = f"{counter}.jpg"
                new_file_path = os.path.join(folder_path, new_file_name)

                # Convert and save as JPG
                img.convert("RGB").save(new_file_path, "JPEG")
                # print(f"Converted and renamed {filename} to {new_file_name}")

                # Remove the original file
                os.remove(file_path)

                # Increment the counter
                counter += 1
                # break
        except UnidentifiedImageError:
            print(f"Cannot open {filename}. Removing the file.")
            os.remove(file_path)
        except Exception as e:
            print(f"An error occurred with file {filename}: {e}")


def clean_unreadable_files(folder_path):
    """
    Checks if files in a folder can be opened as images. 
    Removes any files that cannot be opened.

    Args:
        folder_path (str): Path to the folder containing files.
    """
    print('starting')
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return

    # Iterate through all files in the directory
    files = sorted(os.listdir(folder_path))

    for filename in files:
        file_path = os.path.join(folder_path, filename)

        # Skip if it's not a file
        if not os.path.isfile(file_path):
            print(f"Skipping non-file: {file_path}")
            continue

        # try:
        #     # Try opening the file as an image
        #     with Image.open(file_path) as img:
        #         img.verify()  # Verifies the file is a valid image
        # except UnidentifiedImageError:
        #     print(f"Cannot open {filename}. Removing the file.")
        #     os.remove(file_path)
        # except Exception as e:
        #     print(f"An error occurred with file {filename}: {e}")
        #     os.remove(file_path)

if __name__ == "__main__":
    # folder_path = input("Enter the path to the folder containing images: ").strip()
    # convert_and_rename_to_jpg(folder_path)
    folder_path = input("Enter the path to the folder to clean: ").strip()
    clean_unreadable_files(folder_path)