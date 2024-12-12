import os
import random
from PIL import Image
import cv2

def calculate_average_image_size(image_directory, sample_size=100):

    if not os.path.exists(image_directory):
        print(f"Directory {image_directory} does not exist.")
        return None, None


    image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

    if not image_files:
        print(f"No image files found in {image_directory}.")
        return None, None


    sample_images = random.sample(image_files, min(sample_size, len(image_files)))

    total_width = 0
    total_height = 0
    min_width, min_height = float('inf'), float('inf')
    max_width, max_height = 0, 0

    for img_file in sample_images:
        img_path = os.path.join(image_directory, img_file)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                total_width += width
                total_height += height
                width, height = img.size
                min_width, min_height = min(min_width, width), min(min_height, height)
                max_width, max_height = max(max_width, width), max(max_height, height)
        except Exception as e:
            print(f"Error processing file {img_file}: {e}")
    print(f"Min Size: {min_width}x{min_height}, Max Size: {max_width}x{max_height}")

    average_width = total_width / len(sample_images)
    average_height = total_height / len(sample_images)

    return average_width, average_height

def random_downscale(image):
    scale_factor = random.uniform(0.23, 0.6)
    new_width = max(int(image.shape[1] * scale_factor), 224)
    new_height = max(int(image.shape[0] * scale_factor), 224)
    downscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return downscaled_image



def random_upscale(image):
    scale_factor = random.uniform(1.5, 2.2)
    new_width = max(224,int(image.shape[1] * scale_factor))
    new_height = max(224,int(image.shape[0] * scale_factor))
    upscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return upscaled_image


if __name__ == "__main__":
    # image_directory = '/home/ginger/code/gderiddershanghai/deep-learning/data/Coco/rescaled_images'
    # image_directory = '/home/ginger/code/gderiddershanghai/deep-learning/data/JDB/fake_rescaled'
    # avg_width, avg_height = calculate_average_image_size(image_directory, sample_size=100)
    # if avg_width and avg_height:
    #     print(f"Average Image Size: {avg_width:.2f} x {avg_height:.2f} pixels")
    # else:
    #     print("Failed to calculate average image size.")

    # input_dir = "/home/ginger/code/gderiddershanghai/deep-learning/data/JDB/fake"
    # output_dir = "/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_train/fake"
    input_dir = "/home/ginger/code/gderiddershanghai/deep-learning/data/Coco/real"
    output_dir = "/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_train/real"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)
        if os.path.isfile(img_path):
            image = cv2.imread(img_path)
            if image is not None:
                rescaled_image = random_upscale(image)
                save_path = os.path.join(output_dir, img_file)
                cv2.imwrite(save_path, rescaled_image)
