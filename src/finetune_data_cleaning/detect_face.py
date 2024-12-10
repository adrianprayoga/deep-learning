# Yolo Implementation doesn't work too well
# from ultralytics import YOLO
# import cv2
# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt
# import os

# def function_testing(fp):
#     print(f'Processing file: {fp}')
#     image = cv2.imread(fp)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     model = YOLO('yolov8n.pt')
#     results = model(image_rgb)
#     if len(results[0].boxes.xyxy) > 0:
#         boxes = results[0].boxes.xyxy.cpu().numpy()
#         confidences = results[0].boxes.conf.cpu().numpy()
#         best_idx = confidences.argmax()
#         x_min, y_min, x_max, y_max = map(int, boxes[best_idx])
#         cropped_face = image_rgb[y_min:y_max, x_min:x_max]
#         plt.figure(figsize=(6, 6))
#         plt.imshow(cropped_face)
#         plt.axis("off")
#         plt.title(f"Best Detected Face: {os.path.basename(fp)}")
#         plt.show()
#     else:
#         print(f"No faces detected in {fp}")

# if __name__ == "__main__":
#     directory = '/home/ginger/code/gderiddershanghai/deep-learning/data/face_finder_test'
#     for fp in os.listdir(directory):
#         full_path = os.path.join(directory, fp)
#         if os.path.isfile(full_path):
#             function_testing(full_path)

import pandas as pd
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import random


def function_testing(fp):
    print(f'processing this file: {fp}')
    mp_face_detection = mp.solutions.face_detection

    image = cv2.imread(fp)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

        if results.detections:
            best_face = max(results.detections, key=lambda d: d.score[0])
            bboxC = best_face.location_data.relative_bounding_box
            h, w, _ = image.shape
            x_min = int(bboxC.xmin * w)
            y_min = int(bboxC.ymin * h)
            x_max = int((bboxC.xmin + bboxC.width) * w)
            y_max = int((bboxC.ymin + bboxC.height) * h)

            
            padding = random.uniform(0.07, 0.13)
            x_min = max(0, int(x_min - padding * w))
            y_min = max(0, int(y_min - padding * h))
            x_max = min(w, int(x_max + padding * w))
            y_max = min(h, int(y_max + padding * h))

            cropped_face = image_rgb[y_min:y_max, x_min:x_max]
            plt.figure(figsize=(6, 6))
            plt.imshow(cropped_face)
            plt.axis("off")
            plt.title(f" Detected Face: {os.path.basename(fp)}")
            plt.show()
        else:
            print(f"No faces in {fp}")

def crop_and_save_faces(input_dir, output_dir, noface_dir, csv_path, img_col, crop=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(noface_dir):
        os.makedirs(noface_dir)

    if csv_path:
        df = pd.read_csv(csv_path)
        valid_images = set(df[img_col].values)
    processed_count = 0
    skipped_count = 0

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.175) as face_detection:
        for i, img_name in enumerate(os.listdir(input_dir)):
            # if i==1: break
            full_path = os.path.join(input_dir, img_name)
            img_check = img_name in valid_images if csv_path else True
            
            if os.path.isfile(full_path) and img_check:
                image = cv2.imread(full_path)
                if image is None:
                    print(f"Failed to load image: {full_path}. Skipping...")
                    skipped_count += 1
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image_rgb)

                if results.detections:
                    if crop:
                        best_face = max(results.detections, key=lambda d: d.score[0])
                        bboxC = best_face.location_data.relative_bounding_box
                        h, w, _ = image.shape
                        x_min = int(bboxC.xmin * w)
                        y_min = int(bboxC.ymin * h)
                        x_max = int((bboxC.xmin + bboxC.width) * w)
                        y_max = int((bboxC.ymin + bboxC.height) * h)

                        padding = random.uniform(0.07, 0.13)
                        x_min = max(0, int(x_min - padding * w))
                        y_min = max(0, int(y_min - padding * h))
                        x_max = min(w, int(x_max + padding * w))
                        y_max = min(h, int(y_max + padding * h))

                        cropped_face = image[y_min:y_max, x_min:x_max]

                        save_path = os.path.join(output_dir, img_name)
                        try:
                            if not cv2.imwrite(save_path, cropped_face):
                                print(f"Failed to save cropped face for {img_name}.")
                            processed_count += 1
                        except Exception as e:
                            print(f"Error saving {save_path}: {e}")
                    else:
                        skipped_count += 1
                        if skipped_count % 100 == 0:
                            print(f"Skipped {skipped_count} images so far (no face detected).")
                        no_face_path = os.path.join(noface_dir, img_name)
                        cv2.imwrite(no_face_path, image_rgb)

                    try:
                        os.remove(full_path)
                    except Exception as e:
                        print(f"Error deleting {full_path}: {e}")

                if i % 500 == 0 and i > 0:
                    print(f"Processed {i} images so far.")

    print(f"Processed {processed_count} images. Skipped {skipped_count} images with no face.")


if __name__ == "__main__":
    # directory = '/home/ginger/code/gderiddershanghai/deep-learning/data/face_finder_test'
    # for fp in os.listdir(directory):
    #     full_path = os.path.join(directory, fp)
    #     if os.path.isfile(full_path):
    #         function_testing(full_path)
    input_directory = "/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_random/real"
    # input_directory = os.path.expanduser("~/fiftyone/coco-2017/train/data")


    output_directory = "/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_random/face"
    no_face_directory = "/home/ginger/code/gderiddershanghai/deep-learning/data/JDB_random/no_face"
    csv_file = None #'/home/ginger/code/gderiddershanghai/deep-learning/data/jdb_info.csv'
    image_column = 'img_path'
    crop = False
    crop_and_save_faces(input_directory, output_directory, no_face_directory, csv_file, image_column, crop)

