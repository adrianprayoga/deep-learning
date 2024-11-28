import os
import shutil
from fileinput import filename

import pandas as pd

def copy_images(df, store_path):
    for index, row in df.iterrows():
        file_name = row['Image'].split('/')[-1]

        if not os.path.exists(row['Image']):
            print(f"Source file does not exist: {row['Image']}")

        print('copying files from', row['Image'], 'to', os.path.join(store_path, file_name))
        shutil.copy(row['Image'], os.path.join(store_path, file_name))

def read_and_analyse(file_path_, num_of_image_):
    df = pd.read_csv(file_path_)
    model = df['Model'][0]
    training_ds = df['Training_Dataset'][0]
    testing_ds = df['Testing_Dataset'][0]

    # Folder path
    real_correct_path = os.path.join('outputs', model, training_ds, testing_ds, 'real', 'correct')
    os.makedirs(real_correct_path, exist_ok=True)
    real_wrong_path = os.path.join('outputs', model, training_ds, testing_ds, 'real', 'wrong')
    os.makedirs(real_wrong_path, exist_ok=True)
    fake_correct_path = os.path.join('outputs', model, training_ds, testing_ds, 'fake', 'correct')
    os.makedirs(fake_correct_path, exist_ok=True)
    fake_wrong_path = os.path.join('outputs', model, training_ds, testing_ds, 'fake', 'wrong')
    os.makedirs(fake_wrong_path, exist_ok=True)

    real = df[df['True_Label'] == 0].sort_values(by='Prediction')
    fake = df[df['True_Label'] == 1].sort_values(by='Prediction')

    real_correct = real[0:num_of_image_]
    real_wrong = real[-num_of_image_:]
    fake_correct = fake[-num_of_image_:]
    fake_wrong = fake[0:num_of_image_]


    copy_images(real_correct, real_correct_path)
    copy_images(real_wrong, real_wrong_path)
    copy_images(fake_correct, fake_correct_path)
    copy_images(fake_wrong, fake_wrong_path)


if __name__ == "__main__":
    # update this, run from root folder e.g. python src/utils/error_analysis.py
    file_path = 'outputs/SPSL_FS_to_StarGan_predictions.csv'
    num_of_image = 5
    read_and_analyse(file_path, num_of_image)

