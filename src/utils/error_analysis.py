import os
import shutil
from fileinput import filename
from os import listdir
from os.path import isfile, join

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

    real_correct, real_wrong, fake_correct, fake_wrong = get_image_classif(df, num_of_image_)

    copy_images(real_correct, real_correct_path)
    copy_images(real_wrong, real_wrong_path)
    copy_images(fake_correct, fake_correct_path)
    copy_images(fake_wrong, fake_wrong_path)

def get_image_classif(df, num_of_image_):
    real = df[df['True_Label'] == 0].sort_values(by='Prediction')
    fake = df[df['True_Label'] == 1].sort_values(by='Prediction')

    real_correct = real[0:num_of_image_]
    real_wrong = real[-num_of_image_:]
    fake_correct = fake[-num_of_image_:]
    fake_wrong = fake[0:num_of_image_]

    return real_correct, real_wrong, fake_correct, fake_wrong

def check_and_copy(path_to_mj_dataset, path_to_stargan_dataset, folder, filename, copy_path, identifier):
    if isfile(join(path_to_mj_dataset, folder, filename)):
        shutil.copy(join(path_to_mj_dataset, folder, filename), join(copy_path, identifier+"_"+filename))
    elif isfile(join(path_to_stargan_dataset, folder, filename)):
        shutil.copy(join(path_to_stargan_dataset, folder, filename), join(copy_path, identifier+"_"+filename))
    else:
        print('file does not exist', filename, identifier,
              '\n', join(path_to_mj_dataset, folder, filename),
              '\n', join(path_to_stargan_dataset, folder, filename)
              )

def get_images(path_to_mj_dataset, path_to_stargan_dataset):
    pth = 'outputs/cleaned_outputs'
    files = [f for f in listdir(pth) if isfile(join(pth, f))]
    df = pd.concat([pd.read_csv(join(pth, file)) for file in files], ignore_index=True)

    df['image_name'] = df['Image'].apply(lambda x: x.split('/')[-1])
    stargan = df[df['Testing_Dataset'] == 'STARGAN']
    mj = df[df['Testing_Dataset'] == 'MJ']
    real_df = df[df['True_Label'] == 0]
    real_df = real_df.groupby('image_name')['Prediction'].mean()
    real_df = real_df.sort_values()


    real_correct = real_df.index[0]
    real_wrong = real_df.index[-1]

    print('real_correct', real_correct)
    print('real_wrong', real_wrong)

    check_and_copy(path_to_mj_dataset, path_to_stargan_dataset, 'real', real_correct, 'dataset', 'real_correct')
    check_and_copy(path_to_mj_dataset, path_to_stargan_dataset, 'real', real_wrong, 'dataset', 'real_wrong')

    for train_type, df_ in [('stargan', stargan), ('mj', mj)]:
        fake = df_[df_['True_Label'] == 1]
        fake = fake.groupby('image_name')['Prediction'].mean()
        fake = fake.sort_values()
        fake_correct = fake.index[-1]
        fake_wrong = fake.index[0]

        print('===')
        print('fake_correct', fake_correct)
        print('fake_wrong', fake_wrong)

        check_and_copy(path_to_mj_dataset, path_to_stargan_dataset, 'fake', fake_wrong, 'dataset', 'fake_wrong_' + train_type)
        check_and_copy(path_to_mj_dataset, path_to_stargan_dataset, 'fake', fake_correct, 'dataset', 'fake_correct_' + train_type)


if __name__ == "__main__":
    # update this, run from root folder e.g. python src/utils/error_analysis.py

    option = input("What do you want to do? 1. Analyse predictions 2. Get most wrong/correct picture across model:\n     ").strip()

    if option == '1':
        file_path = 'outputs/SPSL_FS_to_StarGan_predictions.csv'
        num_of_image = 5
        read_and_analyse(file_path, num_of_image)
    elif option == '2':
        path_to_mj_dataset = 'dataset/midjourney'
        path_to_stargan_dataset = 'dataset/starganv2'
        get_images(path_to_mj_dataset, path_to_stargan_dataset)


