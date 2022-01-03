import os
import glob
import random
from pathlib import Path
from shutil import copyfile


CHARS = "0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호바사아자배abcdefghijklmnopq"  # exclude IO


def divide_dataset(root_path, train_target_dir, val_target_dir):
    total = glob.glob(os.path.join(root_path, '*.jpg'))
    random.shuffle(total)
    print("Total image number: {}".format(len(total)))
    train_set = random.sample(total, int(len(total) * 0.8))
    val_set = list(set(total) - set(train_set))

    for idx, i in enumerate(train_set):
        base_name = os.path.basename(i)
        print(i)
        # if idx > 100: break
        copyfile(i, os.path.join(train_target_dir, base_name))

    for idx, i in enumerate(val_set):
        base_name = os.path.basename(i)
        print(i)
        # if idx > 100: break
        copyfile(i, os.path.join(val_target_dir, base_name))


def clean_data(path):
    for i in glob.glob(os.path.join(path, "*.jpg")):
        base_name = os.path.basename(i)
        f_name = base_name.split('_')[0]
        print(base_name)
        os.rename(i, i.replace('이', '아'))
        # for f in f_name:
            # if f not in CHARS:
            #     os.rename(i, os.path.join(r'C:\dataset\license_plate\license_plate_recognition\tmp', base_name))


if __name__ == '__main__':
    original_path = r"C:\dataset\license_plate\license_plate_from_police"
    train_target_dir = r"C:\dataset\license_plate\mini_LPR_dataset\train"
    val_target_dir = r"C:\dataset\license_plate\mini_LPR_dataset\val"

    Path(train_target_dir).mkdir(exist_ok=True, parents=True)
    Path(val_target_dir).mkdir(exist_ok=True, parents=True)

    divide_dataset(original_path, train_target_dir, val_target_dir)
    # clean_data(r'C:\dataset\license_plate\license_plate_recognition\tmp')
    pass