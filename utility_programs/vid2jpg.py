# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import cv2
import sys
import subprocess
from tqdm import tqdm
from joblib import Parallel, delayed

commands = []

def to_images(video_file_path, dst_directory_path):
    vidcap = cv2.VideoCapture(video_file_path)
    success, image = vidcap.read()
    if not success:
        print(f'Error: {video_file_path}')
    count = 1
    while success:
        cv2.imwrite(os.path.join(dst_directory_path, f'image_{count:05}.jpg'), image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        success, image = vidcap.read()
        count += 1


def class_process(dir_path, dst_dir_path, class_name):
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_class_path):
        os.mkdir(dst_class_path)
    for file_name in os.listdir(class_path):
        if '.avi' not in file_name and '.mp4' not in file_name:
            continue
        name, ext = os.path.splitext(file_name)
        dst_directory_path = os.path.join(dst_class_path, name)

        video_file_path = os.path.join(class_path, file_name)
        try:
            if os.path.exists(dst_directory_path):
                if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
                    subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                    print('remove {}'.format(dst_directory_path))
                    os.mkdir(dst_directory_path)
                else:
                    continue
            else:
                os.mkdir(dst_directory_path)
        except:
            print(dst_directory_path)
            continue
        commands.append((video_file_path, dst_directory_path))


if __name__ == "__main__":
    dir_path = sys.argv[1]
    dst_dir_path = sys.argv[2]

    if not os.path.exists(dst_dir_path):
        os.mkdir(dst_dir_path)

    for class_name in sorted(os.listdir(dir_path)):
        class_process(dir_path, dst_dir_path, class_name)

    Parallel(n_jobs=14)(delayed(lambda x: to_images(x[0], x[1]))(cmd) for cmd in tqdm(commands))
