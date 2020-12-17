from torchvision.transforms import ToTensor
from PIL import Image
from pathlib import Path
import numpy as np
import torch
import os


def path_to_video(path, clip_len):
    video = []
    n_frames = count_frames(path)
    if clip_len < n_frames:
        time_index = np.random.randint(n_frames - clip_len)
        it = range(time_index + 1, time_index + clip_len + 1)
    else:
        it = np.linspace(1, n_frames, num=clip_len, dtype=np.int32)
    for i in it:
        image_path = os.path.join(path, f'image_{i:05}.jpg')
        if os.path.exists(image_path):
            img = load_image(image_path)
            img = ToTensor()(np.array(img))
            video.append(img)
        else:
            break
    video = torch.stack(video, dim=1)
    return video


def load_image(image_path):
    with open(image_path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def count_frames(path):
    count = 0
    for f in os.listdir(path):
        if f.endswith('.jpg'):
            count += 1
    return count


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def print_values(mode, loss, accuracy, top_5_accuracy):
    print(f'm: {mode} l: {loss} a: {accuracy} a5: {top_5_accuracy}')


def accuracy(output, target, k):
    argsorted_y = np.argsort(output)[:, -k:]
    return np.asarray(np.any(argsorted_y.T == target, axis=0).sum(dtype='f'))


def checkpoint(model, epoch, directory, name):
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), os.path.join(directory, f'{name}_{epoch:04d}.pt'))
