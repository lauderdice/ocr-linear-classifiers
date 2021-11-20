from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL.PngImagePlugin import PngImageFile
from typing import Tuple, List
from os import listdir
from os.path import isfile, join


# load single example
def load_example(img_path: str) -> Tuple[np.array, str, PngImageFile]:
    Y = img_path[img_path.rfind('_') + 1:-4]

    img = Image.open(img_path)
    img_mat = np.asarray(img)

    n_letters = len(Y)
    im_height = int(img_mat.shape[0])
    im_width = int(img_mat.shape[1] / n_letters)
    n_pixels = im_height * im_width

    X = np.zeros([int(n_pixels + n_pixels * (n_pixels - 1) / 2), n_letters])
    for i in range(n_letters):

        # single letter
        letter = img_mat[:, i * im_width:(i + 1) * im_width] / 255

        # compute features
        x = letter.flatten()
        X[0:len(x), i] = x
        cnt = n_pixels
        for j in range(0, n_pixels - 1):
            for k in range(j + 1, n_pixels):
                X[cnt, i] = x[j] * x[k]
                cnt = cnt + 1

        X[:, i] = X[:, i] / np.linalg.norm(X[:, i])

    return X, Y, img


# load all examples from a folder
def load_examples(image_folder: str) -> Tuple[List[np.ndarray], List[str], List[PngImageFile]]:
    files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]

    X = []
    Y = []
    img = []
    for i, file in enumerate(listdir(image_folder)):
        path = join(image_folder, file)
        if isfile(path):
            X_, Y_, img_ = load_example(path)
            X.append(X_)
            Y.append(Y_)
            img.append(img_)

    return X, Y, img


