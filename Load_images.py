import os
import numpy as np
# import keras
import cv2
from random import shuffle

def rgb2gray(rgb):
    import numpy as np
    if len(rgb.shape) > 3:
        r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    else:
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = original.max()
    psnr = 10 * np.log10(max_pixel**2 / mse)
    return psnr


def generate_Data(directory, batch_size, image_size, train=True):
    masks = [rgb2gray(cv2.resize(cv2.imread('Dataset/adi/mask.png'), image_size)),
             rgb2gray(cv2.resize(cv2.imread('Dataset/Motorcycle/mask_50.png'), image_size))]
    print(masks[0].shape, masks[1].shape)
    i = 0
    file_list = os.listdir(directory)
    shuffle(file_list)
    while True:
        imageBatch = []
        labelBatch = []
        for j in range(batch_size):
            if (i < len(file_list)):
                sample = file_list[i]
                i += 1
                image = cv2.imread(directory + sample)
                image = rgb2gray(cv2.resize(image, image_size))
                imageM = np.where(masks[0].round() != 255, 0, image) if i % 2 == 0 else np.where(
                    masks[1].round() <= 120, 0, image)
                image, imageM = image.astype(float) / 255, imageM.astype(float) / 255
                imageBatch.append(imageM)
                labelBatch.append(image)
            else:
                i = 0
        labelBatch = np.expand_dims(np.array(labelBatch), axis=-1)
        imageBatch = np.expand_dims(np.array(imageBatch), axis=-1)
        if train:
            yield imageBatch.astype('float32'), labelBatch.astype('float32')
        else:
            yield imageBatch
