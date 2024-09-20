# -*- coding: UTF-8 -*-
import numpy as np


def threshold(image, thresh: int) -> np.array:
    image = np.asarray(image)
    image = image.copy()
    image[image <= thresh] = 0
    image[image > thresh] = 1
    return image


def get_histogram(image):
    histogram = np.zeros(256)
    # slow fashion:
    # for h in range(image.shape[0]):
    #     for w in range(image.shape[1]):
    #         histogram[image[h, w]] += 1
    # fast fashion:
    for i in range(256):
        histogram[i] = np.sum(image == i)
    return histogram


def otsu(histogram: np.array) -> int:
    N = np.sum(histogram)
    # frequency
    prob = np.zeros(256)
    cdf = np.zeros(256)
    mean = np.zeros(256)
    prob[0] = cdf[0] = histogram[0] / N
    for i in range(1, 256):
        prob[i] = histogram[i] / N
        cdf[i] = cdf[i - 1] + prob[i]
        mean[i] = mean[i - 1] + i * prob[i]

    # maximize the Variance
    Var_max = 0
    thresh = 0
    for T in range(1, 256):
        Var = np.square(mean[255] * cdf[T] - mean[T]) / (cdf[T] * (1 - cdf[T]) + 1e-5)
        if Var > Var_max:
            Var_max = Var
            thresh = T
    return thresh


def entropy(histogram: np.array) -> int:
    smooth = 1e-5
    N = np.sum(histogram)
    # frequency
    prob = np.zeros(256)
    cdf = np.zeros(256)
    prob[0] = cdf[0] = histogram[0] / N
    for i in range(1, 256):
        prob[i] = histogram[i] / N
        cdf[i] = cdf[i - 1] + prob[i]

    # maximize the entropy
    H_max = 0
    thresh = 0
    for T in range(0, 255):
        # foreground entropy H_b
        H_b = 0
        for i in range(0, T + 1):
            prob_fg = prob[i] / (cdf[T] + smooth)
            H_b = H_b - prob_fg * np.log(prob_fg + smooth)
        # background entropy H_w
        H_w = 0
        for i in range(T + 1, 256):
            prob_bg = prob[i] / (1 - cdf[T] + smooth)
            H_w = H_w - prob_bg * np.log(prob_bg + smooth)
        H = H_b + H_w
        if H > H_max:
            H_max = H
            thresh = T
    return thresh


def main():
    import time
    import cv2
    from matplotlib import pyplot as plt

    image_path = r'.\sa_183.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    start = time.time()
    histogram = get_histogram(image)
    # thresh = otsu(histogram)
    # thresh = entropy(histogram)
    # image_thresh = threshold(image, thresh)
    end = time.time()
    print(image.shape)
    print(end - start)

    # plt.subplot(311)
    # plt.imshow(image, cmap='gray')
    # plt.subplot(312)
    # plt.plot(histogram)
    # plt.axvline(x=thresh)
    # plt.subplot(313)
    # plt.imshow(image_thresh, cmap='gray')
    # plt.show()
    # print(thresh)


if __name__ == '__main__':
    main()

