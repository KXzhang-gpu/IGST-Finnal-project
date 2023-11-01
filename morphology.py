# -*- coding: UTF-8 -*-
import numpy as np

from convolution import conv2d_img2col as conv2d

square_SE = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
cross_SE = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
disc_SE = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
disc_SE5 = [[0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]]


def erosion(image: np.array, kernel: np.array):
    H, W = image.shape
    border_h = kernel.shape[0] // 2
    border_w = kernel.shape[1] // 2
    eroded_image = np.zeros_like(image)

    for i in range(border_h, H - border_h):
        for j in range(border_w, W - border_w):
            patch = image[i - border_h: i + border_h + 1, j - border_w: j + border_w + 1]
            eroded_image[i, j] = np.min(patch * kernel)
    return eroded_image


def dilation(image: np.array, kernel: np.array):
    H, W = image.shape
    border_h = kernel.shape[0] // 2
    border_w = kernel.shape[1] // 2
    dilated_image = np.zeros_like(image)

    for i in range(border_h, H - border_h):
        for j in range(border_w, W - border_w):
            patch = image[i - border_h: i + border_h + 1, j - border_w: j + border_w + 1]
            dilated_image[i, j] = np.max(np.dot(patch, kernel))

    return dilated_image


def binary_erosion(image, kernel, anchor=(-1, -1)):
    """
    The binary erosion operator for 2d gray/binary image

    Parameters
    ----------
    image : array_like
        Input image
    kernel : array_like
        structure element kernel, the shape of kernel is square
    anchor : tuple
        the coordinate of the kernal anchor, (-1,-1) denotes that the anchor is in the kernal center
    """
    if not anchor == (-1, -1):
        raise ValueError('this parameter is not completed yet, please wait a minute')
    eroded_image = conv2d(image, kernel)
    return (eroded_image == np.sum(kernel)).astype(int)


def binary_dilation(image, kernel, anchor=(-1, -1)):
    """
    The binary dilation operator for 2d gray/binary image

    Parameters
    ----------
    image : array_like
        Input image
    kernel : array_like
        structure element kernel, the shape of kernel is square
    anchor : tuple
        the coordinate of the kernal anchor, (-1,-1) denotes that the anchor is in the kernal center
    """
    if not anchor == (-1, -1):
        raise ValueError('this parameter is not completed yet, please wait a minute')
    eroded_image = conv2d(image, kernel)
    return (eroded_image > 0).astype(np.uint8)


def opening(image, kernel, anchor=(-1, -1), erosion=binary_erosion, dilation=binary_dilation):
    image = np.asarray(image)
    image = erosion(image, kernel, anchor)
    image = dilation(image, kernel, anchor)
    return image


def closing(image, kernel, anchor=(-1, -1), erosion=binary_erosion, dilation=binary_dilation):
    image = np.asarray(image)
    image = dilation(image, kernel, anchor)
    image = erosion(image, kernel, anchor)
    return image


def normalization(image):
    image = np.asarray(image)
    i_max = np.max(image)
    return (image / i_max * 255).astype(np.uint8)


def distance_transform(image, mode='chessboard'):
    """
    The implementation of distance transform for 2d binary image

    Parameters
    ----------
    image : array_like
        Input image
    mode : str, optional
        distance mode, including 'chessboard', 'city_block' and 'Euclidean'
    """
    image = np.asarray(image)
    if mode == 'chessboard':
        kernel = square_SE
    elif mode == 'city_block':
        kernel = cross_SE
    elif mode == 'Euclidean':
        # todo: disc kernel setting
        kernel = disc_SE5
    else:
        raise ValueError("the mode must be 'chessboard', 'city_block' or 'Euclidean'")

    output = np.zeros_like(image)
    for i in range(np.sum(image)):
        eroded_image = binary_erosion(image, kernel)
        output = output + (image - eroded_image) * (i + 1)
        image = eroded_image
        if np.sum(image) == 0:
            output = normalization(output)
            return output
    raise ValueError('fatal error, check the code!')


def skeletonization(image, get_sub_skeleton=False):
    image = np.asarray(image)
    subset = image.copy()
    kernel = disc_SE
    skeleton = np.zeros_like(image)
    if get_sub_skeleton:
        sub_skeletons = []
    r = 1
    while np.sum(subset) > 0:
        subset = image.copy()
        for i in range(r):
            subset = binary_erosion(subset, kernel)
        subset = subset - opening(subset, kernel)
        r += 1
        skeleton = skeleton + subset
        if get_sub_skeleton:
            sub_skeletons.append((subset > 0).astype(np.uint8))  # noqa
    skeleton[skeleton > 0] = 1
    if get_sub_skeleton:
        return skeleton, sub_skeletons
    return skeleton


def skeleton_reconstruction(sub_skeletons):
    kernel = disc_SE
    image = np.zeros_like(sub_skeletons[0])
    for r, sub_skeleton in enumerate(sub_skeletons):
        for i in range(r + 1):
            sub_skeleton = binary_dilation(sub_skeleton, kernel)
        image = image + sub_skeleton
    image[image > 0] = 1
    return image


if __name__ == '__main__':
    import time
    import cv2
    from matplotlib import pyplot as plt

    # import skimage.morphology as sm

    image_path = r'.\horse.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = (image > 50).astype(np.uint8)
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])
    big_kernel = np.ones((11, 11))
    start = time.time()
    # output = binary_erosion(image, kernel=big_kernel)
    # output = opening(image, kernel=kernel)
    # output = distance_transform(image, mode='Euclidean')
    output, subs = skeletonization(image, get_sub_skeleton=True)
    restore = skeleton_reconstruction(subs)
    end = time.time()
    print(end - start)

    plt.subplot(311)
    plt.imshow(image, cmap='gray')
    plt.subplot(312)
    plt.imshow(output, cmap='binary')
    plt.subplot(313)
    plt.imshow(restore, cmap='gray')
    plt.show()
    # sm.binary_erosion
