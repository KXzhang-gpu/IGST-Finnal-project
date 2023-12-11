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


def erosion(image, kernel, anchor=(-1, -1)):
    image = np.array(image)
    kernel = np.asarray(kernel)

    output = np.zeros_like(image)
    kernel_size = kernel.shape[0]
    image = np.pad(image, pad_width=kernel.shape[0] // 2, mode='edge')

    for h in range(output.shape[0]):
        for w in range(output.shape[1]):
            patch = image[h:h + kernel_size, w:w + kernel_size]
            output[h, w] = np.min(patch - kernel)
    return output


def dilation(image, kernel, anchor=(-1, -1)):
    image = np.array(image)
    kernel = np.asarray(kernel)

    output = np.zeros_like(image)
    kernel_size = kernel.shape[0]
    image = np.pad(image, pad_width=kernel.shape[0] // 2, mode='edge')

    for h in range(output.shape[0]):
        for w in range(output.shape[1]):
            patch = image[h:h + kernel_size, w:w + kernel_size]
            output[h, w] = np.max(patch + kernel)
    return output


def erosion_img2col(image, kernel, anchor=(-1, -1)):
    image = np.array(image)
    kernel = np.asarray(kernel)

    image = np.pad(image, pad_width=kernel.shape[0] // 2, mode='edge')

    sub_shape = tuple(np.subtract(image.shape, kernel.shape) + 1) + kernel.shape
    sub_strides = image.strides * 2
    sub_patches = np.lib.stride_tricks.as_strided(image, shape=sub_shape, strides=sub_strides)
    # todo how to deal with negative number
    return np.min(sub_patches - kernel[None, None, ...], axis=(-2, -1))


def dilation_img2col(image, kernel, anchor=(-1, -1)):
    image = np.array(image)
    kernel = np.asarray(kernel)

    image = np.pad(image, pad_width=kernel.shape[0] // 2, mode='edge')

    sub_shape = tuple(np.subtract(image.shape, kernel.shape) + 1) + kernel.shape
    sub_strides = image.strides * 2
    sub_patches = np.lib.stride_tricks.as_strided(image, shape=sub_shape, strides=sub_strides)
    return np.max(sub_patches + kernel[None, None, ...], axis=(-2, -1))


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
    eroded_image = conv2d(image, kernel, flip=True)
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
    eroded_image = conv2d(image, kernel, flip=True)
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


def gray_opening(image, kernel, anchor=(-1, -1)):
    return opening(image, kernel, anchor, erosion=erosion_img2col, dilation=dilation_img2col)


def gray_closing(image, kernel, anchor=(-1, -1)):
    return closing(image, kernel, anchor, erosion=erosion_img2col, dilation=dilation_img2col)


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


def skeletonization(image, get_sub_skeleton=False, UI=None):
    image = np.asarray(image)
    subset = image.copy()
    kernel = disc_SE
    skeleton = np.zeros_like(image)
    if get_sub_skeleton:
        sub_skeletons = []
    r = 1
    while np.sum(subset) > 0:
        # get the sub skeleton
        subset = image.copy()
        for i in range(r):
            subset = binary_erosion(subset, kernel)
        subset = subset - opening(subset, kernel)
        skeleton = skeleton + subset

        # For Qt: update progress
        if UI is not None:
            if r < 100:
                UI.progressBar.setValue(r)
                _skeleton = (skeleton > 0).astype(int)
                UI.print_image(_skeleton * 255, UI.viewLeftBottom, UI.labelLB, title='Skeletonizaiton')
        r += 1
        if get_sub_skeleton:
            sub_skeletons.append((subset > 0).astype(np.uint8))  # noqa
    skeleton[skeleton > 0] = 1
    if get_sub_skeleton:
        return skeleton, sub_skeletons
    return skeleton


def skeleton_reconstruction(sub_skeletons, UI=None):
    kernel = disc_SE
    image = np.zeros_like(sub_skeletons[0])
    for r, sub_skeleton in enumerate(sub_skeletons):
        for i in range(r + 1):
            sub_skeleton = binary_dilation(sub_skeleton, kernel)
        image = image + sub_skeleton

        # For Qt: update progress
        if UI is not None:
            if r < 100:
                UI.progressBar.setValue(r)
                _image = (image > 0).astype(int)
                UI.print_image(_image * 255, UI.viewRightBottom, UI.labelRB, title='Skeleton reconstruction')
    image[image > 0] = 1
    return image


def edge_decetion(image, mode='standard'):
    """
    The implementation of edge decetion for 2d grayscale image

    Parameters
    ----------
    image : array_like
        Input image
    mode : str, optional
        decetion mode, including 'standard', 'external' and 'internal'
    """
    image = np.asarray(image)
    kernel = square_SE
    # todo how to deal with negative number
    if mode == 'standard':
        return dilation_img2col(image, kernel) - erosion_img2col(image, kernel)
    elif mode == 'external':
        return dilation_img2col(image, kernel) - image
    elif mode == 'internal':
        return image - erosion_img2col(image, kernel)
    else:
        raise ValueError("the mode must be 'standard', 'external' or 'internal'")


def get_gradient(image, mode='strandard'):
    return edge_decetion(image, mode) / 2


def MSmooth(image, k_size=3):
    kernel = np.ones((k_size, k_size))
    return gray_closing(gray_opening(image, kernel), kernel)


def DSmooth(image, k_size=3):
    kernel = np.ones((k_size, k_size))
    return (dilation_img2col(image, kernel) + erosion_img2col(image, kernel)) / 2


def top_hat_transform(image, k_size=3, mode='WTT'):
    image = np.asarray(image)
    kernel = np.ones((k_size, k_size))
    if mode == 'WTT':
        return image - gray_opening(image, kernel)
    elif mode == 'BTT':
        return gray_closing(image, kernel) - image
    else:
        raise ValueError("the mode must be 'WTT' or 'BTT'")


def conditional_dilation(marker, mask, kernel, anchor=(-1, -1)):
    try:
        while True:
            marker_prev = marker.copy()
            marker = binary_dilation(marker_prev, kernel, anchor)
            marker = marker * mask
            if np.array_equal(marker, marker_prev):
                break
    except Exception as e:
        print("An error occurred during conditional dilation:", str(e))
    return marker


def grayscale_reconstruction(marker, mask, kernel, anchor=(-1, -1)):
    try:
        while True:
            marker_prev = marker.copy()
            marker = dilation_img2col(marker_prev, kernel, anchor)
            marker = np.minimum(marker, mask)
            if np.array_equal(marker, marker_prev):
                break
    except Exception as e:
        print("An error occurred during grayscale reconstruction:", str(e))
    return marker


if __name__ == '__main__':
    import time
    import cv2
    from matplotlib import pyplot as plt

    # import skimage.morphology as sm

    image_path = r'.\conditional_dilation_mask.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = (image < 50).astype(np.uint8)
    marker_path = r'.\conditional_dilation_marker.png'
    marker = cv2.imread(marker_path, cv2.IMREAD_GRAYSCALE)
    marker = cv2.resize(marker,
                        dsize=(image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    marker = (marker < 50).astype(np.uint8)
    # cv2.imwrite('1.png', image*255)
    # cv2.imwrite('2.png', marker*255)
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])
    big_kernel = np.ones((11, 11))
    start = time.time()
    # output = binary_erosion(image, kernel=kernel)
    # output = opening(image, kernel=kernel)
    # output = distance_transform(image, mode='Euclidean')
    # output, subs = skeletonization(image, get_sub_skeleton=True)
    # restore = skeleton_reconstruction(subs)

    # output = dilation_img2col(image, big_kernel)
    # output = closing(image, big_kernel, erosion=erosion_img2col, dilation=dilation_img2col)
    # output = top_hat_transform(image, k_size=21)
    output = conditional_dilation(marker, image, kernel)
    end = time.time()
    print(end - start)

    plt.subplot(211)
    plt.imshow(image, cmap='gray')
    plt.subplot(212)
    plt.imshow(output, cmap='gray')
    plt.show()
    # sm.binary_erosion
