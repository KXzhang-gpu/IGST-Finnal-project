# -*- coding: UTF-8 -*-
import numpy as np

from threshold import get_histogram

# Edge Operations
Roberts_kernel_x = [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
Roberts_kernel_y = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
Prewitt_kernel_x = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
Prewitt_kernel_y = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
Sobel_kernel_x = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
Sobel_kernel_y = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]


def padding_non_suqare(image, k_shape, padding):
    """
    padding function for non-suqare kernel
    image: input image
    k_shape: tuple
        the shape of kernal (k_h, k_w)
    """
    h, w = k_shape
    h_pad, w_pad = [h // 2, h // 2], [w // 2, w // 2]
    if h % 2 == 0:
        h_pad[1] = h_pad[1] - 1
    if w % 2 == 0:
        w_pad[1] = w_pad[1] - 1
    pad_width = (h_pad, w_pad)
    return np.pad(image, pad_width=pad_width, mode=padding)


def conv2d(image, kernel, padding='constant', flip=False):
    """
    The convolution operator for 2d gray image

    Parameters
    ----------
    image : array_like
        Input image
    kernel : array_like
        convolution kernel, the shape of kernel is square
    padding : str, optional
        padding mode for convolution, including 'constant', 'reflect', 'symmetric', 'wrap'
        more padding mode can be found at numpy.pad() parameter 'mode'
    flip: bool, optional
    """
    image = np.array(image)
    kernel = np.asarray(kernel)
    # flip the kernel
    if flip:
        kernel = kernel[::-1, ...][:, ::-1]

    output = np.zeros_like(image)
    kernel_size = kernel.shape[0]
    image = np.pad(image, pad_width=kernel_size // 2, mode=padding)
    for h in range(output.shape[0]):
        for w in range(output.shape[1]):
            patch = image[h:h + kernel_size, w:w + kernel_size]
            output[h, w] = np.sum(patch * kernel)
    return output


def conv2d_img2col(image, kernel, padding='constant', flip=False):
    """
    The convolution operator for 2d gray image by using img2col

    Parameters
    ----------
    image : array_like
        Input image
    kernel : array_like
        convolution kernel, the shape of kernel is square
    padding : str, optional
        padding mode for convolution, including 'constant', 'reflect', 'symmetric', 'wrap'
        more padding mode can be found at numpy.pad() parameter 'mode'
    flip : bool, optional
    """
    image = np.array(image)
    kernel = np.asarray(kernel)
    # flip the kernel
    if flip:
        kernel = kernel[::-1, ...][:, ::-1]

    # image = np.pad(image, pad_width=kernel.shape[0] // 2, mode=padding)
    image = padding_non_suqare(image, kernel.shape, padding)

    # for images having additional channel:
    # sub_shape = (other_channels..., output_h, output_w, kernel_h, kernel_w)
    # sub_strides = image.strides[:-2] + image.strides[-2:] * stride + image.strides[-2:]
    sub_shape = tuple(np.subtract(image.shape, kernel.shape) + 1) + kernel.shape
    sub_strides = image.strides * 2
    sub_patches = np.lib.stride_tricks.as_strided(image, shape=sub_shape, strides=sub_strides)
    return np.einsum('ij,klij->kl', kernel, sub_patches)


def conv2d_fft(image, kernel):
    image = np.asarray(image)
    kernel = np.asarray(kernel)
    # kernel np.flipud(np.fliplr(kernel))

    h, w = image.shape
    k_h, k_w = kernel.shape
    kernel = np.pad(kernel, pad_width=((0, h - k_h), (0, w - k_w)))
    image_fr = np.fft.fft2(image)
    kernel_fr = np.fft.fft2(kernel)
    output = np.real(np.fft.ifft2(image_fr * kernel_fr))
    return output


def edge_operation(image, mode='Sobel'):
    if mode == 'Roberts':
        g_x = conv2d_img2col(image, Roberts_kernel_x)
        g_y = conv2d_img2col(image, Roberts_kernel_y)
    elif mode == 'Prewitt':
        g_x = conv2d_img2col(image, Prewitt_kernel_x)
        g_y = conv2d_img2col(image, Prewitt_kernel_y)
    elif mode == 'Sobel':
        g_x = conv2d_img2col(image, Sobel_kernel_x)
        g_y = conv2d_img2col(image, Sobel_kernel_y)
    else:
        raise ValueError('Operation mode can not be found! Please choose from Roberts, Prewitt and Sobel.')
    return g_x, g_y


def Gaussian_filter(image, kernel_size: int, sigma: float):
    gaussian_kernel = creat_gaussian_kernel(kernel_size, sigma)
    blur_image = conv2d_img2col(image, gaussian_kernel)
    return blur_image


def creat_gaussian_kernel(kernel_size: int, sigma: float):
    if sigma == 0:
        # if sigma==0, it is computed from ksize as following, according to cv2.getGaussianKernel
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    radium = kernel_size // 2
    X = np.linspace(-radium, radium, kernel_size)
    Y = np.linspace(-radium, radium, kernel_size)
    x, y = np.meshgrid(X, Y)
    gauss = 1 / (2 * np.pi * sigma ** 2) * np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2))
    # Normalization
    Z = gauss.sum()
    gauss = (1 / Z) * gauss
    return gauss


def median_filter(image, kernel_size: int):
    image = np.array(image)
    output = np.zeros_like(image)
    image = np.pad(image, pad_width=kernel_size // 2, mode='constant')
    for h in range(output.shape[0]):
        for w in range(output.shape[1]):
            output[h, w] = np.median(image[h:h + kernel_size, w:w + kernel_size])
    return output


class Histogram:
    def __init__(self, image, threshold):
        self.histogram = get_histogram(image)
        self.threshold = threshold
        self.init_median_value()

    def init_median_value(self):
        self.sumCnt = 0
        self.median_val = 0
        for i in range(256):
            self.sumCnt += self.histogram[i]
            if self.sumCnt >= self.threshold:
                self.median_val = i
                break

    def get_median_value(self):
        return self.median_val

    def update(self, left_col, right_col):
        histogram = self.histogram
        sumCnt, median_val, threshold = self.sumCnt, self.median_val, self.threshold

        # update histogram
        histogram[left_col] -= 1
        histogram[right_col] += 1
        sumCnt -= np.sum(left_col <= median_val)
        sumCnt += np.sum(right_col <= median_val)
        if sumCnt < threshold:
            for i in range(median_val + 1, 256):
                sumCnt += histogram[i]
                if sumCnt >= threshold:
                    median_val = i
                    break
        elif sumCnt > threshold:
            for i in range(median_val-1, -1, -1):
                sumCnt -= histogram[i]
                if sumCnt < threshold:
                    sumCnt += histogram[i+1]
                    median_val = i
                    break

        self.histogram = histogram
        self.sumCnt = sumCnt
        self.median_val = median_val


def fast_median_filter(image, kernel_size: int):
    """
    Unfortunately this so-called fast algorithm is much slower than the normal one, which called median_filter()
    The reason could be that np.median() is a very efficient algorithm greatly accelerated by using C.
    """
    image = np.array(image)
    output = np.zeros_like(image)
    image = np.pad(image, pad_width=kernel_size // 2, mode='constant')
    threshold = kernel_size**2 // 2
    for h in range(output.shape[0]):
        patch = image[h:h + kernel_size, 0:kernel_size]
        Hist = Histogram(patch, threshold)
        output[h, 0] = Hist.median_val
        for w in range(1, output.shape[1]):
            left_col = image[h:h + kernel_size, w - 1]
            right_col = image[h:h + kernel_size, w + kernel_size - 1]
            Hist.update(left_col, right_col)
            output[h, w] = Hist.median_val
    return output


if __name__ == '__main__':
    import time
    import cv2
    # from scipy import signal
    from matplotlib import pyplot as plt

    # image = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # image = np.random.randint(0, 255, (543, 543))
    image_path = r'.\classical_image.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    # kernel = np.ones((100, 1))
    kernel = np.ones((3, 3))
    start = time.time()
    # output = conv2d(image, kernel)
    # output = conv2d_fft(image, kernel)
    output = conv2d_img2col(image, kernel, flip=True)
    # output_refer = signal.convolve2d(image, kernel, mode='same')
    # print((output == output_refer).all())
    # output = median_filter(image, kernel_size=11)
    # output = fast_median_filter(image, kernel_size=11)
    end = time.time()
    print(end - start)

    # output[output > 255] = 255
    # output[output < 0] = 0
    plt.subplot(211)
    plt.imshow(image, cmap='gray')
    plt.subplot(212)
    plt.imshow(output, cmap='gray')
    plt.show()
    # kernel_1d = cv2.getGaussianKernel(5, 1)
    # kernel_2d_ref = kernel_1d * kernel_1d.T
    # kernel_2d = creat_gaussian_kernel(5, 1)
    # print((kernel_2d == kernel_2d_ref).all())
    # print(kernel_2d)
    # print(kernel_2d_ref)
