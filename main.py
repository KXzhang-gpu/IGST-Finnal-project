# -*- coding: UTF-8 -*-
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QGraphicsScene
from PyQt5.QtCore import Qt, QTimer

from threshold import *
from convolution import conv2d_img2col as conv2d
from convolution import Gaussian_filter, median_filter
from morphology import *
from draw_board_Qt import DrawingScene

import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)


# Edge Operators
Roberts_kernel_x = [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
Roberts_kernel_y = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
Prewitt_kernel_x = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
Prewitt_kernel_y = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
Sobel_kernel_x = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
Sobel_kernel_y = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]


def check_image_loaded(func):
    def wrapper(self):
        if self.image is None:
            msg_box = QMessageBox(QMessageBox.Critical, 'Error', 'Please load image first!')
            msg_box.exec_()
        else:
            return func(self)
    return wrapper


def normalization(image: np.array):
    return (image - np.min(image)) / (np.max(image) - np.min(image)) * 255


def get_disk_se(radius):
    size = 2 * radius + 1
    se = np.zeros((size, size), dtype=np.uint8)

    center = (radius, radius)
    cv2.circle(se, center, radius, 1, thickness=-1) # noqa

    return se


class mainWindow(QMainWindow):
    def __init__(self):
        super(mainWindow, self).__init__()
        uic.loadUi('./main.ui', self)
        self.setFont(QFont('times new roman'))
        self.image = None
        self.thres_image = None
        self.sub_skeletons = None
        self.binMarker = None
        self.grayMarker = None
        self.button_init()
        self.component_init()

    def button_init(self):
        # initialize all the button
        self.selectBtn.clicked.connect(self._selectBtn_click)
        self.histogramBtn.clicked.connect(self._histogramBtn_click)
        self.thresholdBtn.clicked.connect(self._thresholdBtn_click)
        self.edgeFiltBtn.clicked.connect(self._edgeFiltBtn_click)
        self.binOperationBtn.clicked.connect(self._binOperationBtn_click)
        self.distanceBtn.clicked.connect(self._distanceBtn_click)
        self.skeletonBtn.clicked.connect(self._skeletonBtn_click)
        self.grayOperationBtn.clicked.connect(self._grayOperationBtn_click)
        self.grayEdgeGradBtn.clicked.connect(self._grayEdgeGradBtn_click)
        self.binMarkerBtn.clicked.connect(self._binMarkerBtn_click)
        self.grayMarkerBtn.clicked.connect(self._grayMarkerBtn_click)
        self.conditionalDilationBtn.clicked.connect(self._conditionalDilationBtn_click)
        self.grayReconstructBtn.clicked.connect(self._grayReconstructBtn_click)

    def _selectBtn_click(self):
        # load image form given path
        file_path, _ = QFileDialog.getOpenFileName(self, "Open image", "img", "*.jpg;*.tif;*.png;;All Files(*)")
        if not file_path == '':
            # clear pervious image view and data
            self._clear()
            self.filePath.setText(file_path)
            # print image to ui
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.print_image(self.image, self.viewLeftTop, self.labelLT, 'Original image')

    @check_image_loaded
    def _histogramBtn_click(self):
        histogram = get_histogram(self.image)
        self.drawHist(histogram, self.viewLeftBottom, self.labelLB)

    @check_image_loaded
    def _thresholdBtn_click(self):
        histogram = get_histogram(self.image)
        if self.mannualBtn.isChecked():
            thres = self.spinBox.value()
        elif self.otsuBtn.isChecked():
            thres = otsu(histogram)
        elif self.entropyBtn.isChecked():
            thres = entropy(histogram)
        else:
            raise ValueError
        self.thres_image = threshold(self.image, thres)
        self.print_image(self.thres_image * 255, self.viewRightTop, self.labelRT,
                         title='Thresholded image with threshold=%d' % thres)
        self.drawHist(histogram, self.viewLeftBottom, self.labelLB)

    @check_image_loaded
    def _edgeFiltBtn_click(self):
        # edge detection
        operator = None
        edge_operator_name = self.edgeCombo.currentText()
        if edge_operator_name == 'Roberts':
            operator = Roberts_kernel_x + Roberts_kernel_y
        elif edge_operator_name == 'Prewitt':
            operator = Prewitt_kernel_x + Prewitt_kernel_y
        elif edge_operator_name == 'Sobel':
            operator = Sobel_kernel_x + Sobel_kernel_y
        edge = conv2d(self.image, kernel=operator)
        edge = normalization(edge)
        self.print_image(edge, self.viewLeftBottom, self.labelLB,
                         title='Edge detection of ' + edge_operator_name + 'opterator')

        # smoothing the image
        filtered_image = None
        filter_name = self.filterCombo.currentText()
        kernel_size = self.ksizeSpin.value()
        title = 'Results of %s filter with kernal size=%s' % (filter_name, kernel_size)
        if filter_name == 'Mean':
            kernel = np.ones((kernel_size, kernel_size))
            filtered_image = conv2d(self.image, kernel=kernel) / kernel_size**2
        elif filter_name == 'Gaussian':
            sigma = self.sigmaSpin.value()
            filtered_image = Gaussian_filter(self.image, kernel_size=kernel_size, sigma=sigma)
            title = title + ', sigma=' + str(sigma)
        elif filter_name == 'Medium':
            filtered_image = median_filter(self.image, kernel_size=kernel_size)
        self.print_image(filtered_image, self.viewRightBottom, self.labelRB, title=title)

    @check_image_loaded
    def _binOperationBtn_click(self):
        # get kernel
        kernel_shape = self.binSEshapeCombo.currentText()
        kernel_size = self.binSEsizeSpin.value()
        kernel = None
        if kernel_shape == 'Square':
            kernel = np.ones((kernel_size, kernel_size))
        elif kernel_shape == 'Cross':
            kernel = np.zeros((kernel_size, kernel_size))
            center = kernel_size // 2
            kernel[center, :] = 1
            kernel[:, center] = 1
        elif kernel_shape == 'Disc':
            radius = kernel_size // 2
            kernel = get_disk_se(radius)
        else:
            msg_box = QMessageBox(QMessageBox.Critical, 'Sorry',
                                  'Sorry, user custom option is not avialable now, please try other choice!')
            msg_box.exec_()

        # check binary image available
        if self.thres_image is None:
            histogram = get_histogram(self.image)
            thres = otsu(histogram)
            self.thres_image = threshold(self.image, thres)
            title = 'Auto threshold by Otsu with threshold=%d' % thres
            self.print_image(self.thres_image * 255, self.viewRightTop, self.labelRT, title=title)

        # binary morphplogy operation
        if self.binErosionBtn.isChecked():
            viewLB_image = binary_erosion(self.thres_image, kernel)
            titleLB = 'Result of binary erosion'
        elif self.binOpenBtn.isChecked():
            viewLB_image = opening(self.thres_image, kernel)
            titleLB = 'Result of binary opening'
        if self.binDilationBtn.isChecked():
            viewRB_image = binary_dilation(self.thres_image, kernel)
            titleRB = 'Result of binary dilation'
        elif self.binCloseBtn.isChecked():
            viewRB_image = closing(self.thres_image, kernel)
            titleRB = 'Result of binary closing'

        # show the results
        viewLB_image = viewLB_image * 255 # noqa
        viewRB_image = viewRB_image * 255 # noqa
        self.print_image(viewLB_image, self.viewLeftBottom, self.labelLB, title=titleLB) # noqa
        self.print_image(viewRB_image, self.viewRightBottom, self.labelRB, title=titleRB) # noqa

    @check_image_loaded
    def _distanceBtn_click(self):
        mode = self.DTmodeCombo.currentText()

        # check binary image available
        if self.thres_image is None:
            histogram = get_histogram(self.image)
            thres = otsu(histogram)
            self.thres_image = threshold(self.image, thres)
            title = 'Auto threshold by Otsu with threshold=%d' % thres
            self.print_image(self.thres_image * 255, self.viewRightTop, self.labelRT, title=title)

        distance = distance_transform(self.thres_image, mode)
        self.print_image(normalization(distance), self.viewLeftBottom, self.labelLB, title='Distance Tranform')

    @check_image_loaded
    def _skeletonBtn_click(self):
        # check binary image available
        if self.thres_image is None:
            histogram = get_histogram(self.image)
            thres = otsu(histogram)
            self.thres_image = threshold(self.image, thres)
            title = 'Auto threshold by Otsu with threshold=%d' % thres
            self.print_image(self.thres_image * 255, self.viewRightTop, self.labelRT, title=title)

        self.progressBar.setValue(0)
        self.progressBar.setVisible(True)
        if self.getSkeletonBtn.isChecked():
            skeleton, self.sub_skeletons = skeletonization(self.thres_image, get_sub_skeleton=True, UI=self)
            self.print_image(skeleton * 255, self.viewLeftBottom, self.labelLB, title='Skeletonizaiton')
        elif self.restoreSkeletonBtn.isChecked():
            if self.sub_skeletons is None:
                pass
            else:
                image = skeleton_reconstruction(self.sub_skeletons, UI=self)
                self.print_image(image * 255, self.viewRightBottom, self.labelRB, title='Skeleton reconstruction')
        self.progressBar.setValue(100)

    @check_image_loaded
    def _grayOperationBtn_click(self):
        # get kernel
        kernel_shape = self.graySEshapeCombo.currentText()
        kernel_size = self.graySEsizeSpin.value()
        kernel = None
        if kernel_shape == 'Square':
            kernel = np.ones((kernel_size, kernel_size))
        elif kernel_shape == 'Cross':
            kernel = np.zeros((kernel_size, kernel_size))
            center = kernel_size // 2
            kernel[center, :] = 1
            kernel[:, center] = 1
        elif kernel_shape == 'Disc':
            radius = kernel_size // 2
            kernel = get_disk_se(radius)
        else:
            msg_box = QMessageBox(QMessageBox.Critical, 'Sorry',
                                  'Sorry, user custom option is not avialable now, please try other choice!')
            msg_box.exec_()

        # binary morphplogy operation
        if self.binErosionBtn.isChecked():
            viewLB_image = erosion_img2col(self.image, kernel)
            titleLB = 'Result of gray erosion'
        elif self.binOpenBtn.isChecked():
            viewLB_image = gray_opening(self.image, kernel)
            titleLB = 'Result of gray opening'
        if self.binDilationBtn.isChecked():
            viewRB_image = dilation_img2col(self.image, kernel)
            titleRB = 'Result of gray dilation'
        elif self.binCloseBtn.isChecked():
            viewRB_image = gray_closing(self.image, kernel)
            titleRB = 'Result of gray closing'

        # show the results
        self.print_image(viewLB_image, self.viewLeftBottom, self.labelLB, title=titleLB)  # noqa
        self.print_image(viewRB_image, self.viewRightBottom, self.labelRB, title=titleRB)  # noqa

    @check_image_loaded
    def _grayEdgeGradBtn_click(self):
        mode = self.grayEdgeCombo.currentText()
        if self.getGrayEdgeBtn.isChecked():
            edge = edge_decetion(self.image, mode=mode)
            self.print_image(normalization(edge), self.viewLeftBottom, self.labelLB,
                             title='Gray scale edge decetion')
        elif self.getGrayGradBtn.isChecked():
            grad = get_gradient(self.image, mode=mode)
            self.print_image(normalization(grad), self.viewRightBottom, self.labelRB,
                             title='Gray scale gradient of input image')

    @check_image_loaded
    def _binMarkerBtn_click(self):
        # load image form given path
        file_path, _ = QFileDialog.getOpenFileName(self, "Set your Marker", "img", "*.jpg;*.tif;*.png;;All Files(*)")
        if not file_path == '':
            # print marker to ui
            self.binMarker = threshold(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE), 50)
            h, w = self.image.shape
            self.binMarker = cv2.resize(self.binMarker, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            self.print_image(self.binMarker * 255, self.viewLeftBottom, self.labelLB, title='Marker')

        # self.draw_board(self.thres_image * 255, self.viewLeftBottom, self.labelLB, title='Set your Marker here')

    @check_image_loaded
    def _grayMarkerBtn_click(self):
        # load image form given path
        file_path, _ = QFileDialog.getOpenFileName(self, "Set your Marker", "img", "*.jpg;*.tif;*.png;;All Files(*)")
        # print marker to ui
        if not file_path == '':
            self.grayMarker = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            h, w = self.image.shape
            self.grayMarker = cv2.resize(self.grayMarker, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
            self.print_image(self.grayMarker, self.viewLeftBottom, self.labelLB, title='Marker')

    @check_image_loaded
    def _conditionalDilationBtn_click(self):
        if self.binMarker is None:
            msg_box = QMessageBox(QMessageBox.Critical, 'Error', 'Please set binary Marker first!')
            msg_box.exec_()
        else:
            if self.thres_image is None:
                histogram = get_histogram(self.image)
                thres = otsu(histogram)
                self.thres_image = threshold(self.image, thres)
                title = 'Auto threshold by Otsu with threshold=%d' % thres
                self.print_image(self.thres_image * 255, self.viewRightTop, self.labelRT, title=title)

            # get kernel
            kernel_shape = self.conditionSEshapeCombo.currentText()
            kernel_size = self.conditionSEsizeSpin.value()
            kernel = None
            if kernel_shape == 'Square':
                kernel = np.ones((kernel_size, kernel_size))
            elif kernel_shape == 'Cross':
                kernel = np.zeros((kernel_size, kernel_size))
                center = kernel_size // 2
                kernel[center, :] = 1
                kernel[:, center] = 1
            elif kernel_shape == 'Disc':
                radius = kernel_size // 2
                kernel = get_disk_se(radius)
            else:
                msg_box = QMessageBox(QMessageBox.Critical, 'Sorry',
                                      'Sorry, user custom option is not avialable now, please try other choice!')
                msg_box.exec_()

            output = conditional_dilation(marker=self.binMarker, mask=self.thres_image, kernel=kernel)
            self.print_image(output * 255, self.viewRightBottom, self.labelRB, title='Conditional dilation result')

    @check_image_loaded
    def _grayReconstructBtn_click(self):
        if self.grayMarker is None:
            msg_box = QMessageBox(QMessageBox.Critical, 'Error', 'Please set gray scale Marker first!')
            msg_box.exec_()
        else:
            # get kernel
            kernel_shape = self.conditionSEshapeCombo.currentText()
            kernel_size = self.conditionSEsizeSpin.value()
            kernel = None
            if kernel_shape == 'Square':
                kernel = np.ones((kernel_size, kernel_size))
            elif kernel_shape == 'Cross':
                kernel = np.zeros((kernel_size, kernel_size))
                center = kernel_size // 2
                kernel[center, :] = 1
                kernel[:, center] = 1
            elif kernel_shape == 'Disc':
                radius = kernel_size // 2
                kernel = get_disk_se(radius)
            else:
                msg_box = QMessageBox(QMessageBox.Critical, 'Sorry',
                                      'Sorry, user custom option is not avialable now, please try other choice!')
                msg_box.exec_()

            output = grayscale_reconstruction(marker=self.grayMarker, mask=self.image, kernel=kernel)
            self.print_image(output, self.viewRightBottom, self.labelRB, title='Gray scale reconstruction result')

    def component_init(self):
        # initialize all the component except button
        self.splider.valueChanged.connect(self._splider_change)
        self.spinBox.valueChanged.connect(self._spinbox_change)
        self.binSEsizeSpin.valueChanged.connect(self._binSEsizeSpin_change)
        self.progressBar.setVisible(False)
        # self.setMarkerBtn.setToolTip('please draw your marker at left bottom graphic window')

    def _splider_change(self):
        self.spinBox.setValue(self.splider.value())

    def _spinbox_change(self):
        self.splider.setValue(self.spinBox.value())

    def _binSEsizeSpin_change(self):
        value = self.binSEsizeSpin.value()
        if value % 2 == 0:
            self.binSEsizeSpin.setValue(value - 1)

    def _clear(self):
        # clear view
        scene = QGraphicsScene()
        self.viewRightTop.setScene(scene)
        self.viewLeftBottom.setScene(scene)
        self.viewRightBottom.setScene(scene)
        self.labelRT.setText('None')
        self.labelLB.setText('None')
        self.labelRB.setText('None')
        # clear data
        self.thres_image = None
        self.sub_skeletons = None
        self.binMarker = None
        self.grayMarker = None

    @staticmethod
    def print_image(image, view, label=None, title='image'):
        image = image.astype(np.uint8)
        # rename title
        if label is not None:
            label.setText(title)
        # build Scene
        scene = QGraphicsScene()
        view.setScene(scene)
        # transform image to Qt
        image_w, image_h = image.shape
        qt_img = QImage(image.data, image_h, image_w, image_h, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qt_img)
        # show image
        scene.addPixmap(pixmap)
        view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        view.show()

    @staticmethod
    def drawHist(histogram, view, label=None, title='Histogram'):
        # rename title
        if label is not None:
            label.setText(title)
        # build Scene
        scene = QGraphicsScene()
        view.setScene(scene)
        # draw histogram by plt
        fig, ax = plt.subplots(figsize=(view.width() / 80, view.height() / 80), dpi=80)
        ax.bar(range(len(histogram)), histogram)
        # ax.axis('off')

        # transfer canvas into pixmap
        canvas = FigureCanvas(fig)
        pixmap = QPixmap.fromImage(canvas.grab().toImage())
        # show image
        scene.addPixmap(pixmap)
        view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        view.show()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
                                     "Do you want to close all windows?",
                                     QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            sys.exit(0)
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = mainWindow()
    ui.show()
    # ui = Ui_MainWindow()
    # mainWindow = QtWidgets.QMainWindow()
    # ui.setupUi(mainWindow)
    # mainWindow.show()
    sys.exit(app.exec())
