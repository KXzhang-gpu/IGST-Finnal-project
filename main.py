# -*- coding: UTF-8 -*-
import sys

from PyQt5 import QtWidgets

from hello import Ui_MainWindow

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    mainWindow = QtWidgets.QMainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec())
