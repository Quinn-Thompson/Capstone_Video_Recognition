import sys

from PyQt5.QtWidgets import QApplication

from src.ui.ui import MLGestureRecognition

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = MLGestureRecognition()
    ui.mainWindow.show()
    sys.exit(app.exec_())
