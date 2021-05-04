import sys

from PyQt5.QtWidgets import QApplication
from src.network_quinn.stream_combo import run_stream
from src.ui.ui import MLGestureRecognition

if __name__ == "__main__":
    run_stream()
    app = QApplication(sys.argv)
    ui = MLGestureRecognition()
    ui.mainWindow.show()
    sys.exit(app.exec_())
