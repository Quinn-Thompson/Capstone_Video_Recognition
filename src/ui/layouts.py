from PyQt5 import QtWidgets


class ClassificationLayout(QtWidgets.QGridLayout):
    def __init__(self):
        super(ClassificationLayout, self).__init__()

        self.labelClassHead = QtWidgets.QLabel()
        self.labelClassHead.setStyleSheet("background-color: rgba(65, 65, 65, 255); color: rgba(200, 200, 200, 255);")
        self.labelClassHead.setObjectName("EUlabelClassHead")
        self.labelClassHead.setText("Classification")

        self.labelPredict10 = QtWidgets.QLabel()
        
        self.labelPredict10.setObjectName("EUlabelPredict10")
        self.labelPredict10.setStyleSheet("background-color: rgba(65, 65, 65, 255) ; color: rgba(200, 200, 200, 255);")
        self.labelPredict10.setText("1:")
        self.labelPredict11 = QtWidgets.QLabel()
        self.labelPredict11.setStyleSheet("background-color: rgba(65, 65, 65, 255); color: rgba(200, 200, 200, 255);")
        self.labelPredict11.setObjectName("EUlabelPredict11")
        self.labelPredict11.setText("A")

        self.labelPredict20 = QtWidgets.QLabel()
        self.labelPredict20.setStyleSheet("background-color: rgba(65, 65, 65, 255); color: rgba(200, 200, 200, 255);")
        self.labelPredict20.setObjectName("EUlabelPredict20")
        self.labelPredict20.setText("2:")
        self.labelPredict21 = QtWidgets.QLabel()
        self.labelPredict21.setStyleSheet("background-color: rgba(65, 65, 65, 255); color: rgba(200, 200, 200, 255);")
        self.labelPredict21.setObjectName("EUlabelPredict21")
        self.labelPredict21.setText("B")

        self.labelPredict30 = QtWidgets.QLabel()
        self.labelPredict30.setStyleSheet("background-color: rgba(65, 65, 65, 255); color: rgba(200, 200, 200, 255);")
        self.labelPredict30.setObjectName("EUlabelPredict30")
        self.labelPredict30.setText("3:")
        self.labelPredict31 = QtWidgets.QLabel()
        self.labelPredict31.setStyleSheet("background-color: rgba(65, 65, 65, 255); color: rgba(200, 200, 200, 255);")
        self.labelPredict31.setObjectName("EUlabelPredict31")
        self.labelPredict31.setText("C")

        self.addWidget(self.labelClassHead, 0, 0, 1, 2)
        self.addWidget(self.labelPredict10, 1, 0)
        self.addWidget(self.labelPredict11, 1, 1)
        self.addWidget(self.labelPredict20, 2, 0)
        self.addWidget(self.labelPredict21, 2, 1)
        self.addWidget(self.labelPredict30, 3, 0)
        self.addWidget(self.labelPredict31, 3, 1)

    def updatePredictions(self, predictions):
        self.labelPredict11.setText(predictions[0])
        self.labelPredict21.setText(predictions[1])
        self.labelPredict31.setText(predictions[2])
