from PyQt5 import QtWidgets


class ClassificationLayout(QtWidgets.QGridLayout):
    def __init__(self):
        super(ClassificationLayout, self).__init__()

        self.labelClassHead = QtWidgets.QLabel()
        self.labelClassHead.setObjectName("EUlabelClassHead")
        self.labelClassHead.setText("Classification")

        self.labelPredict1 = QtWidgets.QLabel()
        self.labelPredict1.setObjectName("EUlabelPredict10")
        self.labelPredict1.setText("1:")

        self.labelPredict2 = QtWidgets.QLabel()
        self.labelPredict2.setObjectName("EUlabelPredict20")
        self.labelPredict2.setText("2:")

        self.labelPredict3 = QtWidgets.QLabel()
        self.labelPredict3.setObjectName("EUlabelPredict30")
        self.labelPredict3.setText("3:")

        self.addWidget(self.labelClassHead, 0, 0, 1, 2)
        self.addWidget(self.labelPredict1, 1, 0)
        self.addWidget(self.labelPredict2, 2, 0)
        self.addWidget(self.labelPredict3, 3, 0)

    def updatePredictions(self, predictions):
        self.labelPredict1.setText(f"1. {predictions[0]}")
        self.labelPredict2.setText(f"2. {predictions[1]}")
        self.labelPredict3.setText(f"3. {predictions[2]}")
