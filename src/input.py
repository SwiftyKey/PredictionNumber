import sys

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow

from global_params import *

# reading the weight of the neural network
WEIGHTS = read_weights()
MAX_VALUE = 2 ** SIZE - 1


class Window(QMainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()

        uic.loadUi("main.ui", self)

        self.button_prediction.clicked.connect(self.prediction)

    def prediction(self):
        def binContentIsCorrect():
            return all(map(lambda sym: sym in "01", content))

        def transfer():
            bin_content = content + '0' * (SIZE - len(content))
            bin_content = list(map(int, bin_content))
            return array([0] * (SIZE - len(bin_content)) + list(map(int, bin_content)))

        content = self.input.text()
        # content is a size-bit vector or is a usual number in [0; max_value]
        if binContentIsCorrect() and 0 < len(content) <= SIZE:
            self.output.setText(
                f"Result: {(str(int(*sigmoid(dot(transfer(), WEIGHTS))[0].round())))}")
        # content is a bit vector, but length < size
        elif binContentIsCorrect() and len(content) > SIZE:
            self.output.setText("Length error")
        # user entered nothing
        elif not content:
            self.output.setText("")
        # letter in content
        elif not binContentIsCorrect():
            self.output.setText("Letter in input")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Window()
    main.show()
    sys.exit(app.exec_())
