from tkinter import Entry, Tk, Label
from numpy import array, dot, exp


# input and output result
def calculation(event):
    global ENTRY, LABEL, WEIGHTS, SIZE, MAX_VALUE

    content = ENTRY.get()

    def sigmoid(arg):
        return 1 / (1 + exp(-arg))

    def binContentIsCorrect():
        return all(map(lambda sym: sym in "01", content))

    def transfer():
        bin_content = content + '0' * (SIZE - len(content))
        bin_content = list(map(int, bin_content))
        return array([0] * (SIZE - len(bin_content)) + list(map(int, bin_content)))

    def output(inputs):
        LABEL['text'] = "Result: " + str(int(*sigmoid(dot(inputs, WEIGHTS))[0].round()))

    # content is a size-bit vector or is a usual number in [0; max_value]
    if binContentIsCorrect() and 0 < len(content) <= SIZE:
        output(transfer())
    # content is a bit vector, but length < size
    elif binContentIsCorrect() and len(content) > SIZE:
        LABEL['text'] = "Length error"
    # user entered nothing
    elif not content:
        LABEL['text'] = ""
    # letter in content
    elif not binContentIsCorrect():
        LABEL['text'] = "Letter in input"


# reading the weight of the neural network
FILE = open(r"D:\Projects\PredictionNumber\weights.txt", "r")
WEIGHTS = [[]]
for line in FILE:
    WEIGHTS[0].append([float(line[:-2])])
FILE.close()
SIZE = len(WEIGHTS[0])
MAX_VALUE = 2 ** SIZE - 1

# GUI
MASTER = Tk()
MASTER.title('')
MASTER.resizable(False, False)
ENTRY = Entry(MASTER, font=("Bookman Old Style", 36, "bold"), width=8)
LABEL = Label(MASTER, font=("Bookman Old Style", 36, "bold"), width=12)

ENTRY.grid(row=0, columnspan=2)
LABEL.grid(row=1, columnspan=2)
MASTER.bind("<KeyPress>", calculation)
MASTER.mainloop()
