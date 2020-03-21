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
        bin_content = list(bin(int(content)))[2:] if not binContentIsCorrect() else content
        return array([0] * (SIZE - len(bin_content)) + list(map(int, bin_content)))

    def output(inputs):
        LABEL['text'] = "Result: " + str(int(*sigmoid(dot(inputs, WEIGHTS))[0].round()))

    try:
        # content is a size-bit vector or is a usual number in [0; max_value]
        if binContentIsCorrect() and 0 < len(content) <= SIZE or 0 <= int(content) <= MAX_VALUE:
            output(transfer())
        # content is a bit vector, but length < size
        elif binContentIsCorrect() and len(content) > SIZE:
            LABEL['text'] = "Length error"
        # content is a number, but more than max_value
        elif not binContentIsCorrect():
            LABEL['text'] = f"Number > {MAX_VALUE}"
    except ValueError:
        # user entered nothing
        if not content:
            LABEL['text'] = ""
        # letter in content
        else:
            LABEL['text'] = "Letter in input"


# reading the weight of the neural network
FILE = open(r"\PredictionNumber\weights.txt", "r")
WEIGHTS = [[]]
for line in FILE:
    WEIGHTS[0].append([float(line[:-2])])
FILE.close()
SIZE = len(WEIGHTS[0])
MAX_VALUE = 2 ** SIZE - 1

# GUI
MASTER = Tk()
ENTRY = Entry(MASTER, font=("Bookman Old Style", 36, "bold"), width=8)
LABEL = Label(MASTER, font=("Bookman Old Style", 36, "bold"), width=12)

ENTRY.grid(row=0, columnspan=2)
LABEL.grid(row=1, columnspan=2)
MASTER.bind("<KeyPress>", calculation)
MASTER.mainloop()
