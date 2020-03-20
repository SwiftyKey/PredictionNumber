from tkinter import Entry, Tk, Label
from numpy import array, dot, exp


# input bit vector and output result
def calculation(event):
    global ENTRY, LABEL, WEIGHTS

    size = len(WEIGHTS[0])
    max_value = 2 ** size - 1

    def sigmoid(arg):
        return 1 / (1 + exp(-arg))

    def binContentIsCorrect():
        return all(map(lambda sym: sym in "01", content))

    def transfer():
        bin_content = list(bin(int(content)))[2:]
        return array([0] * (size - len(bin_content)) + list(map(lambda el: int(el), bin_content)))

    content = ENTRY.get()

    try:
        # content is a 6-bit vector
        if binContentIsCorrect() and len(content) == size:
            inputs = array(list(map(lambda el: int(el), content)))
            output = sigmoid(dot(inputs, WEIGHTS))
            LABEL['text'] = "Result: " + str(int(*output[0].round()))
        # content is a usual number in [0; 63]
        elif 0 <= int(content) <= max_value:
            inputs = transfer()
            output = sigmoid(dot(inputs, WEIGHTS))
            LABEL['text'] = "Result: " + str(int(*output[0].round()))
        # content is a bit vector, but length < 6
        elif binContentIsCorrect() and len(content) != size:
            LABEL['text'] = "Length error"
        # content is a number, but more than 63
        elif not binContentIsCorrect():
            LABEL['text'] = f"Number > {max_value}"
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

# GUI
MASTER = Tk()
ENTRY = Entry(MASTER, font=("Bookman Old Style", 36, "bold"), width=8)
LABEL = Label(MASTER, font=("Bookman Old Style", 36, "bold"), width=12)

ENTRY.grid(row=0, columnspan=2)
LABEL.grid(row=1, columnspan=2)
MASTER.bind("<KeyPress>", calculation)
MASTER.mainloop()
