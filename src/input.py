from tkinter import Entry, Tk, Label
from numpy import array, dot, exp


# ввод битового вектора и вывод результата
def calculation(event):
    global ENTRY, LABEL, WEIGHTS

    def sigmoid(arg):
        return 1 / (1 + exp(-arg))

    def contentIsCorrect():
        return all(map(lambda sym: sym in "01", content))

    content = ENTRY.get()

    if contentIsCorrect() and len(content) == 6:
        inputs = array([int(i) for i in content])
        output = sigmoid(dot(inputs, WEIGHTS))
        LABEL['text'] = "Result: " + str(int(*output[0].round()))
    if contentIsCorrect() and len(content) != 6:
        LABEL['text'] = "Length error"
    if not contentIsCorrect():
        LABEL['text'] = "Letter in input"


# считываем веса нейросети
FILE = open(r"D:\PredictionNumber\weights.txt", "r")
WEIGHTS = [[]]
for line in FILE:
    WEIGHTS[0].append([float(line[:-2])])
FILE.close()

# GUI
MASTER = Tk()
ENTRY = Entry(MASTER, font=("TimesNewRoman", 36, "bold"), width=7)
LABEL = Label(MASTER, font=("TimesNewRoman", 36, "bold"), width=12)

ENTRY.grid(row=0, columnspan=2)
LABEL.grid(row=1, columnspan=2)
MASTER.bind("<KeyPress>", calculation)
MASTER.mainloop()
