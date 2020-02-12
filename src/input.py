from tkinter import Entry, Tk, Button, Label
from numpy import array, dot, exp


# ввод битового вектора и вывод результата
def calculation(event):
    global ENTRY, LABEL, WEIGHTS
    
    def sigmoid(arg):
        return 1 / (1 + exp(-arg))

    try:
        inputs = array([int(i) for i in ENTRY.get()])
        output = sigmoid(dot(inputs, WEIGHTS))
        LABEL['text'] = "Result: " + str(int(*output[0].round()))
    except ValueError:
        LABEL['text'] = "Wrong input"


# считываем веса нейросети
FILE = open("weights.txt", "r")
WEIGHTS = [[]]
for line in FILE:
    WEIGHTS[0].append([float(line[:-2])])
FILE.close()

# GUI
MASTER = Tk()
ENTRY = Entry(MASTER, font=("Comic Sans MS", 36, "bold"), width=7)
ENTER = Button(MASTER, text="Enter", fg="black", font=("Comic Sans MS", 24, "bold"))
LABEL = Label(MASTER, font=("Comic Sans MS", 36, "bold"), width=12)

ENTER.bind("<Button-1>", calculation)
ENTRY.grid(row=0, column=0)
ENTER.grid(row=0, column=1)
LABEL.grid(row=1, columnspan=2)
MASTER.mainloop()
