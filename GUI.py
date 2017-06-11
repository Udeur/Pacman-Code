from Tkinter import *
import tkMessageBox
from ttk import *


def infoBox():
    QuietMode1=QuietMode.get()
    tkMessageBox.showinfo("Variable", QuietMode1)
    return

app = Tk()
app.title("Pac-Man Game")
app.geometry('650x400+200+200')

#Do you want a pacman userface?
QuietMode = StringVar(None)

OutputButton = Checkbutton(app, text="Quiet Mode", variable=QuietMode, onvalue="-q", offvalue="")

#Text
textlayout= StringVar()
textlayout.set("Choose Layout:")
LayoutText = Label(app, textvariable=textlayout)

#Layout

layouts = ('Classic', 'Small', 'Large')
layout = Combobox(app, values=layouts)


#Start the App
StartButton = Button(app, text="Start Pac-Man", command=infoBox)

OutputButton.pack()
LayoutText.pack()
layout.pack()
StartButton.pack(side= 'bottom')
app.mainloop()

