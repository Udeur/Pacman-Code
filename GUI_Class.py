from Tkinter import *
from ttk import *
from PIL import Image, ImageTk
import tkMessageBox
import inspect
import os
import random


#Version 21.06.2017


class MainWindow:
    counter = 0


    def __init__(self):
        global photo
        self.root = Tk()
        self.root.title("Pac-Man Game")
        self.list = []
        self.agent = StringVar()
        self.extractor = StringVar()
        self.layout = StringVar()
        self.training = IntVar()
        self.evaluation = IntVar()
        self.quiet = IntVar()

        name = "/GUI_pic/pacman_keyboard.png"
        path_pic = os.path.dirname(os.path.realpath(__file__))
        photo = ImageTk.PhotoImage(Image.open(path_pic + name).resize((160, 100), Image.ANTIALIAS))
        self.lbl_pic = Label(self.root, image=photo)
        self.lbl_pic.grid(row=4, column=3, columnspan=6, rowspan=4)

    def configurestyles(self):
        bg = 'black'
        fg = 'green'

        self.root.configure(background=bg)
        style = Style()
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TCombobox", background=fg, foreground=fg, fieldbackground=bg, borderthickness=0)
        style.configure("TCheckbutton", background=bg, foreground=fg)
        style.configure("TEntry", background=bg, foreground=fg, borderwidth=1, fieldbackground=bg, borderthickness=0)
        style.configure("TButton", relief='raised', background='#ece74d', foreground=bg)
        return None

    def createElements(self):
        self.createLabels()
        self.createCombo()
        self.createEntrys()
        self.createCheckbuttons()
        return None

    def createLabels(self):
        lbl_agent = Label(self.root, text="Agent:")
        lbl_extract = Label(self.root, text="Extractor:")
        lbl_layout = Label(self.root, text="Choose Layout:")
        lbl_training = Label(self.root, text="Training Rounds(int):")
        lbl_evaluation = Label(self.root, text="Evaluation Rounds(int):")
        lbl_quiet = Label(self.root, text="Quiet Mode:")


        self.list.append(lbl_agent)
        self.list.append(lbl_extract)
        self.list.append(lbl_layout)
        self.list.append(lbl_training)
        self.list.append(lbl_evaluation)
        self.list.append(lbl_quiet)

        return None

    def createCombo(self):
    #Agents
        agents = Combobox(self.root, textvariable=self.agent)
        agents['values'] = ('KeyboardAgent', 'ApproximateQAgent', 'PacmanQAgent', 'DQLAgent')
        agents.set('KeyboardAgent')
        agents.bind("<<ComboboxSelected>>", self.agentchanged)
    #Extractors
        extractors = Combobox(self.root, values=self.extractor)
        extractors['values'] = ('IdentityExtractor', 'CoordinateExtractor', 'BetterExtractor', 'BestExtractor', 'No Extractor')
        self.extractor.set("IdentityExtractor")
    #Layouts
        layouts = Combobox(self.root, values=self.layout)
        layouts['values'] = ('capsuleClassic', 'contestClassic', 'mediumClassic', 'mediumGrid', 'minimaxClassic', 'openClassic', \
                   'originalClassic', 'smallClassic', 'smallGrid', 'testClassic', 'trappedClassic', 'trickyClassic')
        self.layout.set("mediumClassic")

        self.list[0] = (self.list[0], agents)
        self.list[1] = (self.list[1], extractors)
        self.list[2] = (self.list[2], layouts)

        return None

    def createEntrys(self):
        txt_training = Entry(self.root, textvariable=self.training, width=10)
        self.training.set(0)
       # txt_training.delete(0, END)
        txt_evaluation = Entry(self.root, textvariable=self.evaluation, width=10)
        self.evaluation.set(0)
        #evaluation.delete(0, END)

        self.list[3] = (self.list[3], txt_training)
        self.list[4] = (self.list[4], txt_evaluation)

        return None

    def createCheckbuttons(self):
        cb_quiet = Checkbutton(self.root, variable=self.quiet, onvalue=" -q", offvalue="")
        self.list[5] = (self.list[5], cb_quiet)
        return None

    def createButtons(self, index):
        StartButton = Button(self.root, text="Start Pac-Man")
        StartButton.grid(row=index + 1, column=8, padx=20, pady=20)
        return None

    def orderElements(self):
        i = 0
        padding = 4
        for item in self.list:
            if i in [0, 4]:
                Label(self.root, text='').grid(row=i, column=0, ipady=padding * 2, padx=20)
                i += 1
            item[0].grid(row=i, column=1, sticky=E, pady=padding, padx=1)
            item[1].grid(row=i, column=2, sticky=W, pady=padding)
            i += 1
        self.createButtons(i)
        return None

    def loadimage(self):
        self.setPicture(0)
        return None

    def setPicture(self, index=0):
        name = "/GUI_pic/pacman_neural.png"
        if index == 0:
            name = "/GUI_pic/pacman_keyboard.png"
        path_pic = os.path.dirname(os.path.realpath(__file__))
        self.photo = ImageTk.PhotoImage(Image.open(path_pic + name).resize((160, 100), Image.ANTIALIAS))
        self.lbl_pic.configure(image=self.photo)
        self.lbl_pic.image = self.photo


#Events
    def agentchanged(self, event):
        if self.agent.get() == 'KeyboardAgent':
            self.setPicture(0)
        else:
            self.setPicture(1)

if __name__ == '__main__':
    app = MainWindow()
    app.configurestyles()
    app.createElements()
    app.orderElements()
    app.root.mainloop()


