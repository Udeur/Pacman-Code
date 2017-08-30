from Tkinter import *
from ttk import *
from PIL import Image, ImageTk
import tkMessageBox
import os
import webbrowser
import pacman

#Version 30/08/2017



class MainWindow:
    counter = 0

    def __init__(self):
        global photo
        self.root = Tk()
        self.root.title("Pac-Man Game")
        self.list = []
        self.agent = StringVar()
        self.layout = StringVar()
        self.gamecount = IntVar()


        self.tup_agents = ('KeyboardAgent', 'SearchAgent', 'ApproximateQAgent', 'DeepQAgent SE (TF)', 'DeepQAgent EE (TF)', 'DeepQAgent 1HL (PyT)', 'DeepQAgent 3HL (PyT)')
        self.tup_layout = ('capsuleClassic', 'contestClassic', 'mediumClassic', 'mediumGrid', 'minimaxClassic', 'openClassic', \
                   'originalClassic', 'smallClassic', 'smallGrid', 'testClassic', 'trappedClassic', 'trickyClassic')

        name = "/GUI_pic/pacman_keyboard.png"
        path_pic = os.path.dirname(os.path.realpath(__file__))
        photo = ImageTk.PhotoImage(Image.open(path_pic + name).resize((160, 100), Image.ANTIALIAS))
        self.lbl_pic = Label(self.root, image=photo)
        self.lbl_pic.grid(row=1, column=4, columnspan=6, rowspan=4)

    def configurestyles(self):
        bg = 'black'
        fg = 'green'

        self.root.configure(background=bg)
        style = Style()
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TCombobox", background=fg, foreground=fg, fieldbackground=bg, borderthickness=0)
        style.configure("TEntry", background=bg, foreground=fg, borderwidth=1, fieldbackground=bg, borderthickness=0)
        style.configure("TButton", relief='raised', background='#ece74d', foreground=bg)
        style.configure("TMenu", background=bg, foreground=fg)
        return None

    def createElements(self):
        self.createLabels()
        self.createCombo()
        self.createEntrys()
        return None

    def createLabels(self):
        lbl_agent = Label(self.root, text="Agent:")
        lbl_layout = Label(self.root, text="Choose Layout:")
        lbl_gamecount = Label(self.root, text="Rounds to Play:")


        self.list.append(lbl_agent)
        self.list.append(lbl_layout)
        self.list.append(lbl_gamecount)


        return None

    def createCombo(self):
    #Agents
        agents = Combobox(self.root, textvariable=self.agent)
        agents['values'] = self.tup_agents
        agents.set('KeyboardAgent')
        agents.bind("<<ComboboxSelected>>", self.agentchanged)

    #Layouts -------------------------------------------------------------------------------
        layouts = Combobox(self.root, textvariable=self.layout)
        layouts['values'] = self.tup_layout
        self.layout.set('mediumClassic')

        self.list[0] = (self.list[0], agents)
        self.list[1] = (self.list[1], layouts)

        return None

    def createEntrys(self):
        txt_gamecount = Entry(self.root, textvariable=self.gamecount, width=10)
        self.gamecount.set(1)
        self.list[2] = (self.list[2], txt_gamecount)
        return None

    def createButtons(self, index):
        StartButton = Button(self.root, text="Start Pac-Man", command=self.clickStart)
        StartButton.grid(row=index + 1, column=8, padx=20, pady=20)
        return None

    def createMenu(self):
        menu = Menu(self.root)
        self.root.config(menu=menu)
        menu.add_command(label="Credits", command=self.showCredits)
        menu.config(bg='black', fg='green')


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
        self.createMenu()
        return None

    def loadimage(self):
        self.setPicture(0)
        return None

    def setPicture(self, index=0):
        name =  "/GUI_pic/pacman_"
        name += ['keyboard', 'magnifier', 'q', 'neural'][index if index < 4 else 3] + '.png'
        path_pic = os.path.dirname(os.path.realpath(__file__))
        self.photo = ImageTk.PhotoImage(Image.open(path_pic + name).resize((160, 100), Image.ANTIALIAS))
        self.lbl_pic.configure(image=self.photo)
        self.lbl_pic.image = self.photo

    def intputsAreValid(self):
        str = ''
        if not self.agent.get() in self.tup_agents: str += 'Selected agent is not valid \n'
        if not self.layout.get() in self.tup_layout: str += 'Selected layout is not valid \n'
        if not self.gamecount.get() > 0: str += 'You must play at least one time'
        if len(str) == 0:
            return True
        else:
            tkMessageBox.showinfo("Invalid Parameters", str)
            return False

#Events----------------------------------------------------------------------------------
    #Update of pacman pics
    def agentchanged(self, event):
        self.setPicture(self.tup_agents.index(self.agent.get()))

    def clickStart(self):
        agent = self.agent.get()
        #check correctness of inputs before proceeding
        if self.intputsAreValid():
            list = []
            list.append("-p")
            list.append(agent)
            list.append("-l")
            list.append(self.layout.get())
            list.append("-n")
            list.append(str(self.gamecount.get()))

            if agent == 'DeepQAgent - SE':
                list[1] = "deepqlearningAgents"
                list.append("-a")
                list.append("extractor=SimpleExtractor")
            elif agent == 'DeepQAgent - EE':
                list[1] = "deepqlearningAgents"
                list.append("-a")
                list.append("extractor=BestExtractor")
            elif agent == 'DeepQAgent 1HL (PyT)':
                list[1] = "TorchAgent1NN"
            elif agent == 'DeepQAgent 3HL (PyT)':
                list[1] = "TorchAgent3NN"
            elif agent == 'SearchAgent':
                list[1] = "ClosestDotSearchAgent"
               # list.append('-a')
               # list.append('fn=bfs')
            elif agent == 'ApproximateQAgent':
                list.append("-a")
                list.append("extractor=BetterExtractor")

            list.append('-g')
            list.append('DirectionalGhost')

            args = pacman.readCommand(list)
            pacman.runGames(**args)

    def showCredits(self):
        credit = Credits()
        credit.show()

#------------------------------------------------------------------
class Credits:

    def __init__(self):
        global img
        self.root = Tk()
        self.root.title('Credits')
        self.configureStyle()
        self.createLabels()
        self.orderLabels()

    def createLabels(self):
        self.lbl_created = Label(self.root, text='Created by:\n\tFleiner, Christian  \n\tGoergen, Konstantin  \n\tJohn, Felix '
                                                 '\n\tMichalczyk, Sven \n\tPickl, Max ')
        self.lbl_created.configure(background='black', foreground='green')

        self.lbl_opening = Label(self.root, text='Thanks to all Contributors\nof knowledge and source code')
        self.lbl_opening.configure(background='black', foreground='green')

        self.lbl_bk = HyperlinkLabel(self.root, "UC Berkeley CS188 Intro to AI")
        self.lbl_bk.setLink(r"http://ai.berkeley.edu/home.html")

        self.lbl_aifb = HyperlinkLabel(self.root, "Institut AIFB des KIT")
        self.lbl_aifb.setLink("http://www.aifb.kit.edu/web/Hauptseite")

    def configureStyle(self):
        self.root.configure(background='black')
        style = Style()
        style.configure("TLabel", background='black', foreground='green')

    def orderLabels(self):
        self.lbl_created.grid(row=1, padx=5, pady = 10)
        self.lbl_opening.grid(row=2, padx=5, pady = 10)
        self.lbl_bk.grid(row=3, padx=5)
        self.lbl_aifb.grid(row=4, padx=5)

    def show(self):
        self.root.mainloop()

#Klasse um Label mit Hyperlink zu koppeln
class HyperlinkLabel(Label):
    link = ''

    def __init__(self, root, text):
        Label.__init__(self, master=root, text=text, cursor='hand2', background='black', foreground='blue')
        self.bind("<Button-1>", self.openLink)

    def setLink(self, str):
        self.link = str

    def openLink(self, event):
        webbrowser.open_new(self.link)

if __name__ == '__main__':
    app = MainWindow()
    app.configurestyles()
    app.createElements()
    app.orderElements()
    app.root.mainloop()


