from Tkinter import *
import tkMessageBox
from ttk import *
import inspect
import os

#Version 21.06.2017
# Just for testing, gives back variable in a pop-up window
def infoBox():
    QuietMode1 = QuietMode.get()
    tkMessageBox.showinfo("Variable", QuietMode1)
    return


# Get current path

def getPath():
    path = inspect.getfile((inspect.currentframe()))
    return path

#
# run pacmany.py with given parameters

def runPacman():
    train = ""
    try:
        if rounds.get() >= 0:
            train = "-x " + str(rounds.get())
        else:
            return
    except ValueError:
        tkMessageBox.showinfo("Warning", "Please specify a number of training rounds or put 0 as a default")
        return

    try:
        totalgames = rounds.get() + evalrounds.get()
        if evalrounds.get() > 0:
            games = "-n " + str(totalgames)
        elif totalgames == 0:
            games = "-n 1"
        else:
            return
    except ValueError:
        tkMessageBox.showinfo("Warning", "Please specify a number of evaluation rounds or put 0 as a default")
        return

    if (str(QuietMode.get()) != ""):
        quiet = str(QuietMode.get())
    else:
        quiet = ""

    # Get Layout
    grids = "-l " + str(layout.get())

    # Get Agent
    agentchoice = "-p " + str(agent.get())

    #Get Extractor
    if str(extractor.get()) == "No Extractor":
        extractorchoice = ""
    else:
        extractorchoice = "-a " + "extractor=" + str(extractor.get())

    running = "python pacman.py " + extractorchoice + " " + agentchoice + " " + quiet + " " + train + " " + grids +\
              " " + games + " "
    print(running)
    try:
        os.system(running)
    except:
       # tkMessageBox.showinfo("We are here", "Training not possible with Keyboard agent")
        if "KeyboardAgent" in str(layout.get()) and rounds.get() != 0:
            tkMessageBox.showinfo("Wrong Agent Type", "Training not possible with Keyboard agent")
    return

    return


# define TKinterface

app = Tk()
app.title("Pac-Man Game")
app.geometry('650x400+200+200')

# Do you want a pacman userface?
QuietMode = StringVar(None)

OutputButton = Checkbutton(app, text="Quiet Mode", variable=QuietMode, onvalue=" -q", offvalue="")

# Text
textlayout = StringVar()
textlayout.set("Choose Layout:")
LayoutText = Label(app, textvariable=textlayout)

# Layout

layouts = ('capsuleClassic', 'contestClassic', 'mediumClassic', 'mediumGrid', 'minimaxClassic', 'openClassic', \
           'originalClassic', 'smallClassic', 'smallGrid', 'testClassic', 'trappedClassic', 'trickyClassic')

layout = Combobox(app, values=layouts)
layout.set("mediumClassic")

# Heading Training Rounds
roundsHeading = StringVar()
roundsHeading.set("Choose Number of Training Rounds (integer):")
RoundsLabel = Label(app, textvariable=roundsHeading)

# Training rounds

rounds = IntVar()
rounds.set(0)
training = Entry(app, textvariable=rounds, width=10)
training.delete(0, END)

# Heading Evaluation Rounds
evalroundsHeading = StringVar()
evalroundsHeading.set("Choose Number of Evaluation Rounds (integer):")
evalRoundsLabel = Label(app, textvariable=evalroundsHeading)

# Evaluation rounds

evalrounds = IntVar()
evalrounds.set(0)
evaluation = Entry(app, textvariable=evalrounds, width=10)
evaluation.delete(0, END)

# Heading Agents
agentHeading = StringVar()
agentHeading.set("Choose the Agent to run the game with (default is keyboard):")
AgentLabel = Label(app, textvariable=agentHeading)

# Agent Choice
agents = ('KeyboardAgent', 'ApproximateQAgent', 'PacmanQAgent', 'DQLAgent')

agent = Combobox(app, values=agents)
agent.set("KeyboardAgent")

# Heading Extractor
extrHeading = StringVar()
extrHeading.set("Choose the Extractor to run the game with (default is IdentityExtractor):")
ExtrLabel = Label(app, textvariable=extrHeading)

# Agent Choice
extractors = ('IdentityExtractor', 'CoordinateExtractor', 'BetterExtractor', 'BestExtractor', 'No Extractor')

extractor = Combobox(app, values=extractors)
extractor.set("IdentityExtractor")


# Start the App
StartButton = Button(app, text="Start Pac-Man", command=runPacman)

OutputButton.pack()
LayoutText.pack()
layout.pack()
RoundsLabel.pack()
training.pack()
evalRoundsLabel.pack()
evaluation.pack()
AgentLabel.pack()
agent.pack()
ExtrLabel.pack()
extractor.pack()
StartButton.pack(side='bottom')
app.mainloop()
