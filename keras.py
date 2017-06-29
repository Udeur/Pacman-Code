import numpy as np
import game
from game import *
import pacman
from util import *
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv2D
from keras import optimizers
from keras import losses


class NN():

    def dataprep(self, state):
        # Matrix-Generierung:
        # Pacman_Position
        width, heigth = state.data.layout.width,state.data.layout.height
        matPacPosition = np.zeros((width, heigth),dtype= int)
        x,y = state.getPacmanPostion()
        if y < heigth/2:
            matPacPosition[x, y + heigth - 1] = 1
        else:
            matPacPosition[x, y - heigth + 1] = 1

        #Walls Position
        matWallsPosition = np.zeros((width, heigth),dtype= int)
        walls = state.getWalls()
        for i in range(width):
            for j in range(heigth):
                if walls[i,j]:
                    if j < heigth/2:
                        matWallsPosition[i, j + heigth - 1]
                    else:
                        matWallsPosition[i, j - heigth + 1]

        #Ghost Position

        return


    def _init_(self, state):
         width, height = state.data.layout.width, state.data.layout.height
         self.action.size = 4;  #5 
         self.model = self._build_model()
         self.epsilon = ""
        
    def _build_model(self):   
        
        model = Sequential()
        model.add(Conv2D(16, kernel_size = (3,3),input_shape=(width, height, 6), strides = 1, activation='relu'))
        model.add(Conv2D(32, kernel_size = (3,3), strides = 1, activation='relu'))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(4, activation ='linear'))
        model.compile(loss='mean_squared_error', optimizer='RMSprop',metrics=['accuracy'])
        
      
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
     
    #Beispielcode   
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    def replay(self, batch_size):
