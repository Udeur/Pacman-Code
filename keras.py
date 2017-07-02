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
        
    def trainFromBatch():
        
    def Save(self):
