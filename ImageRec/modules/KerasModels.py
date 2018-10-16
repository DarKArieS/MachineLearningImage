from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, LeakyReLU, ReLU, Input, Add
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.regularizers import l2
#from keras import regularizers

#classifier

def Simple_DNN(in_nD, out_nD):
	model = Sequential()
	model.add(Dense(2000, input_dim=in_nD, kernel_initializer = 'glorot_uniform'))
	model.add(Activation('sigmoid'))
	model.add(Dense(1000))
	model.add(Activation('sigmoid'))
	model.add(Dense(out_nD, activation='softmax'))
	
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return model

def Simple_CNN(input_shape, out_nD):
	model = Sequential()
	# nParameters = 3 x 3 (x 3) x 32 + 32
	model.add(Conv2D(filters = 32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(filters = 64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(out_nD, activation='softmax'))
	
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return model
	
def LeNet(input_shape, out_nD):
	weight_decay  = 0.0001
	
	model = Sequential()
	model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), input_shape=input_shape))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Flatten())
	model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
	model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
	model.add(Dense(out_nD, activation = 'softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
	sgd = SGD(lr=.1, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model
	
def AlexNet(input_shape, out_nD):
	model = Sequential()
	model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(227,227,3),padding='valid',activation='relu',kernel_initializer='uniform'))
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
	model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
	model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
	model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
	model.add(Flatten())
	model.add(Dense(4096,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000,activation='softmax'))
	model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

	return model
	
def NiN(input_shape, out_nD):

	pass