import json
import argparse
import numpy as np
import pandas as pd
import keras

import modules.LoadData as LoadData
import modules.KerasModels as KerasModels
import modules.PreProcessing as PreProcessing

parser =  argparse.ArgumentParser(description='Image trainer')
parser.add_argument('-f', '--configFile', dest="fname", type=str, default='config.json', required=False, 
					help="Json config file")
parser.add_argument('-v', '--verbose', dest="verbose", type=int, default=0, required=False, 
					help="verbose")

opt = parser.parse_args()

#Parameters
Params = json.load(open(opt.fname, 'r'))
Num_epoch = Params['Num_epoch']
Batch_size = Params['Batch_size']
Model_name = Params['Model']['Name']
Model_flatten = Params['Model']['Flatten']
Other_cmd = ''


#===================================================================================
# Load Data
#===================================================================================
(x_train, y_train), (x_test, y_test) = LoadData.Load_dataset(Params['Dataset'])

y_test_org = y_test.copy()
#show one plot
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0])
# plt.show()

# check colorful or not
#print(x_train.shape)
#print(np.shape(x_train.shape))

# scaling
x_train = PreProcessing.PrePro_1(x_train)
x_test = PreProcessing.PrePro_1(x_test)
# (x_train,x_test) = PreProcessing.color_preprocessing_CIFAR10(x_train,x_test)

# one-hot encoding
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

#===================================================================================
# Build Model
#===================================================================================
estimator = []
Input_dim = x_train.shape[1:4]
Output_dim = y_train.shape[1]

#flatten (just for simple DNN)
if Model_flatten == True:
	Input_dim = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
	x_train = x_train.reshape(x_train.shape[0],Input_dim)
	x_test = x_test.reshape(x_test.shape[0],Input_dim)

exec('estimator = KerasModels.'+ Model_name +'('+str(Input_dim)+','+str(Output_dim)+' )')
# estimator = KerasModels.Simple_CNN(Input_dim,Output_dim)
print(estimator.summary())

#===================================================================================
# set callback
# 1. Different learning rate schedule may get different training/testing accuracy!
#===================================================================================
# def scheduler(epoch):
    # if epoch < 100:
        # return 0.01
    # if epoch < 150:
        # return 0.005
    # return 0.001
# from keras.callbacks import LearningRateScheduler, TensorBoard
# tb_cb = TensorBoard(log_dir='./lenet_dp_da_wd', histogram_freq=0)
# change_lr = LearningRateScheduler(scheduler)
# cbks = [change_lr,tb_cb]

#===================================================================================
# Image preprocessing
# package candidates: keras, imgaug
# 1. Data augmentation
# 2. openCV-python
#===================================================================================
from keras.preprocessing.image import ImageDataGenerator
print('Using real-time data augmentation.')
datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)
datagen.fit(x_train)

#===================================================================================
# Fitting
#===================================================================================
Fit = estimator.fit(x_train,y_train,epochs = Num_epoch, batch_size = Batch_size, verbose=1)
# Fit = estimator.fit_generator(datagen.flow(x_train, y_train,batch_size=Batch_size),epochs=Num_epoch)#,steps_per_epoch=391)#,callbacks=cbks,validation_data=(x_test, y_test))

estimator.save(Model_name+'_ep'+str(Num_epoch)+'_batch'+str(Batch_size) + Other_cmd + '.h5')

# save history
d_History = pd.DataFrame(Fit.history)
d_History.to_csv(Model_name+'_ep'+str(Num_epoch)+'_batch'+str(Batch_size) + Other_cmd + '.txt',sep=' ',index=False)

#===================================================================================
# testing
#===================================================================================

train_score = estimator.evaluate(x_train,y_train, verbose=1)
print('Train Total Loss:', train_score[0])
print('Train Accuracy:', train_score[1])

test_score = estimator.evaluate(x_test,y_test, verbose=1)
print('Test Total Loss:', test_score[0])
print('Test Accuracy:', test_score[1])

# Confusion Matrix (very simple)
print('confuse')
import pandas as pd 
predictions = estimator.predict_classes(x_test)
pd.crosstab(y_test_org.T[0], predictions, rownames=['True'], colnames=['Prediction'])

