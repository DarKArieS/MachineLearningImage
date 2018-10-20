import json
import argparse
import numpy as np
import pandas as pd
import keras

parser =  argparse.ArgumentParser(description='Image trainer')
parser.add_argument('-f', '--configFile', dest="fname", type=str, default='config.json', required=False, 
					help="Json config file")
parser.add_argument('-v', '--verbose', dest="verbose", type=int, default=0, required=False, 
					help="verbose")

opt = parser.parse_args()

#Parameters
Params = json.load(open(opt.fname, 'r'))
Dataset_img_dirct = Params['Dataset']['img_dirct']
Dataset_mask_dirct = Params['Dataset']['mask_dirct']
Model_name = Params['Model']['name']
img_height = Params['Model']['img_height']
img_width = Params['Model']['img_width']
Num_epoch = Params['Num_epoch']
Steps_per_epoch = Params['Steps_per_epoch']
Batch_size = Params['Batch_size']
Other_cmd = ''

import modules.KerasModels as KerasModels
from modules.DataGenerator import *
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
					
TrainGen = trainGenerator(data_gen_args,Dataset_img_dirct,Dataset_mask_dirct,img_height,img_width, batch_size = Batch_size)

'''
import matplotlib.pyplot as plt
a = next(TrainGen)
for n in range (0,8):
    print(n)
    plt.imshow(a[0][n])
    plt.show()
    plt.imshow(a[1][n].reshape(img_height,img_width))
    plt.show()
    
cout = np.unique(a[1][0], return_counts=True)
cout = np.unique(a[1][0].astype(int), return_counts=True)
'''

model = KerasModels.unet(input_size=(img_height,img_width,3))
#from keras.utils import plot_model
#plot_model(model, to_file='myUNet.png', show_shapes = True)
'''
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
Fit = model.fit_generator(TrainGen,steps_per_epoch=Steps_per_epoch,epochs=Num_epoch,callbacks=[model_checkpoint], verbose=2)
#model.summary()

# save history
d_History = pd.DataFrame(Fit.history)
d_History.to_csv('history.txt',sep=' ',index=False)

print("finish!")
'''