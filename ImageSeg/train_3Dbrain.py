import json
import argparse
import numpy as np
import pandas as pd
import keras

parser =  argparse.ArgumentParser(description='Image trainer')
parser.add_argument('-f', '--configFile', dest="fname", type=str, default='config_3Dbrain.json', required=False, 
					help="Json config file")
parser.add_argument('-v', '--verbose', dest="verbose", type=int, default=0, required=False, 
					help="verbose")

opt = parser.parse_args()

#Parameters
Params = json.load(open(opt.fname, 'r'))
Dataset_img_dirct = Params['Dataset']['img_dirct']
Dataset_mask_dirct = Params['Dataset']['mask_dirct']
Dataset_vali_split = Params['Dataset']['vali_split']
Dataset_test_split = Params['Dataset']['test_split']
Data_Aug = Params['Data_Aug']['train_used']
Model_name = Params['Model']['name']
img_height = Params['Model']['img_height']
img_width = Params['Model']['img_width']
img_depth = Params['Model']['img_depth']
img_channels = Params['Model']['img_channels']
Num_epoch = Params['Num_epoch']
Steps_per_epoch = Params['Steps_per_epoch']
Batch_size = Params['Batch_size']

Other_cmd = ''

import modules.KerasModels as KerasModels
from modules.DataGenerator import *

Gen = NIFIT_Generator(Dataset_img_dirct,Dataset_mask_dirct,img_size=(img_height,img_width,img_depth),validation_split=0.1,test_split=0.1)
TrainGen = Gen.GetGenerator(Batch_size,subset="train")
ValiGen = Gen.GetGenerator(Batch_size,subset="vali")

model = KerasModels.unet_3D(input_size=(img_height,img_width,img_depth,img_channels),nclass=4)

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
model_checkpoint_bestLoss = ModelCheckpoint('unet3D_bestLoss.hdf5', monitor='loss',verbose=1, save_best_only=True)
model_checkpoint_bestValiLoss = ModelCheckpoint('unet3D_bestValiLoss.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
Fit = model.fit_generator(TrainGen,steps_per_epoch=Steps_per_epoch,
						validation_data=ValiGen,validation_steps=10,
						epochs=Num_epoch,
						callbacks=[model_checkpoint_bestLoss, model_checkpoint_bestValiLoss],
						verbose=2)

# save history
d_History = pd.DataFrame(Fit.history)
d_History.to_csv('history_3D.txt',sep=' ',index=False)

print("finish!")

