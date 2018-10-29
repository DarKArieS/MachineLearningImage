import json
import argparse
import numpy as np
import pandas as pd
import keras
from keras.models import load_model

from modules.DataGenerator import *
from modules.TestUtils import *

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
Dataset_vali_split = Params['Dataset']['vali_split']
Dataset_test_split = Params['Dataset']['test_split']
Data_Aug = Params['Data_Aug']['train_used']
Model_name = Params['Model']['name']
img_height = Params['Model']['img_height']
img_width = Params['Model']['img_width']
Num_epoch = Params['Num_epoch']
Steps_per_epoch = Params['Steps_per_epoch']
Batch_size = Params['Batch_size']

Load_Model = 'models/181028/unet.hdf5'
Load_history = 'models/181028/history.txt'
# Keras Data Aug opt, not used
data_gen_args = dict()

# Plot training history
plot_Fit_History(Load_history,'history.png')

# Read test dataset
Gen = TrainGenerator(data_gen_args,Dataset_img_dirct,Dataset_mask_dirct,img_height,img_width,Dataset_vali_split,Dataset_test_split)
TrainGen = Gen.GetGenerator(10,subset="train", Shuffle=False)
ValiGen = Gen.GetGenerator(10,subset="vali", Shuffle=False)
TestGen = Gen.GetGenerator(10,subset="test", Shuffle=False)

TestDataset = next(TestGen)
x=TestDataset[0]
y_true=TestDataset[1]

#Get prediction
model = load_model(Load_Model)
y = model.predict(x, verbose = 1)

#Print some example
import matplotlib.pyplot as plt
plt.imshow(x[0])
plt.show()
TestMaskPlot(y_true[0])    
TestMaskPlot(y[0])
iou = IoU(y[0],y_true[0])
acc = Precision(y[0],y_true[0])
print('IoU:')
print (['{0:0.3f}'.format(i) for i in iou])
print('precision:')
print (['{0:0.3f}'.format(i) for i in acc])
print('')
avg_IoU = Mean_IoU(y,y_true)
avg_acc = Mean_Precision(y,y_true)
print('mean IoU:')
print (['{0:0.3f}'.format(i) for i in avg_IoU])
print('mean precision:')
print (['{0:0.3f}'.format(i) for i in avg_acc])