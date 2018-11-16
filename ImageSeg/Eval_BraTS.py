import json
import argparse
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
# limit GPU memory
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)
import keras.backend.tensorflow_backend as KTF
KTF.set_session(sess)

from modules.DataGenerator import *
from modules.TestUtils import *
from modules.utils_3Dimg import *

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

Load_Model = 'Brain_181116_ValiFailed/unet3D_bestValiLoss.hdf5'
# Load_Model = '181029/unet_bestLoss.hdf5'
model = load_model(Load_Model)

# Plot training history
Load_history = 'Brain_181116_ValiFailed/history_3D.txt'
plot_Fit_History(Load_history,'history')

img_size=(img_height,img_width,img_depth)

# Read test dataset
Gen = NIFIT_Generator(Dataset_img_dirct,Dataset_mask_dirct,img_size=(img_height,img_width,img_depth),validation_split=0.1,test_split=0.1)
TrainGen = Gen.GetGenerator(1,subset="train", Shuffle=False, Do_one_hot = False, do_corp_and_roi=False)
ValiGen = Gen.GetGenerator(1,subset="vali", Shuffle=False, Do_one_hot = False, do_corp_and_roi=False)
TestGen = Gen.GetGenerator(1,subset="test", Shuffle=False, Do_one_hot = False, do_corp_and_roi=False)

def corp_all_channels(img, bbmin, bbmax):
    reshape_size = (bbmax[0]-bbmin[0]+1,bbmax[1]-bbmin[1]+1,bbmax[2]-bbmin[2]+1)
    img_ = np.zeros(reshape_size+ (4,))
    for mod in range(img.shape[3]):
        img__ = crop_ND_volume_with_bounding_box(img[... , mod], bbmin, bbmax)
        img_[... , mod] = img__
    
    return img_

# start processes
dice_WTs=[]
dice_TCs=[]
dice_ETs=[]

if not os.path.exists('GIF'): os.makedirs('GIF')

TestDataset = next(TestGen)
Dataset_len = Gen.test_len
print("predicting...")
for idx in range(1,Dataset_len+1):
    img  = TestDataset[0][0]
    mask = TestDataset[1][0]
    
    # corp
    margin = 5
    bbmin, bbmax = get_none_zero_region(img[... , 0], margin)
    ori_shape = (img_channels,)+(bbmax[0]-bbmin[0]+1,bbmax[1]-bbmin[1]+1,bbmax[2]-bbmin[2]+1)
    
    img = corp_all_channels(img,bbmin,bbmax)
    mask = crop_ND_volume_with_bounding_box(mask[..., 0], bbmin, bbmax)
    
    #get patches
    all_cps = slice_center_points(img,img_size)
    sliced_img = eval_slice_patches_cps(img,img_size, all_cps)
    sliced_img = np.asarray(sliced_img)
    
    #get prediction
    test_set = np.transpose(sliced_img, [0,2,3,4,1])
    pred = model.predict(test_set, batch_size = 1,  verbose=1)
    pred = np.transpose(pred, [0,4,1,2,3])
    
    #merge patches
    pred = eval_merge_patches(pred, all_cps, ori_shape)
    pred = np.transpose(pred, [1,2,3,0])
    pred = inverse_one_hot_1D(pred)
    
    dice_WT = brats_dice_score(mask, pred, 0)
    dice_TC = brats_dice_score(mask, pred, 1)
    dice_ET = brats_dice_score(mask, pred, 2)
    
    dice_WTs.append(dice_WT)
    dice_TCs.append(dice_TC)
    dice_ETs.append(dice_ET)
    
    brats_generate_gif('GIF/img'+str(idx),img[...,0]*255, axis = 0)
    brats_generate_gif('GIF/predict'+str(idx),pred*50, axis = 0)
    brats_generate_gif('GIF/truth'+str(idx),mask*50, axis = 0)
    
    TestDataset = next(TestGen)

def get_avg(dice_list):
    sum_ = 0
    for idx in range(len(dice_list)):
        sum_ += dice_list[idx]
    return sum_/len(dice_list)

print('average dice_WTs: {0:0.3f}'.format(get_avg(dice_WTs)))
print('average dice_TCs: {0:0.3f}'.format(get_avg(dice_TCs)))
print('average dice_ETs: {0:0.3f}'.format(get_avg(dice_ETs)))
