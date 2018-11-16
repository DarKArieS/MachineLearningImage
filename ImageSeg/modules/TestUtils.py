
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import pandas

COLOR_DICT = [
       [0,     0,   0],#null
       [128,  64,   0],#hair
       [255, 196, 128],#skin
       [192, 192, 128],#acce
       [128,  64, 128],#hat
       [ 60,  40, 222],#coat
       [128, 128,   0],#shirt
       [192, 128, 128],#trousers
       [ 64,  64, 128],#dress
       [ 64,   0, 128],#skirt
       [ 64,  64,   0],#underwear
       [  0, 128, 192],#socks
       [0,100,200],#scarf/tie
	   [200,100,0]#shoes
       ]
COLOR_DICT = np.array(COLOR_DICT)

Label_DICT = ['Background','Hair','Skin','Acce','Hat','Coat','Shirt','Trousers','Dress','Skirt','Underwear','socks','scarf/tie','shoes']

# import scipy.io
# tag_list_ = scipy.io.loadmat('label_list.mat')['label_list'][0]
# tag_list = []
# for list_ in tag_list_: tag_list.append(list_[0])

def plot_Fit_History(historyfile, outfilename=None, metrics='loss'):
	fig = plt.figure(figsize=(8,6))
	dataframe = pandas.read_csv(historyfile, delim_whitespace=True, header=0)
	if metrics=='loss':
		columns_=[1,3]
		labels=['loss','val_loss']
		for i in columns_:
			d = dataframe.values[:,i]
			plt.plot(d,label = dataframe.columns[i])
	plt.legend(loc='best')
	if(outfilename):fig.savefig(str(outfilename)+'.png')
	plt.show()

def inverse_one_hot_RGB(y):
	plottmp = np.zeros(y[:,:,0].shape + (3,))
	
	for cat in range(0,14):
		plottmp[y[:,:,cat]>0.5,:]=COLOR_DICT[cat]
	
	plottmp.astype(np.uint)
	return plottmp
	
def inverse_one_hot_1D(y):
	plottmp = np.zeros(y[... ,0].shape)
	
	for cat in range(y.shape[-1]):
		plottmp[y[... ,cat]>0.5]=cat
	
	plottmp.astype(np.uint)
	return plottmp

def ImgPlot(x, outputname=None):
	fig = plt.figure(figsize=(3,4))
	plt.imshow(x)
	plt.show()
	if(outputname): fig.savefig(str(outputname)+'.png')
	
def TestMaskPlot(y, outputname=None):
	fig = plt.figure(figsize=(4.5,4))
	ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
	plottmp = np.zeros(y[:,:,0].shape)
	plottmp.astype(np.uint)
	for cat in range(0,14):
		plottmp[y[:,:,cat]>0.5]=cat
	
	im = plt.imshow(plottmp,cmap='tab20b')
	colors = [ im.cmap(im.norm(value)) for value in range(0,14)]
	patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=Label_DICT[i]) ) for i in range(14) ]
	plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
	plt.show()
	if(outputname): fig.savefig(str(outputname)+'.png')
	pass
	
	
def IoU(y,y_true, ncat=14):
	iou_scores = np.zeros(14)
	for n in range(ncat):
		intersection = np.logical_and((y[:,:,n]>0.5), (y_true[:,:,n]>0.5))
		union = np.logical_or((y[:,:,n]>0.5), (y_true[:,:,n]>0.5))
		if (np.sum(union)==0): iou_scores[n] = 0.
		else: iou_scores[n] = np.sum(intersection) / np.sum(union)
	return iou_scores

def Mean_IoU(y,y_true, ncat=14):
	iou_scores_sum = np.zeros(14)
	iou_valiEvent = np.zeros(14)
	
	for evt in range(y.shape[0]):
		for n in range(ncat):
			intersection = np.logical_and((y[evt,:,:,n]>0.5), (y_true[evt,:,:,n]>0.5))
			union = np.logical_or((y[evt,:,:,n]>0.5), (y_true[evt,:,:,n]>0.5))
			if (np.sum(union)!=0):
				iou_scores_sum[n] += np.sum(intersection) / np.sum(union)
				iou_valiEvent[n]+=1
	
	print('sum of IoU:')
	print (["{0:0.3f}".format(i) for i in iou_scores_sum])
	print('Valid #events of each class:')
	print(iou_valiEvent.astype(np.uint))
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		return iou_scores_sum/iou_valiEvent
	
def Precision(y,y_true, ncat=14):
	pixel_acc = np.zeros(14)
	for n in range(ncat):
		intersection = np.logical_and((y[:,:,n]>0.5), (y_true[:,:,n]>0.5))
		mask_sum = np.sum((y_true[:,:,n]>0.5))
		if (mask_sum ==0): pixel_acc[n] = 0.
		else: pixel_acc[n] = np.sum(intersection) / mask_sum
	return pixel_acc
	
	
def Mean_Precision(y,y_true, ncat=14):
	pixel_acc_sum = np.zeros(14)
	pixel_acc_valiEvent = np.zeros(14)
	
	for evt in range(y.shape[0]):
		for n in range(ncat):
			intersection = np.logical_and((y[evt,:,:,n]>0.5), (y_true[evt,:,:,n]>0.5))
			mask_sum = np.sum((y_true[evt,:,:,n]>0.5))
			if (mask_sum!=0):
				pixel_acc_sum[n] += np.sum(intersection) / mask_sum
				pixel_acc_valiEvent[n]+=1
	
	print('sum of precision:')
	print (["{0:0.3f}".format(i) for i in pixel_acc_sum])
	print('Valid #events of each class:')
	print(pixel_acc_valiEvent.astype(np.uint))
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		return pixel_acc_sum/pixel_acc_valiEvent

def binary_dice3d(s,g):
	# code from: https://github.com/tkuanlun350/3DUnet-Tensorflow-Brats18
    """
    dice score of 3d binary volumes
    inputs: 
        s: segmentation volume
        g: ground truth volume
    outputs:
        dice: the dice score
    """
    assert(len(s.shape)==3)
    [Ds, Hs, Ws] = s.shape
    [Dg, Hg, Wg] = g.shape
    assert(Ds==Dg and Hs==Hg and Ws==Wg)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0*s0 + 1e-10)/(s1 + s2 + 1e-10)
    return dice
		
def brats_dice_score(y, y_truth, tumor_type):
	# input: (x,y,z)
	# 0: whole tumor (WT): labeled region			(label >  0)
	# 1: core region (TC): except "edema"  			(label >  1)
	# 2: active region (ET): only enhancing core 	(label == 3)
	if tumor_type == 0:
		s = y > 0
		g = y_truth > 0
	elif tumor_type == 1:
		s = y > 1
		g = y_truth > 1
	elif tumor_type == 2:
		s = y ==3
		g = y_truth ==3
	else:
		return 0
		
	return binary_dice3d(s,g)

import imageio
def brats_generate_gif(output, y, axis = 0):
	assert(len(y.shape)==3)
	input_shape = y.shape
	
	imgs_2D = []
	for frame in range(input_shape[axis]):
		if axis==0 : img = y[frame , ...]
		if axis==1 : img = y[:,frame,:]
		if axis==2 : img = y[... , frame]
		
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = img.astype(np.uint8)
		imgs_2D.append(img)
		
	imageio.mimsave( output + '.gif', imgs_2D)
	pass


	