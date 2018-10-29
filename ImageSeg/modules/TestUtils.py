
#import cv2
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

def plot_Fit_History(historyfile, outfilename, metrics='loss'):
	fig = plt.figure(figsize=(8,6))
	dataframe = pandas.read_csv(historyfile, delim_whitespace=True, header=0)
	if metrics=='loss':
		columns_=[1,3]
		labels=['loss','val_loss']
		for i in columns_:
			d = dataframe.values[:,i]
			plt.plot(d,label = dataframe.columns[i])
	plt.legend(loc='best')
	fig.savefig(str(outfilename)+'.png')
	plt.show()

def inverse_one_hot_RGB(y):
	plottmp = np.zeros(y[:,:,0].shape + (3,))
	plottmp.astype(np.uint)
	for cat in range(0,14):
		plottmp[y[:,:,cat]>0.5,:]=COLOR_DICT[cat]
	
	return plottmp
	
def inverse_one_hot_1D(y):
	plottmp = np.zeros(y[:,:,0].shape)
	plottmp.astype(np.uint)
	for cat in range(0,14):
		plottmp[y[:,:,cat]>0.5]=cat
	
	return plottmp
	
def TestMaskPlot(y):
	plottmp = np.zeros(y[:,:,0].shape)
	plottmp.astype(np.uint)
	for cat in range(0,14):
		plottmp[y[:,:,cat]>0.5]=cat
	
	im = plt.imshow(plottmp,cmap='tab20b')
	colors = [ im.cmap(im.norm(value)) for value in range(0,14)]
	patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=Label_DICT[i]) ) for i in range(14) ]
	plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
	plt.show()
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
