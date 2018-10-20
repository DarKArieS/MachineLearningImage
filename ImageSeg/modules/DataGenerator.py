import os
import numpy as np
from keras.utils import to_categorical
import copy

sim_labels_code=[
        {0},#null
        {19},#hair
        {41},#skin
        {1,2,3,9,10,15,17,18,33,34,47,56,57},#acce
        {20},#hat
        {11,13,24,55},#coat
        {4,5,22,26,38,46,48,49,51,54},#shirt
        {25,27,31,40},#trousers
        {14,35},#dress
        {42},#skirt
        {6,8,23,30,50},#underwear
        {44,45,53},#socks
        {29,37,52},#scarf/tie
        {7,12,16,21,28,32,36,39,43,58}#shoes
        ]

def adjustData(img,mask, nclass=14):
	img = img / 255
	
	#simplified labels
	for i,code in enumerate(sim_labels_code):
		for m in code:
			mask[mask==m]=100+i
	mask=mask-100
	
	# mask = copy.deepcopy(to_categorical(mask, nclass))
	mask = copy.deepcopy(to_categorical(mask, nclass))
	
	print(mask.shape)
	return(img,mask)
	

def trainGenerator(img_dirct, mask_dirct, img_height= 800, img_width= 500):
	#Get filename list from truth
	filename = os.listdir(mask_dirct)
	filelist=[]
	for filename_ in filename: filelist.append(os.path.splitext(filename_)[0])
	print("#images:" + str(len(filelist)))
	
	#Get truth list
	Mask_files=[]
	for filename_ in filelist: Mask_files.append(os.path.join(mask_dirct,filename_)+'.mat')
	
	#Get train list
	Img_list=[]
	for filename_ in filelist: Img_list.append(os.path.join(img_dirct,filename_)+'.jpg')

	# read mask
	import scipy.io
	import cv2
	y = []
	for input_file in Mask_files:
		mat = scipy.io.loadmat(input_file)
		y_ = mat['groundtruth']
		res = cv2.resize(y_, dsize=(img_width,img_height), interpolation=cv2.INTER_NEAREST)
		y.append(res)
		
	# read image
	from keras.preprocessing.image import ImageDataGenerator
	mask_datagen = ImageDataGenerator()
	
	image_generator = mask_datagen.flow_from_directory(
	img_dirct,
	classes = ['.'],
	class_mode = None,
	color_mode = 'rgb',
	target_size = (img_height,img_width),
	batch_size = 1)
	
	train_generator = zip(image_generator, y)
	for (img,mask) in train_generator:
		img,mask = adjustData(img,mask)
		yield (img[0],mask)
	
	