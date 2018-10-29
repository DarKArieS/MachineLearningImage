import os
import numpy as np
from keras.utils import to_categorical
import copy
from random import randint
import cv2
import skimage as sk
import warnings

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

def adjustData(img,mask, nclass=14, data_Aug = False, Aug_seed = None, Do_one_hot = True):
	
	img = img.astype(np.uint8)
	mask = mask.astype(np.uint8)
	
	#Data Aug
	if(data_Aug):
		np.random.seed(Aug_seed)
		for n in range(img.shape[0]):
			Aug_choice = np.random.choice(2,5)
			(h, w) = img[n].shape[:2]
			
			# horizontal flip
			if Aug_choice[0]:
				# print("flip")
				img[n] = cv2.flip(np.array(img[n]), 1)
				mask[n,:,:,0] = cv2.flip(np.array(mask[n,:,:,0]), 1)
			
			
			# noise
			if Aug_choice[1]:
				# print("noise")
				x_aug = sk.util.random_noise(img[n])
				with warnings.catch_warnings():
					warnings.simplefilter("ignore")
					x_aug = sk.img_as_ubyte(x_aug)
				img[n] = x_aug
				
				
			# Averaging blurring
			#option: blur/GaussianBlur/medianBlur/bilateral_blur
			if Aug_choice[2]:
				# print("blur")
				random_degree = np.random.randint(1,5)
				img[n] = cv2.blur(img[n], (random_degree, random_degree))
				
			# Shift(Rolling) left/right
			# if Aug_choice[3]:
				# print("shift")
				# random_degree = np.random.randint(-10,10)
				# img[n] = np.roll(img[n], int(img[n].shape[1]*random_degree*0.01), axis = 1)
				# mask[n,:,:,0] = np.roll(mask[n,:,:,0], int(mask[n,:,:,0].shape[1]*random_degree*0.01), axis = 1)
				
			# Shift
			if Aug_choice[3]:
				# print("shift")
				random_degree = int(w*np.random.randint(-10,10)*0.01)# shift -10%~10%
				M = np.float32([[1, 0, random_degree], [0, 1, 0]])
				img[n] = cv2.warpAffine(img[n], M, (w, h))
				mask[n,:,:,0] = cv2.warpAffine(mask[n,:,:,0], M, (w, h),flags=0)
			
			# rotation
			if Aug_choice[4]:
				# print("rotate")
				center = (w / 2, h / 2)
				angle=np.random.randint(-15,15)
				scale=1.0
				M = cv2.getRotationMatrix2D(center, angle, scale)
				img[n] = cv2.warpAffine(img[n], M, (w, h))
				mask[n,:,:,0] = cv2.warpAffine(mask[n,:,:,0], M, (w, h),flags=0) #flags=0: INTER_NEAREST
			
			# Zoom in/out
			
			#Check mask
			# cout = np.unique(mask[n,:,:,0], return_counts=True)
			# print(cout)
		
	img = img.astype(float)
	img = img / 255
	
	#simplified labels
	for i,code in enumerate(sim_labels_code):
		for m in code:
			mask[mask==m]=100+i
	mask=mask-100
	
	# mask = copy.deepcopy(to_categorical(mask, nclass))
	if(Do_one_hot): mask = to_categorical(mask, nclass)
	
	# print(mask.shape)
	return(img,mask)
	

	

class TrainGenerator:
	def __init__(self, aug_dict, img_dirct, mask_dirct, img_height= 800, img_width= 500, validation_split = 0.0, test_split = 0.0):
		filename = os.listdir(mask_dirct)
		filelist=[]
		for filename_ in filename: filelist.append(os.path.splitext(filename_)[0])
		print("Total #images:" + str(len(filelist)))
		
		#Get truth list
		self.Mask_files=[]
		for filename_ in filelist: self.Mask_files.append(os.path.join(mask_dirct,filename_)+'.mat')
		
		#Get train list
		self.Img_list=[]
		for filename_ in filelist: self.Img_list.append(os.path.join(img_dirct,filename_)+'.jpg')
		
		self.img_height= img_height
		self.img_width= img_width
		
		from keras.preprocessing.image import ImageDataGenerator
		# self.mask_datagen = ImageDataGenerator(**aug_dict, channel_shift_range=0, zca_whitening=False)
		# self.img_datagen = ImageDataGenerator(**aug_dict)
		self.mask_datagen = ImageDataGenerator()
		self.img_datagen = ImageDataGenerator()
		
		self.validation_split=validation_split
		self.test_split=test_split
		if validation_split + test_split > 1.:
			raise ValueError("validation_split + test_split > 1.")
		
	def GetGenerator(self, batch_size = 10, subset=None, data_Aug = False, Aug_seed = None, Do_one_hot = True, Shuffle = True):
		seed = randint(0, 100)
		# seed = 1
		
		#calculate read range
		vali_range = int(len(self.Mask_files)*self.validation_split)
		test_range = int(len(self.Mask_files)*self.test_split)
		start = 0
		stop  = len(self.Mask_files)
		if(subset=="train"):
			start = 0
			stop  = len(self.Mask_files) - vali_range - test_range
		elif(subset=="vali"):
			if self.validation_split ==0.0: raise ValueError("no validation subset!")
			start = len(self.Mask_files) - vali_range - test_range
			stop  = len(self.Mask_files) - test_range
		elif(subset=="test"):
			if self.test_split ==0.0: raise ValueError("no test subset!")
			start = len(self.Mask_files) - test_range
			stop  = len(self.Mask_files)
		
		print('start:' + str(start) +'  stop:'+ str(stop))
		
		# read mask
		import scipy.io
		y = []
		for input_file in self.Mask_files[start:stop]:
			mat = scipy.io.loadmat(input_file)
			y_ = mat['groundtruth']
			res = cv2.resize(y_, dsize=(self.img_width,self.img_height), interpolation=cv2.INTER_NEAREST)
			y.append(res)
		
		
		# ... need to think another way to do data augmentation :(
		# Use Augmentor?
		
		y = np.asarray(y)
		y = y.reshape(y.shape[0],y.shape[1],y.shape[2],1)
		
		# read image
		x=[]
		for filename_ in self.Img_list[start:stop]:
			img = cv2.imread(filename_)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = cv2.resize(img, dsize=(self.img_width,self.img_height), interpolation=cv2.INTER_CUBIC)
			x.append(img)
			
		x = np.asarray(x)
		
		# self.img_datagen.fit(x,augment=True, rounds=3, seed=seed)
		# self.mask_datagen.fit(y,augment=True, rounds=3, seed=seed)

		# image_generator = self.img_datagen.flow(x,save_to_dir='preview/', save_prefix='img',batch_size=batch_size, shuffle=True, seed=seed)
		# mask_generator = self.mask_datagen.flow(y,save_to_dir='preview/', save_prefix='mask',batch_size=batch_size, shuffle=True, seed=seed)

		image_generator = self.img_datagen.flow(x,batch_size=batch_size, shuffle=Shuffle, seed=seed)
		mask_generator = self.mask_datagen.flow(y,batch_size=batch_size, shuffle=Shuffle, seed=seed)
		
		# image_generator = self.img_datagen.flow_from_directory(
		# img_dirct,
		# classes = ['.'],
		# class_mode = None,
		# color_mode = 'rgb',
		# target_size = (img_height,img_width),
		# batch_size = batch_size)
		
		train_generator = zip(image_generator, mask_generator)
		for (img,mask) in train_generator:
			# (img_,mask_) = copy.deepcopy((img,mask))
			img,mask = adjustData(img,mask,data_Aug=data_Aug,Aug_seed=Aug_seed,Do_one_hot=Do_one_hot)
			yield (img,mask)
	