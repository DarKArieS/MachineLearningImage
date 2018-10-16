


def Load_dataset(DatasetName):
	if DatasetName == "MNIST":
		# ( , 28, 28), Color_depth = 0~255
		print('Load MNIST from KERAS')
		from keras.datasets import mnist
		return mnist.load_data()
		
		
	if DatasetName == "CIFAR10":
		# ( , 32, 32, 3), Color_depth = 0~255
		print('Load CIFAR10 from KERAS')
		from keras.datasets import cifar10
		return cifar10.load_data()



